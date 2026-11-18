"""A finished job that cannot be processed is retried or recorded, never dropped."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from eval_integrity_utils import (
    StaticScheduler,
    children,
    good_results,
    make_job,
    make_runner,
    open_database,
)
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.utils import load_results


def _enable_retry_loop(runner: ShinkaEvolveRunner) -> None:
    """Stub the progress bookkeeping that `_retry_failed_db_jobs` reports to."""
    runner._update_completed_generations = lambda: asyncio.sleep(0)
    runner._log_wandb_population_progress = lambda: None
    runner._record_progress = lambda: None
    runner.slot_available = asyncio.Event()


def test_results_timeout_is_retried_on_the_same_read(tmp_path: Path) -> None:
    """A retry waits for the slow read: it must not tie up another scheduler thread."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            results_ready = asyncio.Event()

            async def slow_read() -> dict[str, Any]:
                await results_ready.wait()
                return good_results(2.5)

            scheduler = StaticScheduler(slow_read)
            runner = make_runner(async_db, scheduler, max_retries=5)
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.05
            _enable_retry_loop(runner)
            job = make_job()

            await runner._process_completed_jobs_safely([job])
            await runner._retry_failed_db_jobs()

            assert children(sync_db) == []
            assert runner.failed_jobs_for_retry == {"job-1": job}
            assert job.db_retry_count == 2
            assert scheduler.calls == 1

            results_ready.set()
            await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.correct == 1 and child.combined_score == 2.5
            assert runner.failed_jobs_for_retry == {}
            assert scheduler.calls == 1
            assert runner._result_reads == {}

    asyncio.run(run())


def test_results_that_never_arrive_become_a_failed_row(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            never = asyncio.Event()
            scheduler = StaticScheduler(never.wait)
            runner = make_runner(async_db, scheduler, max_retries=2)
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.05
            _enable_retry_loop(runner)
            job = make_job(generation=6)

            await runner._process_completed_jobs_safely([job])
            await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.generation == 6 and child.correct == 0
            assert child.metadata["results_missing"] is True
            assert "timed out after 0.05s" in child.text_feedback
            assert runner.failed_jobs_for_retry == {}
            assert scheduler.calls == 1
            runner._result_reads.pop(runner._source_job_key(job)).cancel()

    asyncio.run(run())


def test_unreadable_results_become_a_failed_row_once_retries_run_out(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(
                async_db, StaticScheduler(OSError("stale file handle")), max_retries=3
            )
            _enable_retry_loop(runner)
            job = make_job(generation=4)

            await runner._process_completed_jobs_safely([job])
            for _ in range(2):
                assert children(sync_db) == []
                await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.generation == 4
            assert child.correct == 0 and child.combined_score == 0.0
            assert child.metadata["results_missing"] is True
            assert child.metadata["evaluation_failed"] is True
            assert "stale file handle" in child.text_feedback
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_truncated_correctness_file_is_recorded_without_retries(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "metrics.json").write_text('{"combined_score": 3.0}')
    (results_dir / "correct.json").write_text('{"correct": tr')

    class LoadingScheduler:
        async def get_job_results_async(self, job_id: Any, directory: str) -> Any:
            return load_results(directory)

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, LoadingScheduler())
            job = make_job(results_dir=str(results_dir))

            assert await runner._process_single_job_safely(job)

            (child,) = children(sync_db)
            assert child.correct == 0 and child.combined_score == 0.0
            assert "Could not decode JSON" in child.text_feedback
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_failed_maintenance_is_retried_on_the_stored_row(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results(2.5)))
            _enable_retry_loop(runner)
            job = make_job()
            maintain = async_db.run_program_maintenance_async
            attempts = 0

            async def flaky_maintenance(program: Any, **kwargs: Any) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("database is locked")
                await maintain(program, **kwargs)

            async_db.run_program_maintenance_async = flaky_maintenance

            await runner._process_completed_jobs_safely([job])

            (child,) = children(sync_db)
            assert child.metadata["postprocess_db_maintenance_applied"] is False
            assert sync_db.get_best_program().id != child.id
            assert runner.failed_jobs_for_retry == {"job-1": job}

            await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.metadata["postprocess_db_maintenance_applied"] is True
            assert child.metadata["postprocess_side_effects_applied"] is True
            assert sync_db.get_best_program().id == child.id
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_failure_after_maintenance_does_not_replay_side_effects(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results(2.5)))
            runner.evo_config.evolve_prompts = True
            updates = 0

            async def failing_prompt_update(*args: Any, **kwargs: Any) -> None:
                nonlocal updates
                updates += 1
                raise RuntimeError("prompt database is locked")

            runner._update_prompt_fitness = failing_prompt_update
            job = make_job(meta_patch_data={"system_prompt_id": "prompt-1"})

            await runner._process_completed_jobs_safely([job])

            (child,) = children(sync_db)
            assert child.metadata["postprocess_db_maintenance_applied"] is True
            assert sync_db.get_best_program().id == child.id
            assert runner.failed_jobs_for_retry == {}
            assert updates == 1

    asyncio.run(run())


def test_unexpected_persistence_error_queues_a_retry(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results(2.5)))
            _enable_retry_loop(runner)
            job = make_job()
            read_file = runner._read_file_async
            reads = 0

            async def flaky_read(path: str) -> Any:
                nonlocal reads
                reads += 1
                if reads == 1:
                    raise RuntimeError("executor shut down")
                return await read_file(path)

            runner._read_file_async = flaky_read

            await runner._process_completed_jobs_safely([job])

            assert children(sync_db) == []
            assert runner.failed_jobs_for_retry == {"job-1": job}

            await runner._retry_failed_db_jobs()

            assert len(children(sync_db)) == 1
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_job_that_cannot_be_persisted_is_recorded_once_retries_run_out(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(
                async_db, StaticScheduler(good_results(2.5)), max_retries=3
            )
            _enable_retry_loop(runner)

            async def unreadable(path: str) -> Any:
                raise RuntimeError("cannot read program file")

            runner._read_file_async = unreadable
            job = make_job(generation=2, meta_patch_data={"api_costs": 0.25})

            await runner._process_completed_jobs_safely([job])
            for _ in range(2):
                assert children(sync_db) == []
                await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.generation == 2 and child.code == ""
            assert child.correct == 0 and child.combined_score == 0.0
            assert child.metadata["evaluation_failed"] is True
            assert child.metadata["api_costs"] == 0.25
            assert "cannot read program file" in child.text_feedback
            assert "3 attempts" in child.text_feedback
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_row_whose_maintenance_keeps_failing_is_not_recorded_twice(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(
                async_db, StaticScheduler(good_results(2.5)), max_retries=2
            )
            _enable_retry_loop(runner)

            async def broken_maintenance(program: Any, **kwargs: Any) -> None:
                raise RuntimeError("database is locked")

            async_db.run_program_maintenance_async = broken_maintenance

            await runner._process_completed_jobs_safely([make_job()])
            await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.correct == 1 and child.combined_score == 2.5
            assert child.metadata["postprocess_db_maintenance_applied"] is False
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_job_is_given_up_without_raising_when_the_database_stays_down(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(
                async_db, StaticScheduler(good_results(2.5)), max_retries=2
            )
            _enable_retry_loop(runner)
            add_program = async_db.add_program_async
            attempts = 0

            async def database_down(program: Any, **kwargs: Any) -> bool:
                nonlocal attempts
                attempts += 1
                raise RuntimeError("disk I/O error")

            async_db.add_program_async = database_down

            await runner._process_completed_jobs_safely([make_job()])
            await runner._retry_failed_db_jobs()
            async_db.add_program_async = add_program

            assert children(sync_db) == []
            assert attempts == 3  # two attempts and the last-resort failed row
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["results-never-arrive", "program-unreadable"])
def test_job_out_of_retries_counts_as_in_flight_until_its_row_is_written(
    tmp_path: Path, failure: str
) -> None:
    """The run must not be declared finished while the last write is under way."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            never = asyncio.Event()
            scheduler = StaticScheduler(
                never.wait if failure == "results-never-arrive" else good_results()
            )
            runner = make_runner(async_db, scheduler, max_retries=2)
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.05
            _enable_retry_loop(runner)
            runner.running_jobs = []
            runner.active_proposal_tasks = {}
            if failure == "program-unreadable":

                async def unreadable(path: str) -> Any:
                    raise RuntimeError("cannot read program file")

                runner._read_file_async = unreadable
            add_program = async_db.add_program_async
            in_flight_during_write = []

            async def observing_add(program: Any, **kwargs: Any) -> bool:
                in_flight_during_write.append(runner._get_in_flight_work_count())
                return await add_program(program, **kwargs)

            async_db.add_program_async = observing_add

            await runner._process_completed_jobs_safely([make_job()])
            await runner._retry_failed_db_jobs()
            for read in runner._result_reads.values():
                read.cancel()

            assert len(children(sync_db)) == 1
            assert in_flight_during_write == [1]
            assert runner._get_in_flight_work_count() == 0

    asyncio.run(run())


def test_exception_escaping_job_processing_queues_a_retry() -> None:
    async def run() -> None:
        runner = make_runner(None, StaticScheduler())
        job = make_job()

        async def explode(_job: Any) -> bool:
            raise RuntimeError("boom")

        runner._process_single_job_safely = explode

        await runner._process_completed_jobs_safely([job])

        assert runner.failed_jobs_for_retry == {"job-1": job}

    asyncio.run(run())
