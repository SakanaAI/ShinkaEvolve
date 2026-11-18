"""A finished job that cannot be processed is retried or recorded, never dropped."""

import asyncio
from pathlib import Path
from typing import Any

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


def test_results_timeout_is_retried_until_the_results_arrive(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            scheduler = StaticScheduler(asyncio.TimeoutError(), good_results(2.5))
            runner = make_runner(async_db, scheduler)
            _enable_retry_loop(runner)
            job = make_job()

            await runner._process_completed_jobs_safely([job])

            assert children(sync_db) == []
            assert runner.failed_jobs_for_retry == {"job-1": job}

            await runner._retry_failed_db_jobs()

            (child,) = children(sync_db)
            assert child.correct == 1 and child.combined_score == 2.5
            assert runner.failed_jobs_for_retry == {}

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
