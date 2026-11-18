"""A scheduler handle that repeats must not make one job stand in for another."""

import asyncio
from pathlib import Path

from eval_integrity_utils import (
    SEED_ID,
    StaticScheduler,
    children,
    good_results,
    make_job,
    make_runner,
    open_database,
)
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import Program


class LocalHandle:
    """Stands in for ProcessWithLogging, which is named by its PID."""

    returncode = 0

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def __str__(self) -> str:
        return f"ProcessWithLogging(PID: {self.pid})"


def test_reused_pid_in_a_later_session_gets_its_own_row(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (_, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results(1.0)))
            first = make_job(job_id=LocalHandle(500), generation=1)
            assert await runner._process_single_job_safely(first)

        # A resumed run starts a fresh process table and is handed PID 500 again.
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results(9.0)))
            second = make_job(job_id=LocalHandle(500), generation=2)
            assert await runner._process_single_job_safely(second)

            scores = {child.generation: child.combined_score for child in children(sync_db)}
            assert scores == {1: 1.0, 2: 9.0}

    asyncio.run(run())


def test_row_stored_under_the_old_key_format_does_not_block_a_new_job(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            sync_db.add(
                Program(
                    id="old-row",
                    code="x = 1\n",
                    generation=1,
                    parent_id=SEED_ID,
                    metadata={"source_job_id": "ProcessWithLogging(PID: 500)"},
                )
            )
            runner = make_runner(async_db, StaticScheduler(good_results(9.0)))

            for generation in (1, 2):
                job = make_job(job_id=LocalHandle(500), generation=generation)
                assert await runner._process_single_job_safely(job)

            assert sorted(child.generation for child in children(sync_db)) == [1, 1, 2]
            assert sync_db.get("old-row") is not None

    asyncio.run(run())


def test_retry_of_one_job_still_reuses_its_row(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results()))
            job = make_job(job_id=LocalHandle(500))

            assert await runner._process_single_job_safely(job)
            assert await runner._process_single_job_safely(job)

            assert len(children(sync_db)) == 1

    asyncio.run(run())


def test_jobs_sharing_a_handle_each_keep_their_place_in_the_retry_queue(
    tmp_path: Path,
) -> None:
    """A PID reused while an earlier job still waits for a retry must not evict it."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            stale = OSError("stale file handle")
            runner = make_runner(
                async_db, StaticScheduler(stale, stale, good_results()), max_retries=3
            )
            runner._update_completed_generations = lambda: asyncio.sleep(0)
            runner._log_wandb_population_progress = lambda: None
            runner._record_progress = lambda: None
            runner.slot_available = asyncio.Event()
            earlier = make_job(job_id=LocalHandle(500), generation=3)
            later = make_job(job_id=LocalHandle(500), generation=7)

            await runner._process_completed_jobs_safely([earlier, later])

            assert list(runner.failed_jobs_for_retry.values()) == [earlier, later]

            await runner._retry_failed_db_jobs()

            assert sorted(child.generation for child in children(sync_db)) == [3, 7]
            assert runner.failed_jobs_for_retry == {}

    asyncio.run(run())


def test_key_separates_jobs_and_is_stable_for_one_job() -> None:
    job = make_job(job_id=LocalHandle(500), generation=3)
    same_handle_later = make_job(
        job_id=LocalHandle(500),
        generation=3,
        evaluation_submitted_at=job.evaluation_submitted_at + 60.0,
    )
    other_generation = make_job(
        job_id=LocalHandle(500),
        generation=4,
        evaluation_submitted_at=job.evaluation_submitted_at,
    )

    key = ShinkaEvolveRunner._source_job_key(job)
    job.db_retry_count += 1

    assert ShinkaEvolveRunner._source_job_key(job) == key
    assert key.startswith("ProcessWithLogging(PID: 500)")
    assert len(
        {key}
        | {ShinkaEvolveRunner._source_job_key(j) for j in (same_handle_later, other_generation)}
    ) == 3


def test_database_duplicate_check_requires_the_generation_to_match(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            for program_id, generation in (("a", 1), ("b", 2), ("c", 2)):
                await async_db.add_program_async(
                    Program(
                        id=program_id,
                        code="x = 1\n",
                        generation=generation,
                        metadata={"source_job_id": "job-7"},
                    )
                )

            assert sync_db.get("a") is not None
            assert sync_db.get("b") is not None
            assert sync_db.get("c") is None  # same job id and generation as "b"
            assert sync_db.has_program_with_source_job_id("job-7")
            assert sync_db.has_program_with_source_job_id("job-7", generation=2)
            assert not sync_db.has_program_with_source_job_id("job-7", generation=3)

    asyncio.run(run())


def test_row_of_another_generation_is_never_reused() -> None:
    """Even if a lookup returns such a row, the job is retried, not reported done."""

    class MismatchedRowDB:
        async def add_program_async(self, program: Program, **kwargs: object) -> bool:
            return False

        async def get_program_by_source_job_id_async(self, source_job_id: str) -> Program:
            return Program(id="other", code="", generation=1)

    async def run() -> None:
        runner = make_runner(MismatchedRowDB(), StaticScheduler(good_results()))
        job = make_job(job_id="job-7", generation=2)

        assert not await runner._process_single_job_safely(job)
        assert list(runner.failed_jobs_for_retry.values()) == [job]

    asyncio.run(run())
