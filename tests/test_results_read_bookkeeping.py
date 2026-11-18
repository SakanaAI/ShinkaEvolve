"""Results reads are timed, used and let go of correctly around the retry queue."""

import asyncio
import time
from pathlib import Path

from eval_integrity_utils import (
    StaticScheduler,
    children,
    good_results,
    make_job,
    make_runner,
    open_database,
)


def test_job_behind_a_slow_read_is_not_charged_for_the_wait(tmp_path: Path) -> None:
    """Jobs found finished in one poll finished then, whatever order they are read in."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            never = asyncio.Event()
            runner = make_runner(
                async_db, StaticScheduler(never.wait, good_results(2.5)), max_retries=5
            )
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.3
            stuck = make_job(job_id="job-1", generation=1)
            behind = make_job(job_id="job-2", generation=2)

            batch_started_at = time.time()
            await runner._process_completed_jobs_safely([stuck, behind])
            for read in runner._result_reads.values():
                read.cancel()

            (child,) = children(sync_db)
            assert child.generation == 2
            assert child.metadata["evaluation_finished_at"] - batch_started_at < 0.15

    asyncio.run(run())


def test_result_that_arrives_as_the_wait_runs_out_is_used(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            scheduler = StaticScheduler(good_results(2.5))
            runner = make_runner(async_db, scheduler, max_retries=1)
            job = make_job()
            # A read left running by an earlier pass that is about to finish:
            # this pass does not wait for it, yet finds it done.
            runner._result_reads = {
                runner._source_job_key(job): asyncio.ensure_future(
                    scheduler.get_job_results_async(job.job_id, job.results_dir)
                )
            }
            job.results_wait_deadline = time.monotonic() + 60.0

            assert await runner._process_single_job_safely(job)

            (child,) = children(sync_db)
            assert child.correct == 1 and child.combined_score == 2.5
            assert scheduler.calls == 1

    asyncio.run(run())


def test_dropping_queued_jobs_at_the_target_lets_go_of_their_reads(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            never = asyncio.Event()
            runner = make_runner(async_db, StaticScheduler(never.wait), max_retries=5)
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.05
            runner.running_jobs = []
            runner.active_proposal_tasks = {}
            runner.slot_available = asyncio.Event()
            job = make_job()

            await runner._process_completed_jobs_safely([job])
            (read,) = runner._result_reads.values()
            await runner._cancel_surplus_inflight_work()
            await asyncio.sleep(0)

            assert runner.failed_jobs_for_retry == {}
            assert runner._result_reads == {}
            assert read.cancelled()
            assert children(sync_db) == []

    asyncio.run(run())
