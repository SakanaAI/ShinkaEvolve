"""A results read still in progress is not abandoned when the run shuts down."""

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
from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner


def _at_cost_limit_with_one_finished_job(
    runner: ShinkaEvolveRunner, job: AsyncRunningJob
) -> None:
    """Arrange the state the job monitor sees when the budget runs out."""

    async def no_job_running(jobs: list[AsyncRunningJob]) -> list[bool]:
        return [False for _ in jobs]

    runner.scheduler.batch_check_status_async = no_job_running
    runner.running_jobs = [job]
    runner.active_proposal_tasks = {}
    runner.should_stop = asyncio.Event()
    runner.finalization_complete = asyncio.Event()
    runner.slot_available = asyncio.Event()
    runner.processing_lock = asyncio.Lock()
    runner.completed_generations = 0
    runner.evo_config.num_generations = 5
    runner.evo_config.max_api_costs = 1.0
    runner.total_api_cost = 2.0
    runner.cost_limit_reached = False
    runner._update_completed_generations = lambda: asyncio.sleep(0)
    runner._log_wandb_population_progress = lambda: None
    runner._record_progress = lambda: None


@pytest.mark.parametrize("results_arrive", [True, False])
def test_cost_limit_shutdown_waits_for_a_results_read_in_progress(
    tmp_path: Path, results_arrive: bool
) -> None:
    """The last job's result is stored, or recorded as failed, before the run stops."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):

            async def slow_read() -> dict[str, Any]:
                # Longer than one read timeout, and than the whole wait if it never arrives.
                await asyncio.sleep(0.3 if results_arrive else 60.0)
                return good_results(2.5)

            runner = make_runner(async_db, StaticScheduler(slow_read), max_retries=5)
            runner._RESULTS_READ_TIMEOUT_SECONDS = 0.1
            _at_cost_limit_with_one_finished_job(runner, make_job())

            await asyncio.wait_for(runner._job_monitor_task(), timeout=20.0)

            assert runner.should_stop.is_set()
            (child,) = children(sync_db)
            assert bool(child.correct) is results_arrive
            assert child.combined_score == (2.5 if results_arrive else 0.0)
            assert runner.failed_jobs_for_retry == {}
            assert runner._result_reads == {}

    asyncio.run(run())
