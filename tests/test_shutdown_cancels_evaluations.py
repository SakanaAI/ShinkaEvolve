import asyncio
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.core.runtime_slots import LogicalSlotPool
from shinka.launch import JobScheduler, LocalJobConfig, ProcessWithLogging


def _job(job_id: str, generation: int) -> AsyncRunningJob:
    now = time.time()
    return AsyncRunningJob(
        job_id=job_id,
        exec_fname=f"gen_{generation}/main.py",
        results_dir=f"gen_{generation}/results",
        start_time=now,
        proposal_started_at=now,
        evaluation_submitted_at=now,
        generation=generation,
    )


class _Scheduler:
    def __init__(self, events: list[str], failing: Optional[str] = None) -> None:
        self.events = events
        self.failing = failing

    async def cancel_job_async(self, job_id: str) -> bool:
        self.events.append(f"cancel:{job_id}")
        if job_id == self.failing:
            raise OSError("scancel unavailable")
        return True

    def shutdown(self) -> None:
        self.events.append("scheduler.shutdown")


def _cleanup_runner(events: list[str], scheduler: _Scheduler) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.active_proposal_tasks = {}
    runner.prompt_db = None
    runner.db = None
    runner.scheduler = scheduler

    async def record(event: str) -> None:
        events.append(event)

    runner._flush_unwritten_charges = lambda: record("flush_charges")
    runner._finish_wandb_logging = lambda: record("finish_wandb")
    runner.async_db = SimpleNamespace(close_async=lambda: record("close_db"))
    return runner


def test_cleanup_cancels_evaluations_that_are_still_running() -> None:
    events: list[str] = []
    runner = _cleanup_runner(events, _Scheduler(events))
    runner.running_jobs = [_job("101", 1), _job("102", 2)]

    asyncio.run(runner._cleanup_async())

    assert events == [
        "cancel:101",
        "cancel:102",
        "flush_charges",
        "finish_wandb",
        "close_db",
        "scheduler.shutdown",
    ]


def test_cleanup_finishes_when_a_cancellation_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    runner = _cleanup_runner(events, _Scheduler(events, failing="101"))
    runner.running_jobs = [_job("101", 1), _job("102", 2)]

    with caplog.at_level(logging.WARNING):
        asyncio.run(runner._cleanup_async())

    assert events[:2] == ["cancel:101", "cancel:102"]
    assert events[2:] == ["flush_charges", "finish_wandb", "close_db", "scheduler.shutdown"]
    assert "Could not cancel 1 evaluation job(s)" in caplog.text
    assert "['101']" in caplog.text


def test_cleanup_without_running_evaluations_cancels_nothing() -> None:
    events: list[str] = []
    runner = _cleanup_runner(events, _Scheduler(events))
    runner.running_jobs = []

    asyncio.run(runner._cleanup_async())

    assert not any(event.startswith("cancel:") for event in events)


class _SlowSubmission:
    """A scheduler whose submission finishes only when the test lets it."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled: list[str] = []

    async def submit_async_nonblocking(self, exec_fname: str, results_dir: str) -> str:
        self.started.set()
        await self.release.wait()
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def cancel_job_async(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True


def _submitting_runner(scheduler: Any) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.scheduler = scheduler
    runner.evaluation_slot_pool = LogicalSlotPool(1, "evaluation")
    runner.sampling_slot_pool = LogicalSlotPool(1, "sampling")
    return runner


@pytest.mark.parametrize("cancellations", [1, 2], ids=["cancelled", "cancelled-twice"])
@pytest.mark.parametrize(
    ("outcome", "cancelled"),
    [("job-9", ["job-9"]), (OSError("sbatch failed"), [])],
    ids=["job-launched", "submission-failed"],
)
def test_proposal_cancelled_during_submission_leaves_no_untracked_job(
    outcome: Any, cancelled: list[str], cancellations: int
) -> None:
    async def run() -> None:
        scheduler = _SlowSubmission(outcome)
        runner = _submitting_runner(scheduler)
        proposal = asyncio.create_task(
            runner._submit_evaluation_job_with_slot("main.py", "results", None)
        )
        await asyncio.wait_for(scheduler.started.wait(), timeout=2.0)
        assert runner.evaluation_slot_pool.in_use == 1

        # Shutdown cancels proposal tasks once when it starts and again in cleanup.
        for _ in range(cancellations):
            proposal.cancel()
            await asyncio.sleep(0)
            # The worker thread is still inside sbatch/Popen: it cannot be stopped.
            assert not proposal.done()
        scheduler.release.set()

        with pytest.raises(asyncio.CancelledError):
            await proposal
        assert scheduler.cancelled == cancelled
        assert runner.evaluation_slot_pool.in_use == 0

    asyncio.run(run())


def test_failed_submission_still_releases_its_slot() -> None:
    async def run() -> None:
        scheduler = _SlowSubmission(OSError("sbatch failed"))
        scheduler.release.set()
        runner = _submitting_runner(scheduler)

        with pytest.raises(OSError, match="sbatch failed"):
            await runner._submit_evaluation_job_with_slot("main.py", "results", None)

        assert scheduler.cancelled == []
        assert runner.evaluation_slot_pool.in_use == 0

    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_evaluator_launched_after_its_proposal_was_cancelled_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("import time\ntime.sleep(30)\n")
    scheduler = JobScheduler("local", LocalJobConfig(eval_program_path=str(evaluator)))
    in_worker_thread = threading.Event()
    let_popen_run = threading.Event()
    launched: list[ProcessWithLogging] = []
    submit = scheduler.submit_async

    def slow_submit(exec_fname: str, results_dir: str):
        in_worker_thread.set()
        assert let_popen_run.wait(timeout=5.0)
        process = submit(exec_fname, results_dir)
        assert isinstance(process, ProcessWithLogging)
        launched.append(process)
        return process

    monkeypatch.setattr(scheduler, "submit_async", slow_submit)

    async def run() -> None:
        runner = _submitting_runner(scheduler)
        proposal = asyncio.create_task(
            runner._submit_evaluation_job_with_slot(
                "main.py", str(tmp_path / "results"), None
            )
        )
        await asyncio.to_thread(in_worker_thread.wait, 5.0)
        proposal.cancel()
        await asyncio.sleep(0)
        let_popen_run.set()
        with pytest.raises(asyncio.CancelledError):
            await proposal
        assert runner.evaluation_slot_pool.in_use == 0

    try:
        asyncio.run(run())

        (process,) = launched
        assert process.returncode == -signal.SIGKILL
    finally:
        let_popen_run.set()
        for process in launched:
            process.kill_group()
        scheduler.shutdown()
