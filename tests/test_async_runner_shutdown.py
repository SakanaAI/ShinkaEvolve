"""Regression tests for graceful shutdown / job cleanup in the async runner.

Covers:
- ``_request_stop`` (the SIGINT/SIGTERM handler) sets the stop/finalization
  events so run_async unblocks, marks the run interrupted, and is idempotent.
- ``_cleanup_async`` cancels every still-running evaluation job so the process
  never orphans local subprocesses or leaves Slurm jobs running.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import (
    ShinkaEvolveRunner,
    UnconfirmedJobCancellationError,
)

from test_async_runner_recovery import _FakeScheduler, _FakeSlotPool, _build_runner


class _RealEvent:
    """asyncio.Event stand-in usable outside a running loop."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set


def test_request_stop_sets_shutdown_flags_and_is_idempotent():
    runner = _build_runner(
        should_stop=_RealEvent(),
        finalization_complete=_RealEvent(),
        slot_available=_RealEvent(),
    )
    runner._interrupted = False

    runner._request_stop("SIGINT")

    assert runner._interrupted is True
    assert runner.should_stop.is_set()
    assert runner.finalization_complete.is_set()
    assert runner.slot_available.is_set()

    # Second signal must be a harmless no-op, not a crash.
    runner._request_stop("SIGTERM")
    assert runner.should_stop.is_set()


def test_cleanup_cancels_running_jobs():
    async def _run():
        scheduler = _FakeScheduler(cancelled_job_ids=["j1", "j2"])
        jobs = [
            SimpleNamespace(job_id="j1"),
            SimpleNamespace(job_id="j2"),
        ]
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=list(jobs),
            active_proposal_tasks={},
            prompt_db=None,
        )
        # _cleanup_async logs-and-swallows failures from later teardown steps
        # (async_db.close_async etc.); the job cancellation runs first.
        await runner._cleanup_async()

        assert set(scheduler.cancelled_job_ids) == {"j1", "j2"}
        assert runner.running_jobs == []

    asyncio.run(_run())


def test_cleanup_retries_and_retains_job_when_cancellation_fails():
    async def _run():
        scheduler = _FakeScheduler()
        job = SimpleNamespace(job_id="still-running")
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[job],
            active_proposal_tasks={},
            prompt_db=None,
        )

        with pytest.raises(
            UnconfirmedJobCancellationError, match="still-running"
        ):
            await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["still-running"] * 3
        assert runner.running_jobs == [job]
        assert scheduler.shutdown_called is False

    asyncio.run(_run())


def test_cleanup_releases_job_after_retry_succeeds():
    async def _run():
        class _RetryScheduler(_FakeScheduler):
            async def cancel_job_async(self, job_id):
                self.cancelled_job_ids.append(job_id)
                return len(self.cancelled_job_ids) >= 2

        scheduler = _RetryScheduler()
        job = SimpleNamespace(job_id="eventually-cancelled")
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[job],
            active_proposal_tasks={},
            prompt_db=None,
        )

        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["eventually-cancelled"] * 2
        assert runner.running_jobs == []

    asyncio.run(_run())


def test_cleanup_retries_cancellation_exceptions():
    async def _run():
        class _FailingScheduler(_FakeScheduler):
            async def cancel_job_async(self, job_id):
                self.cancelled_job_ids.append(job_id)
                raise RuntimeError("controller unavailable")

        scheduler = _FailingScheduler()
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[SimpleNamespace(job_id="unreachable-job")],
            prompt_db=None,
        )

        with pytest.raises(
            UnconfirmedJobCancellationError, match="unreachable-job"
        ):
            await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["unreachable-job"] * 3
        assert scheduler.shutdown_called is False

    asyncio.run(_run())


def test_cleanup_accepts_terminal_job_after_cancellation_failure():
    async def _run():
        scheduler = _FakeScheduler(terminal_job_ids=["already-finished"])
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[SimpleNamespace(job_id="already-finished")],
            prompt_db=None,
        )

        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["already-finished"]
        assert runner.running_jobs == []
        assert scheduler.shutdown_called is True

    asyncio.run(_run())


def test_cleanup_settles_proposals_before_snapshotting_jobs():
    async def _run():
        scheduler = _FakeScheduler(cancelled_job_ids=["existing", "late"])
        existing_job = SimpleNamespace(job_id="existing")
        late_job = SimpleNamespace(job_id="late")
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[existing_job],
            active_proposal_tasks={},
            prompt_db=None,
        )

        async def proposal():
            try:
                await asyncio.Event().wait()
            finally:
                runner.running_jobs.append(late_job)

        proposal_task = asyncio.create_task(proposal())
        runner.active_proposal_tasks = {"proposal": proposal_task}
        await asyncio.sleep(0)

        await runner._cleanup_async()

        assert set(scheduler.cancelled_job_ids) == {"existing", "late"}
        assert runner.running_jobs == []

    asyncio.run(_run())


def test_cancelled_submission_cancels_eventual_external_job():
    async def _run():
        submit_started = threading.Event()
        allow_submit = threading.Event()
        existing_cancel_started = asyncio.Event()

        class _ExecutorScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                loop = asyncio.get_running_loop()

                def submit():
                    submit_started.set()
                    allow_submit.wait(timeout=2)
                    return "late-job"

                return await loop.run_in_executor(None, submit)

            async def cancel_job_async(self, job_id):
                if job_id == "existing-job":
                    existing_cancel_started.set()
                return await super().cancel_job_async(job_id)

        scheduler = _ExecutorScheduler(
            cancelled_job_ids=["existing-job", "late-job"]
        )
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[SimpleNamespace(job_id="existing-job")],
            prompt_db=None,
        )

        submission_task = asyncio.create_task(
            runner._submit_evaluation_job_with_slot(
                exec_fname="candidate.py",
                results_dir="results",
                sampling_worker_id=None,
            )
        )
        await asyncio.wait_for(asyncio.to_thread(submit_started.wait, 1), timeout=2)

        submission_task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await submission_task

        cleanup_task = asyncio.create_task(runner._cleanup_async())
        await asyncio.wait_for(existing_cancel_started.wait(), timeout=1)

        allow_submit.set()
        await cleanup_task

        assert scheduler.cancelled_job_ids == ["existing-job", "late-job"]
        assert runner.evaluation_slot_pool.in_use == 0

    asyncio.run(_run())


def test_cleanup_retains_late_submission_until_cancellation_is_confirmed():
    async def _run():
        scheduler = _FakeScheduler()
        slot_pool = _FakeSlotPool()
        slot_pool.in_use = 1

        async def submitted_job():
            return "late-job"

        submission = asyncio.create_task(submitted_job())
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
            _pending_evaluation_submissions={submission: 0},
            prompt_db=None,
        )

        with pytest.raises(UnconfirmedJobCancellationError, match="late-job"):
            await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["late-job"] * 3
        assert runner._unconfirmed_job_cancellations == {
            "late-job": ("late-job", 0)
        }
        assert slot_pool.in_use == 1

        scheduler._cancelled_job_ids.add("late-job")
        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["late-job"] * 4
        assert runner._unconfirmed_job_cancellations == {}
        assert slot_pool.in_use == 0
        assert scheduler.shutdown_called is True

    asyncio.run(_run())


def test_cleanup_transfers_late_submission_ownership_before_cancellation():
    async def _run():
        cancellation_started = asyncio.Event()

        class _BlockingScheduler(_FakeScheduler):
            async def cancel_job_async(self, job_id):
                cancellation_started.set()
                await asyncio.Event().wait()
                return True

        async def submitted_job():
            return "late-job"

        submission = asyncio.create_task(submitted_job())
        runner = _build_runner(
            scheduler=_BlockingScheduler(),
            _pending_evaluation_submissions={submission: 0},
            prompt_db=None,
        )
        cleanup_task = asyncio.create_task(runner._cleanup_async())
        await cancellation_started.wait()

        cleanup_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup_task

        assert runner._pending_evaluation_submissions == {}
        assert runner._unconfirmed_job_cancellations == {
            "late-job": ("late-job", 0)
        }

    asyncio.run(_run())


def test_runner_has_signal_handler_api():
    # The handler-install helper must exist and tolerate a missing loop capability.
    assert hasattr(ShinkaEvolveRunner, "_install_signal_handlers")
    assert hasattr(ShinkaEvolveRunner, "_request_stop")
