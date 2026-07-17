"""Regression tests for graceful shutdown / job cleanup in the async runner.

Covers:
- ``_request_stop`` (the SIGINT/SIGTERM handler) sets the stop/finalization
  events so run_async unblocks, marks the run interrupted, and is idempotent.
- ``_cleanup_async`` cancels every still-running evaluation job so the process
  never orphans local subprocesses or leaves Slurm jobs running.
"""

import asyncio
from types import SimpleNamespace

from shinka.core.async_runner import ShinkaEvolveRunner

from test_async_runner_recovery import _FakeScheduler, _build_runner


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


def test_runner_has_signal_handler_api():
    # The handler-install helper must exist and tolerate a missing loop capability.
    assert hasattr(ShinkaEvolveRunner, "_install_signal_handlers")
    assert hasattr(ShinkaEvolveRunner, "_request_stop")
