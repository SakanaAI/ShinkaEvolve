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
from unittest.mock import AsyncMock

import pytest

import shinka.core.async_runner as async_runner_module
from shinka.core.async_runner import (
    ShinkaEvolveRunner,
    UnconfirmedJobCancellationError,
)
from shinka.launch.slurm import AmbiguousSlurmSubmissionError

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


def test_request_stop_interrupts_run_after_stop_was_already_set():
    async def _run():
        runner = _build_runner()
        runner._interrupted = False
        runner.should_stop.set()
        run_task = asyncio.create_task(asyncio.Event().wait())
        runner._run_task = run_task

        runner._request_stop("SIGTERM")

        assert runner._interrupted is True
        with pytest.raises(asyncio.CancelledError):
            await run_task

    asyncio.run(_run())


def test_repeated_cancellation_does_not_abort_finalizer():
    async def _run():
        runner = _build_runner()
        runner._interrupted = True
        allow_finalizer = asyncio.Event()
        finalizer_finished = False

        async def finalize():
            nonlocal finalizer_finished
            await allow_finalizer.wait()
            finalizer_finished = True

        finalizer_task = asyncio.create_task(finalize())
        waiter = asyncio.create_task(
            runner._await_finalizer_resiliently(finalizer_task)
        )
        await asyncio.sleep(0)

        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        assert finalizer_task.cancelled() is False

        allow_finalizer.set()
        await waiter
        assert finalizer_finished is True

    asyncio.run(_run())


def test_signal_during_initial_evaluation_cancels_owned_job():
    async def _run():
        monitor_started = asyncio.Event()

        class _InitialEvaluationScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                return "initial-job"

            async def get_job_results_async(self, job_id, results_dir):
                monitor_started.set()
                await asyncio.Event().wait()

        scheduler = _InitialEvaluationScheduler(
            cancelled_job_ids=["initial-job"]
        )
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
            prompt_db=None,
        )
        runner._interrupted = False
        evaluation = asyncio.create_task(
            runner._run_initial_evaluation("main.py", "results")
        )
        runner._run_task = evaluation
        await monitor_started.wait()

        runner._request_stop("SIGINT")

        with pytest.raises(asyncio.CancelledError):
            await evaluation
        assert runner._unconfirmed_job_cancellations == {
            "initial-job": ("initial-job", 0)
        }

        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["initial-job"]
        assert slot_pool.in_use == 0

    asyncio.run(_run())


def test_initial_result_failure_cancels_job_and_releases_slot():
    async def _run():
        class _FailedResultScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                return "failed-result-job"

            async def get_job_results_async(self, job_id, results_dir):
                raise RuntimeError("result retrieval failed")

        scheduler = _FailedResultScheduler(
            cancelled_job_ids=["failed-result-job"]
        )
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
        )

        with pytest.raises(RuntimeError, match="result retrieval failed"):
            await runner._run_initial_evaluation("main.py", "results")

        assert scheduler.cancelled_job_ids == ["failed-result-job"]
        assert runner._unconfirmed_job_cancellations == {}
        assert slot_pool.in_use == 0

    asyncio.run(_run())


def test_initial_result_failure_aborts_when_cancellation_is_unconfirmed():
    async def _run():
        class _FailedResultScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                return "unconfirmed-job"

            async def get_job_results_async(self, job_id, results_dir):
                raise RuntimeError("result retrieval failed")

        scheduler = _FailedResultScheduler()
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
        )

        with pytest.raises(
            UnconfirmedJobCancellationError, match="unconfirmed-job"
        ):
            await runner._run_initial_evaluation("main.py", "results")

        assert runner._unconfirmed_job_cancellations == {
            "unconfirmed-job": ("unconfirmed-job", 0)
        }
        assert slot_pool.in_use == 1

    asyncio.run(_run())


def test_ambiguous_submission_name_remains_owned_until_cleanup():
    async def _run():
        error = AmbiguousSlurmSubmissionError("conda-unique")

        class _AmbiguousScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                raise error

        scheduler = _AmbiguousScheduler(
            cancelled_job_ids=[error.cancel_target]
        )
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
            prompt_db=None,
        )
        run_task = asyncio.create_task(asyncio.Event().wait())
        runner._run_task = run_task

        with pytest.raises(AmbiguousSlurmSubmissionError):
            await runner._submit_evaluation_job_with_slot(
                "main.py",
                "results",
                sampling_worker_id=None,
            )

        assert list(runner._unconfirmed_job_cancellations.values()) == [
            (error.cancel_target, 0)
        ]
        assert slot_pool.in_use == 1
        assert runner.should_stop.is_set()
        assert runner._fatal_error is error
        with pytest.raises(asyncio.CancelledError):
            await run_task

        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == [error.cancel_target]
        assert runner._unconfirmed_job_cancellations == {}
        assert slot_pool.in_use == 0

    asyncio.run(_run())


def test_run_async_surfaces_ambiguous_background_submission_after_cleanup(
    monkeypatch,
):
    async def _run():
        error = AmbiguousSlurmSubmissionError("background-unique")
        runner = _build_runner(
            slot_available=asyncio.Event(),
            should_stop=asyncio.Event(),
            finalization_complete=asyncio.Event(),
        )
        runner.pricing_snapshot = None
        runner.embedding_client = None
        runner._install_signal_handlers = lambda _loop: []
        runner._setup_async = AsyncMock()
        runner._verify_database_ready = AsyncMock()
        runner._cancel_completed_job_batches = AsyncMock()
        runner._cancel_background_side_effect_worker = AsyncMock()
        runner._cleanup_async = AsyncMock()

        async def monitor():
            await asyncio.Event().wait()

        async def submit_ambiguous_proposal():
            runner._record_fatal_error(error)

        runner._job_monitor_task = monitor
        runner._proposal_coordinator_task = submit_ambiguous_proposal
        monkeypatch.setattr(
            async_runner_module,
            "activate_model_catalog",
            lambda _snapshot: None,
        )

        with pytest.raises(AmbiguousSlurmSubmissionError) as exc_info:
            await runner.run_async()

        assert exc_info.value is error
        runner._cleanup_async.assert_awaited_once()

    asyncio.run(_run())


def test_initial_program_does_not_fallback_after_ambiguous_submission(tmp_path):
    async def _run():
        error = AmbiguousSlurmSubmissionError("initial-unique")

        class _AmbiguousScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                raise error

        async_db = SimpleNamespace(add_program_async=AsyncMock())
        runner = _build_runner(
            scheduler=_AmbiguousScheduler(),
            evaluation_slot_pool=_FakeSlotPool(),
            async_db=async_db,
            results_dir=str(tmp_path),
        )
        runner._get_code_embedding_async = AsyncMock(return_value=(None, 0.0))

        with pytest.raises(AmbiguousSlurmSubmissionError):
            await runner._setup_initial_program_with_metadata(
                "print('hello')",
                "initial",
                "initial program",
                0.0,
            )

        async_db.add_program_async.assert_not_awaited()
        assert list(runner._unconfirmed_job_cancellations.values()) == [
            (error.cancel_target, 0)
        ]

    asyncio.run(_run())


def test_cleanup_retries_until_ambiguous_job_name_disappears():
    async def _run():
        target = AmbiguousSlurmSubmissionError("eventual-unique").cancel_target

        class _EventuallyTerminalScheduler(_FakeScheduler):
            def __init__(self):
                super().__init__()
                self.terminal_checks = 0

            async def cancel_job_async(self, job_id):
                self.cancelled_job_ids.append(job_id)
                return False

            async def is_job_terminal_async(self, job_id):
                self.terminal_checks += 1
                return self.terminal_checks >= 2

        scheduler = _EventuallyTerminalScheduler()
        runner = _build_runner(scheduler=scheduler)

        assert await runner._cancel_job_ids([target]) == []
        assert scheduler.cancelled_job_ids == [target, target]

    asyncio.run(_run())


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


def test_cancelled_submission_retains_eventual_ambiguous_job_name():
    async def _run():
        submit_started = asyncio.Event()
        allow_submit = asyncio.Event()
        error = AmbiguousSlurmSubmissionError("cancelled-waiter-unique")

        class _AmbiguousScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                submit_started.set()
                await allow_submit.wait()
                raise error

        scheduler = _AmbiguousScheduler(
            cancelled_job_ids=[error.cancel_target]
        )
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
            prompt_db=None,
        )
        submission_task = asyncio.create_task(
            runner._submit_evaluation_job_with_slot(
                exec_fname="candidate.py",
                results_dir="results",
                sampling_worker_id=None,
            )
        )
        await submit_started.wait()

        submission_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission_task

        cleanup_task = asyncio.create_task(runner._cleanup_async())
        allow_submit.set()
        await cleanup_task

        assert scheduler.cancelled_job_ids == [error.cancel_target]
        assert runner._unconfirmed_job_cancellations == {}
        assert slot_pool.in_use == 0

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
