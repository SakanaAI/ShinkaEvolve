"""Regression tests for graceful shutdown / job cleanup in the async runner.

Covers:
- ``_request_stop`` (the SIGINT/SIGTERM handler) sets the stop/finalization
  events so run_async unblocks, marks the run interrupted, and is idempotent.
- ``_cleanup_async`` cancels every still-running evaluation job so the process
  never orphans local subprocesses or leaves Slurm jobs running.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import shinka.core.async_runner as async_runner_module
from shinka.database.async_dbase import EvaluationOwnershipConflictError
from shinka.core.async_runner import (
    AsyncRunningJob,
    PendingEvaluationSubmission,
    ShinkaEvolveRunner,
    UnconfirmedJobCancellationError,
)
from shinka.launch.slurm import (
    AmbiguousSlurmSubmissionError,
    JobStatusUnavailableError,
    SlurmJobName,
)
from shinka.launch.scheduler import JobScheduler, SlurmCondaJobConfig
from shinka.launch import slurm

from test_async_runner_recovery import (
    _FakeAsyncDB,
    _FakeScheduler,
    _FakeSlotPool,
    _build_runner,
)


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
        assert runner.async_db.evaluation_ownership[0]["phase"] == "active"

        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["initial-job"]
        assert slot_pool.in_use == 0
        assert runner.async_db.evaluation_ownership[0]["phase"] == "resolved"

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
        assert runner.async_db.evaluation_ownership[0]["phase"] == "resolved"

    asyncio.run(_run())


def test_completed_initial_evaluation_keeps_ownership_until_program_persisted():
    async def _run():
        class _InitialEvaluationScheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                return "initial-job"

            async def get_job_results_async(self, job_id, results_dir):
                return {"correct": {"correct": True}, "metrics": {}}

        scheduler = _InitialEvaluationScheduler(
            terminal_job_ids=["initial-job"]
        )
        runner = _build_runner(scheduler=scheduler)

        await runner._run_initial_evaluation("main.py", "results")

        assert runner.async_db.evaluation_ownership[0]["phase"] == "active"
        assert runner._unconfirmed_job_cancellations == {
            "initial-job": ("initial-job", None)
        }

        await runner._cleanup_async()

        assert runner.async_db.evaluation_ownership[0]["phase"] == "resolved"

    asyncio.run(_run())


def test_initial_evaluation_ownership_conflict_is_fatal():
    async def _run():
        submitted = False

        class _Scheduler(_FakeScheduler):
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                nonlocal submitted
                submitted = True
                return "unexpected-job"

        runner = _build_runner(scheduler=_Scheduler())
        runner.async_db.evaluation_ownership[0] = {
            "generation": 0,
            "phase": "active",
            "job_type": "local",
            "job_id": None,
            "job_name": None,
            "results_dir": "results",
        }

        with pytest.raises(EvaluationOwnershipConflictError):
            await runner._run_initial_evaluation("main.py", "results")

        assert submitted is False
        assert runner.async_db.evaluation_ownership[0]["phase"] == "active"

    asyncio.run(_run())


def test_evolved_evaluation_ownership_conflict_is_fatal(tmp_path):
    async def _run():
        runner = _build_runner(results_dir=str(tmp_path))
        runner.total_proposals_generated = 0
        error = EvaluationOwnershipConflictError(
            "Generation 1 already has active evaluation ownership"
        )
        runner._generate_evolved_proposal = AsyncMock(side_effect=error)

        with pytest.raises(EvaluationOwnershipConflictError):
            await runner._generate_proposal_async(1, "proposal-1")

        assert runner._fatal_error is error
        assert runner.should_stop.is_set()
        assert runner.finalization_complete.is_set()

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


def test_cleanup_retains_dispatched_slurm_name_when_scheduler_reports_absent(
    monkeypatch,
):
    job_name = "conda-" + "a" * 32
    target = SlurmJobName(job_name)

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "scancel":
            return SimpleNamespace(returncode=0, stdout="")
        assert cmd[0] in {"squeue", "sacct"}
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "shinka.launch.scheduler._get_current_user_id", lambda: "1000"
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    async def _run():
        scheduler = JobScheduler(
            "slurm_conda",
            SlurmCondaJobConfig(),
            max_workers=1,
        )
        async_db = _FakeAsyncDB(0)
        async_db.evaluation_ownership[7] = {
            "generation": 7,
            "phase": "submitting",
            "job_type": "slurm_conda",
            "job_id": None,
            "job_name": job_name,
            "results_dir": "results",
            "dispatch_state": 2,
            "updated_at": time.time(),
        }
        slot_pool = _FakeSlotPool()
        slot_pool.in_use = 1
        runner = _build_runner(
            async_db=async_db,
            scheduler=scheduler,
            evaluation_slot_pool=slot_pool,
            _unconfirmed_job_cancellations={id(target): (target, 0)},
            _unconfirmed_job_cancellation_generations={id(target): 7},
            prompt_db=None,
        )

        try:
            with pytest.raises(UnconfirmedJobCancellationError, match=job_name):
                await runner._cleanup_async()

            assert runner._unconfirmed_job_cancellations == {
                id(target): (target, 0)
            }
            assert async_db.evaluation_ownership[7]["phase"] == "submitting"
            assert async_db.evaluation_ownership[7]["dispatch_state"] == 2
            assert slot_pool.in_use == 1
        finally:
            scheduler.shutdown()

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
        runner._install_signal_handlers = lambda _loop: (_ for _ in ()).throw(
            AssertionError("run_async must not install process signal handlers")
        )
        runner._setup_async = AsyncMock()
        runner._verify_database_ready = AsyncMock()
        runner._cancel_completed_job_batches = AsyncMock()
        runner._cancel_background_side_effect_worker = AsyncMock()
        runner._cleanup_async = AsyncMock()
        lease_closes = []
        runner._results_root_lease = SimpleNamespace(
            close=lambda: lease_closes.append(True)
        )

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
        assert lease_closes == [True]
        assert runner._results_root_lease is None

    asyncio.run(_run())


def test_run_async_holds_results_lease_through_final_summary(tmp_path, monkeypatch):
    async def _run():
        results_root = tmp_path / "results"
        results_root.mkdir(mode=0o700)
        lease = async_runner_module._acquire_results_root_lease(results_root)
        finalization_complete = asyncio.Event()
        finalization_complete.set()
        runner = _build_runner(
            finalization_complete=finalization_complete,
            should_stop=asyncio.Event(),
            slot_available=asyncio.Event(),
            results_dir=str(results_root),
            prompt_db=None,
        )
        runner._results_root_lease = lease
        runner.pricing_snapshot = None
        runner.embedding_client = None
        runner.meta_summarizer = None
        runner._setup_async = AsyncMock()
        runner._verify_database_ready = AsyncMock()
        runner._wait_for_completed_job_batches = AsyncMock()
        runner._has_background_side_effect_work = lambda: False
        runner._shutdown_background_side_effect_worker = AsyncMock()
        runner._cancel_completed_job_batches = AsyncMock()
        runner._cancel_background_side_effect_worker = AsyncMock()
        runner._cleanup_async = AsyncMock()
        runner._save_bandit_state = lambda: None

        async def idle_task():
            await asyncio.Event().wait()

        async def assert_lease_held():
            assert runner._results_root_lease is lease
            with pytest.raises(async_runner_module.ResultsRootInUseError):
                async_runner_module._acquire_results_root_lease(results_root)

        runner._job_monitor_task = idle_task
        runner._proposal_coordinator_task = idle_task
        runner._print_final_summary = assert_lease_held
        monkeypatch.setattr(
            async_runner_module,
            "activate_model_catalog",
            lambda _snapshot: None,
        )

        await runner.run_async()

        assert runner._results_root_lease is None
        with async_runner_module._acquire_results_root_lease(results_root):
            pass
        with pytest.raises(RuntimeError, match="already completed"):
            await runner.run_async()

    asyncio.run(_run())


def test_run_async_rejects_concurrent_call_before_first_await_completes():
    async def _run():
        runner = _build_runner()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        main_calls = 0

        async def controlled_main():
            nonlocal main_calls
            main_calls += 1
            if main_calls == 1:
                first_started.set()
                await release_first.wait()

        runner._run_async_main = controlled_main
        first_run = asyncio.create_task(runner.run_async())
        await first_started.wait()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await runner.run_async()
        finally:
            release_first.set()
            await first_run

        assert main_calls == 1

    asyncio.run(_run())


def test_run_async_releases_lease_when_log_handler_close_fails(tmp_path):
    async def _run():
        results_root = tmp_path / "results"
        results_root.mkdir(mode=0o700)
        runner = _build_runner()
        runner._results_root_lease = (
            async_runner_module._acquire_results_root_lease(results_root)
        )
        runner._run_async_main = AsyncMock()
        runner._run_log_handler = SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(OSError("close failed"))
        )

        await runner.run_async()

        with async_runner_module._acquire_results_root_lease(results_root):
            pass

    asyncio.run(_run())


def test_run_async_surfaces_unknown_job_status_after_cancelling_job(
    monkeypatch,
):
    async def _run():
        class _UnknownScheduler(_FakeScheduler):
            async def batch_check_status_async(self, jobs):
                return [None for _ in jobs]

        scheduler = _UnknownScheduler(cancelled_job_ids=["job-unknown"])
        now = time.time()
        running_job = AsyncRunningJob(
            job_id="job-unknown",
            exec_fname="program.py",
            results_dir="results",
            start_time=now,
            proposal_started_at=now,
            evaluation_submitted_at=now,
            generation=1,
        )
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[running_job],
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
        runner._finish_wandb_logging = lambda: None
        runner._has_persistence_work_in_progress = lambda: False
        runner._cancel_surplus_inflight_work = AsyncMock()
        runner._retry_failed_db_jobs = AsyncMock()
        runner._record_progress = lambda: None

        async def coordinator():
            await asyncio.Event().wait()

        runner._proposal_coordinator_task = coordinator
        monkeypatch.setattr(
            async_runner_module,
            "activate_model_catalog",
            lambda _snapshot: None,
        )
        monkeypatch.setattr(
            async_runner_module,
            "JOB_STATUS_UNKNOWN_TIMEOUT_SECONDS",
            0.0,
        )

        with pytest.raises(JobStatusUnavailableError):
            await runner.run_async()

        assert scheduler.cancelled_job_ids == ["job-unknown"]
        assert scheduler.shutdown_called
        assert runner.async_db.closed

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


def test_cleanup_waits_for_known_job_id_to_disappear():
    async def _run():
        target = "123"

        class _EventuallyTerminalScheduler(_FakeScheduler):
            def __init__(self):
                super().__init__()
                self.terminal_checks = 0

            async def cancel_job_async(self, job_id):
                self.cancelled_job_ids.append(job_id)
                return False

            async def is_job_terminal_async(self, job_id):
                self.terminal_checks += 1
                return self.terminal_checks >= 5

        scheduler = _EventuallyTerminalScheduler()
        runner = _build_runner(scheduler=scheduler)

        assert await runner._cancel_job_ids([target]) == []
        assert scheduler.cancelled_job_ids == [target] * 5

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

        assert scheduler.cancelled_job_ids == [
            "still-running"
        ] * async_runner_module.JOB_CANCELLATION_ATTEMPTS
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

        assert scheduler.cancelled_job_ids == [
            "unreachable-job"
        ] * async_runner_module.JOB_CANCELLATION_ATTEMPTS
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
                    generation=7,
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
        assert runner.async_db.evaluation_ownership[7]["phase"] == "resolved"

    asyncio.run(_run())


def test_cancelled_slot_wait_creates_no_durable_ownership():
    async def _run():
        acquire_started = asyncio.Event()
        release_acquire = asyncio.Event()

        class _BlockedSlotPool(_FakeSlotPool):
            async def acquire(self):
                acquire_started.set()
                await release_acquire.wait()
                return 0

        runner = _build_runner(
            evaluation_slot_pool=_BlockedSlotPool(),
            prompt_db=None,
        )
        submission = asyncio.create_task(
            runner._submit_evaluation_job_with_slot(
                exec_fname="candidate.py",
                results_dir="results",
                sampling_worker_id=None,
                generation=8,
            )
        )
        await acquire_started.wait()

        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission

        assert runner.async_db.evaluation_ownership == {}

    asyncio.run(_run())


def test_cancelled_ownership_write_resolves_without_submitting():
    async def _run():
        ownership_started = asyncio.Event()
        allow_ownership = asyncio.Event()

        class _BlockingOwnershipDB(_FakeAsyncDB):
            async def begin_evaluation_ownership_async(
                self, generation, job_type, results_dir, job_name=None
            ):
                ownership_started.set()
                await allow_ownership.wait()
                await super().begin_evaluation_ownership_async(
                    generation,
                    job_type,
                    results_dir,
                    job_name,
                )

        async_db = _BlockingOwnershipDB(total_programs=0)
        scheduler = _FakeScheduler()
        runner = _build_runner(
            async_db=async_db,
            scheduler=scheduler,
            prompt_db=None,
        )
        submission = asyncio.create_task(
            runner._submit_evaluation_job_with_slot(
                exec_fname="candidate.py",
                results_dir="results",
                sampling_worker_id=None,
                generation=8,
            )
        )
        await ownership_started.wait()

        submission.cancel()
        allow_ownership.set()
        with pytest.raises(asyncio.CancelledError):
            await submission

        assert async_db.evaluation_ownership[8]["phase"] == "resolved"
        assert scheduler.cancelled_job_ids == []

    asyncio.run(_run())


def test_cleanup_resolves_confirmed_active_local_ownership():
    async def _run():
        scheduler = _FakeScheduler(cancelled_job_ids=["local-job"])
        scheduler.job_type = "local"
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[
                SimpleNamespace(job_id="local-job", generation=9)
            ],
            prompt_db=None,
        )
        await runner.async_db.begin_evaluation_ownership_async(
            generation=9,
            job_type="local",
            results_dir="results",
        )
        await runner.async_db.activate_evaluation_ownership_async(
            generation=9,
            job_id=None,
            job_name=None,
        )

        await runner._cleanup_async()

        assert runner.async_db.evaluation_ownership[9]["phase"] == "resolved"

    asyncio.run(_run())


def test_cleanup_resolves_terminal_submitted_job_during_persistence():
    async def _run():
        scheduler = _FakeScheduler(terminal_job_ids=["completed-job"])
        scheduler.job_type = "local"
        job = AsyncRunningJob(
            job_id="completed-job",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=10,
        )
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[],
            submitted_jobs={"completed-job": job},
            prompt_db=None,
        )
        await runner.async_db.begin_evaluation_ownership_async(
            generation=10,
            job_type="local",
            results_dir="results",
        )
        await runner.async_db.activate_evaluation_ownership_async(
            generation=10,
            job_id=None,
            job_name=None,
        )

        await runner._cleanup_async()

        assert runner.submitted_jobs == {}
        assert runner.async_db.evaluation_ownership[10]["phase"] == "resolved"

    asyncio.run(_run())


def test_cleanup_retains_submitted_job_when_cancellation_is_unconfirmed():
    async def _run():
        scheduler = _FakeScheduler()
        scheduler.job_type = "local"
        job = AsyncRunningJob(
            job_id="processing-job",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=11,
        )
        runner = _build_runner(
            scheduler=scheduler,
            running_jobs=[],
            submitted_jobs={"processing-job": job},
            prompt_db=None,
        )
        await runner.async_db.begin_evaluation_ownership_async(
            generation=11,
            job_type="local",
            results_dir="results",
        )
        await runner.async_db.activate_evaluation_ownership_async(
            generation=11,
            job_id=None,
            job_name=None,
        )

        with pytest.raises(UnconfirmedJobCancellationError, match="processing-job"):
            await runner._cleanup_async()

        assert runner.running_jobs == [job]
        assert runner.submitted_jobs == {"processing-job": job}
        assert runner.async_db.evaluation_ownership[11]["phase"] == "active"

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
            _pending_evaluation_submissions={
                submission: PendingEvaluationSubmission(0, None)
            },
            prompt_db=None,
        )

        with pytest.raises(UnconfirmedJobCancellationError, match="late-job"):
            await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == [
            "late-job"
        ] * async_runner_module.JOB_CANCELLATION_ATTEMPTS
        assert runner._unconfirmed_job_cancellations == {
            "late-job": ("late-job", 0)
        }
        assert slot_pool.in_use == 1

        scheduler._cancelled_job_ids.add("late-job")
        await runner._cleanup_async()

        assert scheduler.cancelled_job_ids == ["late-job"] * (
            async_runner_module.JOB_CANCELLATION_ATTEMPTS + 1
        )
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
            _pending_evaluation_submissions={
                submission: PendingEvaluationSubmission(0, None)
            },
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


def test_signal_handlers_restore_previous_callbacks(monkeypatch):
    runner = _build_runner()
    previous_handlers = {
        async_runner_module.signal.SIGINT: object(),
        async_runner_module.signal.SIGTERM: object(),
    }
    process_handlers = dict(previous_handlers)
    restored = []

    class _FakeLoop:
        def __init__(self):
            self.scheduled = []

        def call_soon_threadsafe(self, callback, *args):
            self.scheduled.append((callback, args))

    loop = _FakeLoop()
    monkeypatch.setattr(
        async_runner_module.signal,
        "getsignal",
        lambda sig: process_handlers[sig],
    )
    def set_signal(sig, handler):
        previous_handler = process_handlers[sig]
        process_handlers[sig] = handler
        if handler is previous_handlers[sig]:
            restored.append((sig, handler))
        return previous_handler

    monkeypatch.setattr(async_runner_module.signal, "signal", set_signal)
    monkeypatch.setattr(
        async_runner_module.signal,
        "pthread_sigmask",
        lambda how, signals: set(),
    )

    installed = runner._install_signal_handlers(loop)
    runner._restore_signal_handlers(loop, installed)

    assert {
        sig: registration.previous_process_handler
        for sig, registration in installed.items()
    } == previous_handlers
    assert restored == list(previous_handlers.items())
    assert process_handlers == previous_handlers


def test_signal_restore_preserves_newer_process_callback(monkeypatch):
    runner = _build_runner()
    previous_handlers = {
        async_runner_module.signal.SIGINT: object(),
        async_runner_module.signal.SIGTERM: object(),
    }
    process_handlers = {
        async_runner_module.signal.SIGINT: previous_handlers[
            async_runner_module.signal.SIGINT
        ],
        async_runner_module.signal.SIGTERM: previous_handlers[
            async_runner_module.signal.SIGTERM
        ],
    }

    class _FakeLoop:
        def __init__(self):
            self.scheduled = []

        def call_soon_threadsafe(self, callback, *args):
            self.scheduled.append((callback, args))

    loop = _FakeLoop()
    monkeypatch.setattr(
        async_runner_module.signal,
        "getsignal",
        lambda sig: process_handlers[sig],
    )
    def set_signal(sig, handler):
        previous_handler = process_handlers[sig]
        process_handlers[sig] = handler
        return previous_handler

    monkeypatch.setattr(async_runner_module.signal, "signal", set_signal)
    monkeypatch.setattr(
        async_runner_module.signal,
        "pthread_sigmask",
        lambda how, signals: set(),
    )

    installed = runner._install_signal_handlers(loop)
    newer_handler = object()
    set_signal(async_runner_module.signal.SIGINT, newer_handler)

    runner._restore_signal_handlers(loop, installed)

    assert (
        process_handlers[async_runner_module.signal.SIGINT]
        is newer_handler
    )
    assert (
        process_handlers[async_runner_module.signal.SIGTERM]
        is previous_handlers[async_runner_module.signal.SIGTERM]
    )


def test_newer_process_handler_survives_asyncio_loop_close():
    runner = _build_runner()
    previous_sigterm = async_runner_module.signal.getsignal(
        async_runner_module.signal.SIGTERM
    )

    def newer_handler(_sig, _frame):
        return None

    async def _run():
        loop = asyncio.get_running_loop()
        installed = runner._install_signal_handlers(loop)
        async_runner_module.signal.signal(
            async_runner_module.signal.SIGTERM,
            newer_handler,
        )
        runner._restore_signal_handlers(loop, installed)

    try:
        asyncio.run(_run())
        assert (
            async_runner_module.signal.getsignal(
                async_runner_module.signal.SIGTERM
            )
            is newer_handler
        )
    finally:
        async_runner_module.signal.signal(
            async_runner_module.signal.SIGTERM,
            previous_sigterm,
        )


def test_run_uses_signal_owning_wrapper():
    runner = _build_runner()
    calls = []

    async def run_with_signals():
        calls.append("wrapped")

    async def run_without_signals():
        raise AssertionError("run() must use the signal-owning wrapper")

    runner._run_async_with_signal_handlers = run_with_signals
    runner.run_async = run_without_signals

    runner.run()

    assert calls == ["wrapped"]
