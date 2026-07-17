import asyncio
import subprocess
import threading

import pytest

from shinka.launch import scheduler as scheduler_module
from shinka.launch import slurm as slurm_module
from shinka.launch.scheduler import JobScheduler, LocalJobConfig, SlurmCondaJobConfig


def test_slurm_cancel_waits_for_job_to_leave_queue(monkeypatch) -> None:
    statuses = iter(["RUNNING", "COMPLETING", ""])
    status_checks: list[str] = []
    sleeps: list[float] = []
    command_timeouts: list[float] = []
    status_timeouts: list[float] = []

    def run(*args, **kwargs):
        command_timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)

    def get_job_status(job_id: str, timeout: float) -> str:
        status_checks.append(job_id)
        status_timeouts.append(timeout)
        return next(statuses)

    monkeypatch.setattr(scheduler_module, "get_job_status", get_job_status)
    monkeypatch.setattr(scheduler_module.time, "sleep", sleeps.append)

    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        cancelled = asyncio.run(scheduler.cancel_job_async("12345"))
    finally:
        scheduler.shutdown()

    assert cancelled is True
    assert status_checks == ["12345", "12345", "12345"]
    assert len(sleeps) == 2
    assert command_timeouts and all(timeout > 0 for timeout in command_timeouts)
    assert status_timeouts and all(timeout > 0 for timeout in status_timeouts)


def test_slurm_cancel_fails_when_job_remains_queued(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(
        scheduler_module,
        "get_job_status",
        lambda _job_id, timeout: "COMPLETING",
    )
    monkeypatch.setattr(scheduler_module, "_SLURM_CANCEL_TIMEOUT_SECONDS", 0.0)

    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        cancelled = asyncio.run(scheduler.cancel_job_async("12345"))
    finally:
        scheduler.shutdown()

    assert cancelled is False


def test_slurm_cancel_command_timeout_is_failure(monkeypatch) -> None:
    timeouts: list[float] = []

    def timeout(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(scheduler_module.subprocess, "run", timeout)
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        assert asyncio.run(scheduler.cancel_job_async("12345")) is False
    finally:
        scheduler.shutdown()

    assert timeouts and all(command_timeout > 0 for command_timeout in timeouts)


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["squeue"], 0.1),
        subprocess.CalledProcessError(1, ["squeue"]),
    ],
)
def test_slurm_status_errors_are_unknown(monkeypatch, error: Exception) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(slurm_module.subprocess, "run", fail)

    assert slurm_module.get_job_status("12345", timeout=0.1) is None


def test_scheduler_treats_unknown_slurm_status_as_running(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "get_job_status",
        lambda _job_id: None,
    )
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        assert scheduler.check_job_id_status("12345") is True
    finally:
        scheduler.shutdown()


def test_begin_shutdown_queues_cleanup_before_concurrent_executor_shutdown():
    class _ControlledExecutor:
        def __init__(self):
            self.submit_started = threading.Event()
            self.allow_submit = threading.Event()
            self.cleanup_calls = []
            self.closed = False

        def submit(self, callback, *args):
            self.submit_started.set()
            assert self.allow_submit.wait(timeout=0.2)
            if self.closed:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self.cleanup_calls.append((callback, args))
            return object()

        def shutdown(self, wait=True, cancel_futures=False):
            self.closed = True

    scheduler = JobScheduler("local", LocalJobConfig())
    original_executor = scheduler.executor
    controlled_executor = _ControlledExecutor()
    scheduler.executor = controlled_executor
    original_executor.shutdown(wait=True)

    token = object()
    with scheduler._submission_lock:
        scheduler._unclaimed_submissions[token] = "late-job"

    errors = []
    shutdown_started = threading.Event()

    def begin_shutdown():
        try:
            scheduler.begin_shutdown()
        except Exception as error:
            errors.append(error)

    def shutdown_nowait():
        shutdown_started.set()
        scheduler.shutdown_nowait()

    begin_thread = threading.Thread(target=begin_shutdown)
    shutdown_thread = threading.Thread(target=shutdown_nowait)
    begin_thread.start()
    assert controlled_executor.submit_started.wait(timeout=0.2)
    shutdown_thread.start()
    assert shutdown_started.wait(timeout=0.2)
    shutdown_thread.join(timeout=0.05)
    assert shutdown_thread.is_alive()
    controlled_executor.allow_submit.set()
    begin_thread.join(timeout=0.2)
    shutdown_thread.join(timeout=0.2)

    assert not begin_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert errors == []
    assert len(controlled_executor.cleanup_calls) == 1
    assert token in scheduler._shutdown_cleanup_futures
