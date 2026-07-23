"""Regression tests for job-status detection and process-group teardown.

Covers:
- A local job still waiting for a GPU (popen is None) reports as running, not
  done (previously it read as completed and results were loaded too early).
- A squeue failure for a departed job is resolved via sacct instead of being
  treated as "still running forever".
- Killing a wrapped local process tears down the whole process group, so
  wrapper-spawned children are not orphaned.
"""

import asyncio
import io
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from shinka.launch import local, slurm
from shinka.launch.local import ProcessWithLogging, submit
from shinka.launch.slurm import SlurmJobName
from shinka.launch.scheduler import (
    JobScheduler,
    LocalJobConfig,
    SlurmCondaJobConfig,
)


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_local_job_pending_gpu_reports_running(monkeypatch):
    """popen is None (waiting for a GPU) must not be reported as finished."""
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-pending", {"popen": None})
    assert slurm.get_job_status("local-pending") == "local-pending"


def test_local_job_finished_reports_done(monkeypatch):
    finished = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-done", {"popen": finished})
    assert slurm.get_job_status("local-done") == ""


def test_local_job_running_reports_running(monkeypatch):
    running = SimpleNamespace(poll=lambda: None)
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-run", {"popen": running})
    assert slurm.get_job_status("local-run") == "local-run"


def test_squeue_active_returns_status(monkeypatch):
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda *a, **k: _FakeCompleted("12345 R eval\n"),
    )
    assert slurm.get_job_status("12345") == "12345 R eval"


def test_squeue_error_resolved_done_via_sacct(monkeypatch):
    def fake_run(cmd, *a, **k):
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "sacct":
            return _FakeCompleted("TIMEOUT|\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    # Departed/failed job resolved to done ("") instead of hanging as running.
    assert slurm.get_job_status("999") == ""


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "SUSPENDED", "COMPLETING"])
def test_squeue_error_active_sacct_state_reports_running(monkeypatch, state):
    def fake_run(cmd, *a, **k):
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "sacct":
            return _FakeCompleted(f"{state}|\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.get_job_status("999") == "999"


def test_squeue_error_transient_returns_unknown(monkeypatch):
    def fake_run(cmd, *a, **k):
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "sacct":
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    # Neither tool could answer -> unknown, so callers keep polling (not done).
    assert slurm.get_job_status("999") is None


def test_squeue_timeout_returns_unknown(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert kwargs["timeout"] == slurm.SLURM_COMMAND_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.get_job_status("999") is None


def test_sacct_timeout_returns_unknown(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert kwargs["timeout"] == slurm.SLURM_COMMAND_TIMEOUT_SECONDS
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(1, cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.get_job_status("999") is None


def test_monitor_raises_when_status_remains_unknown(monkeypatch):
    monkeypatch.setattr(slurm, "get_job_status", lambda _job_id: None)
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    with pytest.raises(slurm.JobStatusUnavailableError, match="status unknown"):
        slurm.monitor("999", poll_interval=0)


def test_cancellation_is_not_starved_by_submission_executor(monkeypatch):
    executor_blocked = threading.Event()
    release_executor = threading.Event()
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )

    def block_submission_executor():
        executor_blocked.set()
        release_executor.wait(timeout=2)

    scheduler.executor.submit(block_submission_executor)
    assert executor_blocked.wait(timeout=1)

    def fake_run(cmd, **kwargs):
        assert cmd == ["scancel", "123"]
        assert kwargs["timeout"] == slurm.SLURM_COMMAND_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    async def cancel():
        return await asyncio.wait_for(scheduler.cancel_job_async("123"), timeout=1)

    try:
        assert asyncio.run(cancel()) is True
    finally:
        release_executor.set()
        scheduler.shutdown()


def test_scheduler_cancels_and_reconciles_ambiguous_job_name(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "scancel":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = SlurmJobName("conda-unique")

    async def reconcile():
        assert await scheduler.cancel_job_async(target) is True
        assert await scheduler.is_job_terminal_async(target) is True

    try:
        asyncio.run(reconcile())
    finally:
        scheduler.shutdown()

    assert commands == [
        ["scancel", "--name", "conda-unique", "--quiet"],
        ["squeue", "--name", "conda-unique", "--noheader"],
        ["squeue", "--name", "conda-unique", "--noheader"],
    ]


def test_scheduler_retains_ambiguous_name_while_job_is_active(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )

    def fake_run(cmd, **kwargs):
        if cmd[0] == "scancel":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout="123 RUNNING conda-unique")

    monkeypatch.setattr(subprocess, "run", fake_run)

    async def cancel():
        return await scheduler.cancel_job_async(SlurmJobName("conda-unique"))

    try:
        assert asyncio.run(cancel()) is False
    finally:
        scheduler.shutdown()


def test_docker_image_preparation_has_bounded_subprocesses(monkeypatch):
    monkeypatch.setattr(slurm, "load_cache_manifest", lambda: {})

    def fake_run(cmd, **kwargs):
        assert cmd == ["docker", "pull", "example/image:latest"]
        assert kwargs["timeout"] == slurm.DOCKER_COMMAND_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert slurm.get_local_image("example/image:latest") == "example/image:latest"


def test_kill_terminates_child_process_group(tmp_path):
    """ProcessWithLogging.kill must reap wrapper-spawned children too."""
    pidfile = tmp_path / "child.pid"
    proc = submit(
        str(tmp_path),
        ["bash", "-c", f"sleep 60 & echo $! > {pidfile}; wait"],
    )

    deadline = time.time() + 5.0
    while not pidfile.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pidfile.exists(), "grandchild never started"
    child_pid = int(pidfile.read_text().strip())

    os.kill(child_pid, 0)  # child is alive (raises if not)

    proc.kill()

    # The grandchild should be gone once the group is killed.
    for _ in range(40):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild survived process-group kill")


def test_local_cancellation_reports_failure_when_process_cannot_be_killed(
    monkeypatch,
):
    class _UnkillableProcess:
        pid = 123
        returncode = None

        def kill(self):
            raise PermissionError("child kill denied")

        def poll(self):
            return None

    process = ProcessWithLogging(
        _UnkillableProcess(),
        (io.StringIO(), io.StringIO()),
        (threading.Thread(), threading.Thread()),
    )
    monkeypatch.setattr(local.os, "getpgid", lambda _pid: 123)
    monkeypatch.setattr(
        local.os,
        "killpg",
        lambda _pgid, _signal: (_ for _ in ()).throw(PermissionError("denied")),
    )
    scheduler = JobScheduler("local", LocalJobConfig(), max_workers=1)

    async def cancel():
        return await scheduler.cancel_job_async(process)

    try:
        assert asyncio.run(cancel()) is False
    finally:
        scheduler.shutdown()


def test_local_cancellation_signals_group_after_leader_exits(monkeypatch):
    class _ExitedLeader:
        pid = 123
        returncode = 0

        def kill(self):
            raise AssertionError("direct child is already gone")

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    signals = []

    def fake_killpg(process_group_id, sent_signal):
        signals.append((process_group_id, sent_signal))
        if sent_signal == 0:
            raise ProcessLookupError

    process = ProcessWithLogging(
        _ExitedLeader(),
        (io.StringIO(), io.StringIO()),
        (threading.Thread(), threading.Thread()),
    )
    monkeypatch.setattr(
        local.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError),
    )
    monkeypatch.setattr(local.os, "killpg", fake_killpg)

    assert process.kill() is True
    assert signals[0] == (123, local.signal.SIGKILL)


def test_local_timeout_retains_job_when_kill_is_unconfirmed(monkeypatch):
    class _RunningProcess:
        pid = 123
        returncode = None

        def poll(self):
            return None

    process = ProcessWithLogging(
        _RunningProcess(),
        (io.StringIO(), io.StringIO()),
        (threading.Thread(), threading.Thread()),
    )
    monkeypatch.setattr(process, "kill", lambda: False)
    scheduler = JobScheduler(
        "local",
        LocalJobConfig(time="00:00:01"),
        max_workers=1,
    )
    job = SimpleNamespace(
        job_id=process,
        start_time=time.time() - 2,
        generation=1,
    )

    try:
        assert scheduler.check_job_status(job) is True
    finally:
        scheduler.shutdown()


def test_local_monitor_raises_when_timeout_kill_is_unconfirmed(monkeypatch):
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        kill=lambda: False,
    )
    times = iter([0.0, 2.0])
    monkeypatch.setattr(local.time, "time", lambda: next(times))

    with pytest.raises(RuntimeError, match="Could not confirm termination"):
        local.monitor(process, ".", timeout="00:00:01")
