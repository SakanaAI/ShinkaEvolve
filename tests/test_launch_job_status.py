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
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from shinka.launch import local, slurm
from shinka.launch.local import LocalProcessIdentity, ProcessWithLogging, submit
from shinka.launch.slurm import SlurmJobName
from shinka.launch.scheduler import (
    JobScheduler,
    LocalJobConfig,
    SlurmCondaJobConfig,
)


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_local_process_identity_does_not_signal_token_mismatch():
    env = os.environ.copy()
    env["SHINKA_LOCAL_JOB_TOKEN"] = "actual-token"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=env,
        start_new_session=True,
    )
    identity = LocalProcessIdentity(pid=process.pid, token="different-token")

    try:
        assert identity.kill() is False
        assert process.poll() is None
    finally:
        os.killpg(process.pid, 9)
        process.wait(timeout=5)


def test_local_process_identity_reauthenticates_each_group_snapshot(monkeypatch):
    class _Process:
        def __init__(self, pid):
            self.pid = pid
            self.kill_calls = 0

        def kill(self):
            self.kill_calls += 1

    owned_process = _Process(123)
    replacement_process = _Process(456)
    snapshots = iter(
        [
            ([owned_process], True, False),
            ([replacement_process], False, False),
        ]
    )
    monkeypatch.setattr(
        LocalProcessIdentity,
        "_process_group_members",
        lambda _identity: next(snapshots),
    )

    identity = LocalProcessIdentity(pid=123, token="a" * 32)

    assert identity.kill() is False
    assert owned_process.kill_calls == 1
    assert replacement_process.kill_calls == 0


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


def test_sacct_success_without_rows_returns_unknown(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(1, cmd)
        return _FakeCompleted("")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.get_job_status("999") is None


def test_timed_out_submission_recovers_fast_completed_job_from_sacct(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    user_id = "1000"
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["timeout"] == slurm.SUBMISSION_RECOVERY_COMMAND_TIMEOUT_SECONDS
        if cmd[0] == "squeue":
            return _FakeCompleted("")
        if cmd[0] == "sacct":
            return _FakeCompleted(f"123|{job_name}|{user_id}\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: user_id)
    monkeypatch.setattr(
        slurm.time,
        "sleep",
        lambda _seconds: pytest.fail("recovery should finish on first attempt"),
    )

    assert slurm._recover_timed_out_submission(job_name) == "123"
    assert commands == [
        [
            "squeue",
            "--name",
            job_name,
            "--user",
            user_id,
            "--noheader",
            "--format=%A|%U",
        ],
        [
            "sacct",
            "--name",
            job_name,
            "--user",
            user_id,
            "--starttime=now-10minutes",
            "--allocations",
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName%128,UID",
        ],
    ]


@pytest.mark.parametrize(
    "accounting_output",
    [
        "123|different-name|1000\n",
        "123|conda-0123456789abcdef0123456789abcdef|1000\n"
        "456|conda-0123456789abcdef0123456789abcdef|1000\n",
        "--user=other|conda-0123456789abcdef0123456789abcdef|1000\n",
        "123|conda-0123456789abcdef0123456789abcdef|2000\n",
    ],
)
def test_timed_out_submission_rejects_ambiguous_accounting_rows(
    monkeypatch,
    accounting_output,
):
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda _cmd, **_kwargs: _FakeCompleted(accounting_output),
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

    recovered = slurm._recover_submission_from_sacct(
        "conda-0123456789abcdef0123456789abcdef"
    )

    assert recovered is None


def test_sbatch_timeout_returns_completed_accounting_job(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    user_id = "1000"

    def fake_run(cmd, **kwargs):
        if cmd[0] == "sbatch":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if cmd[0] == "squeue":
            return _FakeCompleted("")
        if cmd[0] == "sacct":
            return _FakeCompleted(f"321|{job_name}|{user_id}\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: user_id)

    assert slurm._submit_sbatch("job.sbatch", job_name) == "321"


def test_timed_out_submission_waits_for_delayed_accounting(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    accounting_polls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal accounting_polls
        if cmd[0] == "squeue":
            return _FakeCompleted("")
        accounting_polls += 1
        if accounting_polls < 5:
            return _FakeCompleted("")
        return _FakeCompleted(f"456|{job_name}|1000\n")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    assert slurm._recover_timed_out_submission(job_name) == "456"
    assert accounting_polls == 5


def test_timed_out_submission_does_not_collapse_multiple_queue_ids(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "squeue":
            return _FakeCompleted("123|1000\n456|1000\n")
        pytest.fail("ambiguous queue result must not fall through to accounting")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

    with pytest.raises(slurm.AmbiguousSlurmSubmissionError):
        slurm._recover_timed_out_submission(job_name)


def test_cancelled_submission_rechecks_accounting_before_resolution(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    recovery_results = iter([None, "789"])
    monkeypatch.setattr(
        slurm,
        "_recover_timed_out_submission",
        lambda _job_name: next(recovery_results),
    )
    monkeypatch.setattr(slurm, "_cancel_ambiguous_submission", lambda _job_name: True)
    monkeypatch.setattr(slurm, "get_job_status_by_name", lambda _job_name: "")

    assert slurm._reconcile_ambiguous_submission(job_name) == "789"


def test_cancelled_submission_without_accounting_stays_ambiguous(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    monkeypatch.setattr(
        slurm,
        "_recover_timed_out_submission",
        lambda _job_name: None,
    )
    monkeypatch.setattr(slurm, "_cancel_ambiguous_submission", lambda _job_name: True)
    monkeypatch.setattr(slurm, "get_job_status_by_name", lambda _job_name: "")

    with pytest.raises(slurm.AmbiguousSlurmSubmissionError):
        slurm._reconcile_ambiguous_submission(job_name)


def test_scheduler_preserves_unknown_slurm_status(monkeypatch):
    monkeypatch.setattr(slurm, "get_job_status", lambda _job_id: None)
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )

    try:
        status = scheduler.check_job_status(SimpleNamespace(job_id="999"))
    finally:
        scheduler.shutdown()

    assert status is None


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

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["timeout"] == slurm.SLURM_COMMAND_TIMEOUT_SECONDS
        if cmd[0] == "scancel":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    async def cancel():
        return await asyncio.wait_for(scheduler.cancel_job_async("123"), timeout=1)

    try:
        assert asyncio.run(cancel()) is True
    finally:
        release_executor.set()
        scheduler.shutdown()

    assert commands == [
        ["scancel", "--", "123"],
        ["squeue", "-j", "123", "--noheader"],
    ]


def test_scheduler_retains_known_job_id_until_it_disappears(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )
    status_results = iter(["RUNNING", ""])
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "scancel":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout=next(status_results))

    monkeypatch.setattr(subprocess, "run", fake_run)

    async def reconcile():
        assert await scheduler.cancel_job_async("123") is False
        assert await scheduler.cancel_job_async("123") is True

    try:
        asyncio.run(reconcile())
    finally:
        scheduler.shutdown()

    assert commands == [
        ["scancel", "--", "123"],
        ["squeue", "-j", "123", "--noheader"],
        ["scancel", "--", "123"],
        ["squeue", "-j", "123", "--noheader"],
    ]


def test_scheduler_rejects_invalid_slurm_job_id_before_subprocess(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not invoke scancel"),
    )

    try:
        assert asyncio.run(scheduler.cancel_job_async("--user=other")) is False
    finally:
        scheduler.shutdown()


def test_local_post_launch_failure_reaps_process(monkeypatch, tmp_path):
    spawned_pid = None
    original_popen = local.subprocess.Popen

    def tracking_popen(*args, **kwargs):
        nonlocal spawned_pid
        process = original_popen(*args, **kwargs)
        spawned_pid = process.pid
        return process

    monkeypatch.setattr(local.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(
        local.threading.Thread,
        "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        submit(
            str(tmp_path),
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )

    assert spawned_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(spawned_pid, 0)


@pytest.mark.parametrize(
    "job_id",
    ["123_4", "123+1", "123.batch", "123_4.extern", "123+1.2"],
)
def test_scheduler_accepts_supported_composite_slurm_job_ids(monkeypatch, job_id):
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

    try:
        assert asyncio.run(scheduler.cancel_job_async(job_id)) is True
    finally:
        scheduler.shutdown()

    assert commands[0] == ["scancel", "--", job_id]


def test_scheduler_finds_job_id_by_unique_submission_name(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="123\n123\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        job_ids = asyncio.run(
            scheduler.get_job_ids_by_name_async(
                "conda-0123456789abcdef0123456789abcdef"
            )
        )
    finally:
        scheduler.shutdown()

    assert job_ids == ["123"]
    assert commands == [
        [
            "squeue",
            "--name",
            "conda-0123456789abcdef0123456789abcdef",
            "--noheader",
            "--format=%A",
        ]
    ]


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
        [
            "bash",
            "-c",
            f'env -i PATH="$PATH" sleep 60 & echo $! > {pidfile}; wait',
        ],
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


def test_local_cancellation_uses_durable_identity(monkeypatch):
    class _RunningLeader:
        pid = 123
        returncode = None

        def wait(self, timeout):
            self.returncode = -9
            return self.returncode

        def poll(self):
            return self.returncode

    class _Identity:
        def __init__(self):
            self.kill_calls = 0

        def kill(self):
            self.kill_calls += 1
            return True

        def is_terminated(self):
            return True

    identity = _Identity()
    process = ProcessWithLogging(
        _RunningLeader(),
        (io.StringIO(), io.StringIO()),
        (threading.Thread(), threading.Thread()),
        identity=identity,
    )
    monkeypatch.setattr(
        local.os, "killpg", lambda *_args: pytest.fail("unsafe process-group signal")
    )

    assert process.kill() is True
    assert identity.kill_calls == 1


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
