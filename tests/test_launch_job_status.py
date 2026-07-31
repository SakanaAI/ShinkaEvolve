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
import fcntl
import io
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from shinka.launch import _local_exec, _local_supervisor, local, slurm
from shinka.launch.local import (
    LocalProcessIdentity,
    ProcessWithLogging,
    find_local_process_identities,
    submit,
    submit_with_token,
)
from shinka.launch.slurm import SlurmJobName
from shinka.launch.scheduler import (
    JobScheduler,
    LocalJobConfig,
    SlurmCondaJobConfig,
)


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


@pytest.mark.parametrize(
    "state",
    [
        slurm.SlurmSubmissionRecoveryState.CONFIRMED_ABSENT,
        slurm.SlurmSubmissionRecoveryState.UNAVAILABLE,
        slurm.SlurmSubmissionRecoveryState.AMBIGUOUS,
    ],
)
def test_non_allocation_recovery_states_reject_job_ids(state):
    with pytest.raises(ValueError, match="does not match its state"):
        slurm.SlurmSubmissionRecovery(state, "junk")


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


def test_preallocated_local_token_recovers_live_process(tmp_path, monkeypatch):
    token = "a" * 32
    process = submit_with_token(
        str(tmp_path),
        [sys.executable, "-c", "import time; time.sleep(60)"],
        token,
    )

    try:
        with monkeypatch.context() as recovery_context:
            supervisor = local.psutil.Process(process.pid)
            recovery_context.setattr(
                local.psutil,
                "process_iter",
                lambda _attrs: [supervisor, *supervisor.children()],
            )
            recovered = find_local_process_identities(token)

            assert recovered == [process.identity]
    finally:
        process.kill()
        process.cleanup_logging()


def test_local_token_discovery_ignores_foreign_uid(monkeypatch):
    class _ForeignProcess:
        pid = 4321

        def uids(self):
            return SimpleNamespace(effective=os.geteuid() + 1)

        def environ(self):
            return {local.LOCAL_PROCESS_TOKEN_ENV: "a" * 32}

    monkeypatch.setattr(local.psutil, "process_iter", lambda _attrs: [_ForeignProcess()])
    monkeypatch.setattr(local.os, "getpgid", lambda _pid: 4321)

    assert find_local_process_identities("a" * 32) == []


def test_local_token_discovery_reports_blocked_same_uid(monkeypatch):
    class _BlockedProcess:
        pid = 4321

        def uids(self):
            return SimpleNamespace(effective=os.geteuid())

        def environ(self):
            raise local.psutil.AccessDenied(self.pid)

    monkeypatch.setattr(local.psutil, "process_iter", lambda _attrs: [_BlockedProcess()])

    with pytest.raises(RuntimeError, match="inspect same-user processes"):
        find_local_process_identities("a" * 32)


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
            return _FakeCompleted(f"123|{job_name}|{user_id}|COMPLETED\n")
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
            "--format=JobIDRaw,JobName%128,UID,State%64",
        ],
    ]


@pytest.mark.parametrize(
    ("accounting_output", "expected_state"),
    [
        (
            "123|different-name|1000|COMPLETED\n",
            slurm.SlurmSubmissionRecoveryState.UNAVAILABLE,
        ),
        (
            "123|conda-0123456789abcdef0123456789abcdef|1000|COMPLETED\n"
            "456|conda-0123456789abcdef0123456789abcdef|1000|COMPLETED\n",
            slurm.SlurmSubmissionRecoveryState.AMBIGUOUS,
        ),
        (
            "--user=other|conda-0123456789abcdef0123456789abcdef|1000|RUNNING\n",
            slurm.SlurmSubmissionRecoveryState.UNAVAILABLE,
        ),
        (
            "123|conda-0123456789abcdef0123456789abcdef|2000|RUNNING\n",
            slurm.SlurmSubmissionRecoveryState.UNAVAILABLE,
        ),
        (
            "123|conda-0123456789abcdef0123456789abcdef|1000|FUTURE_STATE\n",
            slurm.SlurmSubmissionRecoveryState.UNAVAILABLE,
        ),
    ],
)
def test_timed_out_submission_rejects_ambiguous_accounting_rows(
    monkeypatch,
    accounting_output,
    expected_state,
):
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda _cmd, **_kwargs: _FakeCompleted(accounting_output),
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

    recovery = slurm._recover_submission_from_sacct(
        "conda-0123456789abcdef0123456789abcdef"
    )

    assert recovery.state == expected_state


def test_sbatch_timeout_returns_completed_accounting_job(monkeypatch):
    job_name = "conda-0123456789abcdef0123456789abcdef"
    user_id = "1000"

    def fake_run(cmd, **kwargs):
        if "sbatch" in cmd:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if cmd[0] == "squeue":
            return _FakeCompleted("")
        if cmd[0] == "sacct":
            return _FakeCompleted(f"321|{job_name}|{user_id}|COMPLETED\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: user_id)

    assert slurm._submit_sbatch("job.sbatch", job_name) == "321"


def test_sbatch_marks_dispatch_immediately_before_parent_bound_launch(monkeypatch):
    events = []

    def fake_run(cmd, **_kwargs):
        events.append(("run", cmd))
        return _FakeCompleted("Submitted batch job 123\n")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    job_id = slurm._submit_sbatch(
        "job.sbatch",
        "conda-0123456789abcdef0123456789abcdef",
        on_dispatch_start=lambda: events.append(("dispatch", None)),
    )

    assert job_id == "123"
    assert [event for event, _ in events] == ["dispatch", "run"]
    assert events[1][1][1] == str(slurm.PARENT_DEATH_EXEC_PATH)


def test_sbatch_nonzero_exit_remains_ambiguous_after_dispatch(monkeypatch):
    original_error = subprocess.CalledProcessError(1, ["sbatch"])

    def fail_submission(_cmd, **_kwargs):
        raise original_error

    def remain_ambiguous(job_name):
        raise slurm.AmbiguousSlurmSubmissionError(job_name)

    monkeypatch.setattr(slurm.subprocess, "run", fail_submission)
    monkeypatch.setattr(slurm, "_reconcile_ambiguous_submission", remain_ambiguous)

    with pytest.raises(slurm.AmbiguousSlurmSubmissionError) as exc_info:
        slurm._submit_sbatch(
            "job.sbatch",
            "conda-0123456789abcdef0123456789abcdef",
        )

    assert exc_info.value.__cause__ is original_error


@pytest.mark.parametrize(
    "submission_error",
    [
        subprocess.TimeoutExpired(["sbatch"], 30),
        subprocess.CalledProcessError(1, ["sbatch"]),
    ],
)
def test_sbatch_reconciliation_error_remains_ambiguous(
    monkeypatch,
    submission_error,
):
    def fail_submission(_cmd, **_kwargs):
        raise submission_error

    def fail_reconciliation(_job_name):
        raise PermissionError("scheduler unavailable")

    monkeypatch.setattr(slurm.subprocess, "run", fail_submission)
    monkeypatch.setattr(
        slurm, "_reconcile_ambiguous_submission", fail_reconciliation
    )

    with pytest.raises(slurm.AmbiguousSlurmSubmissionError) as exc_info:
        slurm._submit_sbatch(
            "job.sbatch",
            "conda-0123456789abcdef0123456789abcdef",
        )

    assert isinstance(exc_info.value.cancel_target, SlurmJobName)


def test_parent_death_exec_prevents_late_submission_side_effect(tmp_path):
    child_pid_file = tmp_path / "submitter.pid"
    late_side_effect = tmp_path / "late-submit"
    child_code = (
        "from pathlib import Path; import os, time; "
        f"Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(0.5); "
        f"Path({str(late_side_effect)!r}).write_text('submitted')"
    )
    parent_code = (
        "import os, subprocess, sys; "
        "subprocess.run(["
        "sys.executable, "
        f"{str(slurm.PARENT_DEATH_EXEC_PATH)!r}, "
        "str(os.getpid()), sys.executable, '-c', "
        f"{child_code!r}"
        "])"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    child_pid = None
    try:
        deadline = time.time() + 5
        while not child_pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_file.read_text())

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        time.sleep(0.75)

        assert not late_side_effect.exists()
        try:
            child = local.psutil.Process(child_pid)
        except local.psutil.NoSuchProcess:
            child = None
        assert child is None or child.status() == local.psutil.STATUS_ZOMBIE
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
        return _FakeCompleted(f"456|{job_name}|1000|COMPLETED\n")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    assert slurm._recover_timed_out_submission(job_name) == "456"
    assert accounting_polls == 5


def test_submission_recovery_confirms_absence_after_successful_empty_checks(
    monkeypatch,
):
    monkeypatch.setattr(
        slurm.subprocess, "run", lambda _cmd, **_kwargs: _FakeCompleted("")
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    recovery = slurm.recover_submission_by_name(
        "conda-0123456789abcdef0123456789abcdef"
    )

    assert (
        recovery.state
        == slurm.SlurmSubmissionRecoveryState.CONFIRMED_ABSENT
    )
    assert recovery.job_id is None


def test_submission_recovery_reports_unavailable_after_query_failures(monkeypatch):
    def fail_query(cmd, **_kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fail_query)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    recovery = slurm.recover_submission_by_name(
        "conda-0123456789abcdef0123456789abcdef"
    )

    assert recovery.state == slurm.SlurmSubmissionRecoveryState.UNAVAILABLE
    assert recovery.job_id is None


def test_submission_recovery_uses_live_queue_allocation_without_accounting(
    monkeypatch,
):
    job_name = "conda-0123456789abcdef0123456789abcdef"

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "squeue":
            return _FakeCompleted("123|1000\n")
        pytest.fail("an active queue allocation is authoritative")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

    recovery = slurm.recover_submission_by_name(job_name)

    assert recovery == slurm.SlurmSubmissionRecovery.active("123")


def test_submission_recovery_requires_an_entire_clean_absence_window(monkeypatch):
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise subprocess.CalledProcessError(1, cmd)
        return _FakeCompleted("")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)

    recovery = slurm.recover_submission_by_name(
        "conda-0123456789abcdef0123456789abcdef"
    )

    assert recovery.state == slurm.SlurmSubmissionRecoveryState.UNAVAILABLE


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


def test_local_submission_hands_results_lease_through_supervisor_exec(
    monkeypatch,
    tmp_path,
):
    lease_read_fd, lease_write_fd = os.pipe()
    monkeypatch.setenv("LD_PRELOAD", "/attacker/preload.so")

    def inspect_popen(command, **kwargs):
        assert command[0].startswith("/proc/self/fd/")
        bash_fd = int(command[0].rsplit("/", 1)[1])
        assert command[1:5] == [
            "--noprofile",
            "--norc",
            "-p",
            "-c",
        ]
        assert command[-3:] == [sys.executable, "-c", "pass"]
        assert f"exec {lease_read_fd}>&-" in command[5]
        assert lease_read_fd in kwargs["pass_fds"]
        assert bash_fd in kwargs["pass_fds"]
        assert str(local.LOCAL_EXEC_PATH) in command
        assert kwargs["env"] == {
            local.LOCAL_PROCESS_TOKEN_ENV: "a" * 32,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        assert os.get_inheritable(lease_read_fd) is False
        raise RuntimeError("stop after exec handoff inspection")

    monkeypatch.setattr(local.subprocess, "Popen", inspect_popen)
    try:
        with pytest.raises(RuntimeError, match="handoff inspection"):
            submit_with_token(
                str(tmp_path),
                [sys.executable, "-c", "pass"],
                "a" * 32,
                ownership_lease_fd=lease_read_fd,
            )
    finally:
        os.close(lease_read_fd)
        os.close(lease_write_fd)


def test_local_submission_ignores_parent_close_error_after_spawn(
    monkeypatch,
    tmp_path,
):
    lease_read_fd, lease_write_fd = os.pipe()
    real_open_trusted_bash = local._open_trusted_bash
    real_close = local.os.close
    bash_fds = []
    close_failures = []

    def tracked_open_trusted_bash():
        bash_path, bash_fd = real_open_trusted_bash()
        bash_fds.append(bash_fd)
        return bash_path, bash_fd

    def fail_once_after_closing_bash(fd):
        real_close(fd)
        if fd in bash_fds and not close_failures:
            close_failures.append(fd)
            raise OSError("injected parent close failure")

    monkeypatch.setattr(local, "_open_trusted_bash", tracked_open_trusted_bash)
    monkeypatch.setattr(local.os, "close", fail_once_after_closing_bash)
    submitted = None
    try:
        submitted = submit_with_token(
            str(tmp_path),
            [sys.executable, "-c", "pass"],
            "a" * 32,
            ownership_lease_fd=lease_read_fd,
        )
        assert submitted.wait(timeout=5) == 0
        assert close_failures == bash_fds
    finally:
        if submitted is not None:
            submitted.cleanup_logging()
        real_close(lease_read_fd)
        real_close(lease_write_fd)


def test_local_exec_restores_environment_after_lease_handoff(monkeypatch):
    environment_read_fd, environment_write_fd = os.pipe()
    os.write(
        environment_write_fd,
        b'{"SHINKA_LOCAL_JOB_TOKEN":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"LD_LIBRARY_PATH":"/evaluator/lib"}',
    )
    os.close(environment_write_fd)
    observed = {}

    def inspect_exec(command, argv, environment):
        observed.update(command=command, argv=argv, environment=environment)
        raise RuntimeError("stop after environment inspection")

    monkeypatch.setattr(_local_exec.os, "execve", inspect_exec)

    with pytest.raises(RuntimeError, match="environment inspection"):
        _local_exec.main(environment_read_fd, ["/evaluator", "--flag"])

    assert observed == {
        "command": "/evaluator",
        "argv": ["/evaluator", "--flag"],
        "environment": {
            "SHINKA_LOCAL_JOB_TOKEN": "a" * 32,
            "LD_LIBRARY_PATH": "/evaluator/lib",
        },
    }


def test_local_lease_handoff_fails_clearly_without_bash(monkeypatch, tmp_path):
    lease_read_fd, lease_write_fd = os.pipe()
    monkeypatch.setattr(local.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        local.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not launch without Bash"),
    )
    try:
        with pytest.raises(RuntimeError, match="requires Bash"):
            submit_with_token(
                str(tmp_path),
                [sys.executable, "-c", "pass"],
                "a" * 32,
                ownership_lease_fd=lease_read_fd,
            )
    finally:
        os.close(lease_read_fd)
        os.close(lease_write_fd)


def test_local_lease_handoff_rejects_replaceable_bash_path(monkeypatch, tmp_path):
    replaceable_dir = tmp_path / "replaceable"
    replaceable_dir.mkdir()
    replaceable_dir.chmod(0o777)
    fake_bash = replaceable_dir / "bash"
    fake_bash.write_bytes(b"not actually bash")
    fake_bash.chmod(0o755)
    monkeypatch.setattr(local.shutil, "which", lambda _name: str(fake_bash))

    with pytest.raises(PermissionError, match="replaceable"):
        local._open_trusted_bash()


def test_stalled_supervisor_startup_does_not_retain_parent_lease(
    monkeypatch,
    tmp_path,
):
    stalled_supervisor = tmp_path / "stalled_supervisor.py"
    startup_visible = tmp_path / "startup-visible"
    stalled_supervisor.write_text(
        "from pathlib import Path; import time; "
        f"Path({str(startup_visible)!r}).write_text('ready'); time.sleep(60)\n"
    )
    exported_function_marker = tmp_path / "exported-function-ran"
    xtrace_marker = tmp_path / "xtrace-ran"
    monkeypatch.setenv(
        "BASH_FUNC_exec%%",
        f'() {{ touch {exported_function_marker}; builtin exec "$@"; }}',
    )
    monkeypatch.setenv("SHELLOPTS", "xtrace")
    monkeypatch.setenv("PS4", f"$(touch {xtrace_marker})")
    monkeypatch.setattr(local, "LOCAL_SUPERVISOR_PATH", stalled_supervisor)
    original_popen = local.subprocess.Popen
    spawned_pids = []

    def tracking_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        spawned_pids.append(process.pid)
        return process

    monkeypatch.setattr(local.subprocess, "Popen", tracking_popen)
    token = "a" * 32
    lock_path = tmp_path / "runner.lock"
    lease_fd: int | None = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600
    )
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    unrelated_read_fd, initial_unrelated_write_fd = os.pipe()
    unrelated_write_fd: int | None = initial_unrelated_write_fd
    os.set_inheritable(unrelated_write_fd, True)
    os.set_blocking(unrelated_read_fd, False)
    launch_errors = []

    def launch():
        try:
            submit_with_token(
                str(tmp_path / "logs"),
                [sys.executable, "-c", "pass"],
                token,
                ownership_lease_fd=lease_fd,
            )
        except BaseException as error:
            launch_errors.append(error)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    identities = []
    replacement_fd = None
    try:
        deadline = time.time() + 5
        while not startup_visible.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert startup_visible.exists()
        assert not exported_function_marker.exists()
        assert not xtrace_marker.exists()
        os.close(unrelated_write_fd)
        unrelated_write_fd = None
        assert os.read(unrelated_read_fd, 1) == b""

        os.close(lease_fd)
        lease_fd = None
        replacement_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
        fcntl.flock(replacement_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        assert len(spawned_pids) == 1
        supervisor = local.psutil.Process(spawned_pids[0])
        assert supervisor.environ()[local.LOCAL_PROCESS_TOKEN_ENV] == token
        identities = [LocalProcessIdentity(spawned_pids[0], token)]
        os.killpg(identities[0].pid, signal.SIGKILL)
        launch_thread.join(timeout=5)
        assert not launch_thread.is_alive()
        assert launch_errors
    finally:
        for identity in identities:
            try:
                os.killpg(identity.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if lease_fd is not None:
            os.close(lease_fd)
        if replacement_fd is not None:
            os.close(replacement_fd)
        os.close(unrelated_read_fd)
        if unrelated_write_fd is not None:
            os.close(unrelated_write_fd)
        launch_thread.join(timeout=5)


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
        return SimpleNamespace(stdout="123|1000\n123|1000\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

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
            "--user",
            "1000",
            "--noheader",
            "--format=%A|%U",
        ]
    ]


@pytest.mark.parametrize("stdout", ["malformed\n", "999|2000\n", "bad|1000\n"])
def test_scheduler_treats_unexpected_named_job_output_as_unknown(
    monkeypatch,
    stdout,
):
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout),
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")

    assert slurm.get_job_ids_by_name("conda-" + "a" * 32) is None


def test_scheduler_keeps_absent_submission_name_ambiguous(monkeypatch):
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
    monkeypatch.setattr(
        "shinka.launch.scheduler._get_current_user_id", lambda: "1000"
    )
    monkeypatch.setattr(slurm, "_get_current_user_id", lambda: "1000")
    monkeypatch.setattr(slurm.time, "sleep", lambda _seconds: None)
    job_name = "conda-" + "a" * 32
    target = SlurmJobName(job_name)

    async def reconcile():
        assert await scheduler.cancel_job_async(target) is False
        assert await scheduler.is_job_terminal_async(target) is False

    try:
        asyncio.run(reconcile())
    finally:
        scheduler.shutdown()

    assert commands[0] == [
        "scancel",
        "--name",
        job_name,
        "--user",
        "1000",
        "--quiet",
    ]
    assert {command[0] for command in commands[1:]} == {"squeue", "sacct"}


def test_scheduler_confirms_terminal_submission_name(monkeypatch):
    scheduler = JobScheduler(
        "slurm_conda",
        SlurmCondaJobConfig(),
        max_workers=1,
    )
    target = SlurmJobName("conda-" + "a" * 32)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "shinka.launch.scheduler.recover_submission_by_name",
        lambda _job_name: slurm.SlurmSubmissionRecovery.terminal("123"),
    )

    async def reconcile():
        assert await scheduler.cancel_job_async(target) is True
        assert await scheduler.is_job_terminal_async(target) is True

    try:
        asyncio.run(reconcile())
    finally:
        scheduler.shutdown()


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


def test_local_identity_survives_evaluator_environment_sanitization(tmp_path):
    ready_file = tmp_path / "sanitized.ready"
    proc = submit(
        str(tmp_path),
        [
            "env",
            "-i",
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import os, time; "
                f"Path({str(ready_file)!r}).write_text("
                "os.environ.get('SHINKA_LOCAL_JOB_TOKEN', 'missing')); "
                "time.sleep(60)"
            ),
        ],
    )

    try:
        deadline = time.time() + 5
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready_file.read_text() == "missing"
        assert proc.kill() is True
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait(timeout=5)
        proc.cleanup_logging()


def test_local_supervisor_preserves_synchronous_launch_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        submit(str(tmp_path), ["/definitely/missing/shinka-command"])


def test_local_supervisor_preserves_permission_launch_errors(tmp_path):
    command = tmp_path / "not-executable"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o600)

    with pytest.raises(PermissionError, match="Permission denied"):
        submit(str(tmp_path), [str(command)])


@pytest.mark.parametrize(
    "termination",
    ["api", "supervisor_signal", "supervisor_sigkill"],
)
def test_supervisor_termination_reaps_sanitized_evaluator(tmp_path, termination):
    evaluator_pid_file = tmp_path / "evaluator.pid"
    proc = submit(
        str(tmp_path),
        [
            "env",
            "-i",
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import os, time; "
                f"Path({str(evaluator_pid_file)!r}).write_text(str(os.getpid())); "
                "time.sleep(60)"
            ),
        ],
    )

    try:
        deadline = time.time() + 5
        while not evaluator_pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        evaluator_pid = int(evaluator_pid_file.read_text())

        if termination == "api":
            proc.terminate()
        elif termination == "supervisor_sigkill":
            os.kill(proc.pid, signal.SIGKILL)
        else:
            os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)

        try:
            evaluator = local.psutil.Process(evaluator_pid)
        except local.psutil.NoSuchProcess:
            pass
        else:
            deadline = time.time() + 2
            while (
                evaluator.is_running()
                and evaluator.status() != local.psutil.STATUS_ZOMBIE
                and time.time() < deadline
            ):
                time.sleep(0.01)
            assert (
                not evaluator.is_running()
                or evaluator.status() == local.psutil.STATUS_ZOMBIE
            )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait(timeout=5)
        proc.cleanup_logging()


def test_supervisor_sigkill_reaps_wrapper_grandchild(tmp_path):
    grandchild_pid_file = tmp_path / "grandchild.pid"
    proc = submit(
        str(tmp_path),
        [
            "bash",
            "-c",
            f"sleep 60 & echo $! > {grandchild_pid_file}; wait",
        ],
    )

    grandchild_pid = None
    try:
        deadline = time.time() + 5
        while not grandchild_pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        grandchild_pid = int(grandchild_pid_file.read_text())
        os.kill(grandchild_pid, 0)

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

        try:
            grandchild = local.psutil.Process(grandchild_pid)
        except local.psutil.NoSuchProcess:
            grandchild = None
        if grandchild is not None:
            deadline = time.time() + 2
            while (
                grandchild.is_running()
                and grandchild.status() != local.psutil.STATUS_ZOMBIE
                and time.time() < deadline
            ):
                time.sleep(0.01)
            assert (
                not grandchild.is_running()
                or grandchild.status() == local.psutil.STATUS_ZOMBIE
            )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        if grandchild_pid is not None:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.cleanup_logging()


def test_stopped_group_guard_is_killed_within_bound(monkeypatch):
    read_fd, write_fd = os.pipe()
    guard_pid = os.fork()
    if guard_pid == 0:
        os.close(write_fd)
        os.kill(os.getpid(), signal.SIGSTOP)
        os.read(read_fd, 1)
        os._exit(0)
    os.close(read_fd)
    monkeypatch.setattr(
        _local_supervisor, "GROUP_GUARD_STOP_TIMEOUT_SECONDS", 0.05
    )

    start = time.monotonic()
    assert _local_supervisor._stop_group_guard(guard_pid, write_fd) is False
    assert time.monotonic() - start < 0.5
    with pytest.raises(ChildProcessError):
        os.waitpid(guard_pid, os.WNOHANG)


def test_supervisor_survives_closed_readiness_pipe(tmp_path):
    status_read_fd, status_write_fd = os.pipe()
    os.close(status_read_fd)
    ready_file = tmp_path / "ready"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(local.LOCAL_SUPERVISOR_PATH),
            str(status_write_fd),
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import time; "
                f"Path({str(ready_file)!r}).write_text('ready'); time.sleep(60)"
            ),
        ],
        env={**os.environ, local.LOCAL_PROCESS_TOKEN_ENV: "a" * 32},
        start_new_session=True,
        pass_fds=(status_write_fd,),
    )
    os.close(status_write_fd)

    try:
        deadline = time.time() + 5
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()
        assert supervisor.poll() is None
    finally:
        os.kill(supervisor.pid, signal.SIGTERM)
        supervisor.wait(timeout=5)


def test_local_supervisor_readiness_has_timeout(monkeypatch, tmp_path):
    stalled_supervisor = tmp_path / "stalled_supervisor.py"
    stalled_supervisor.write_text("import time; time.sleep(60)\n")
    monkeypatch.setattr(local, "LOCAL_SUPERVISOR_PATH", stalled_supervisor)
    monkeypatch.setattr(local, "LOCAL_SUPERVISOR_START_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(TimeoutError, match="did not report launch"):
        submit(str(tmp_path), [sys.executable, "-c", "pass"])


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_local_submission_fails_closed_off_linux(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(local.sys, "platform", platform)

    with pytest.raises(RuntimeError, match="requires Linux"):
        submit(str(tmp_path / "logs"), [sys.executable, "-c", "pass"])

    assert not (tmp_path / "logs").exists()


def test_local_supervisor_waits_for_entire_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    proc = submit(
        str(tmp_path),
        [
            "bash",
            "-c",
            f"sleep 1 & echo $! > {child_pid_file}; exit 7",
        ],
    )

    child_pid = None
    try:
        deadline = time.time() + 5
        while not child_pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_file.read_text())
        os.kill(child_pid, 0)

        supervisor = local.psutil.Process(proc.pid)
        deadline = time.time() + 5
        evaluator_children = supervisor.children()
        while (
            any(
                child.environ().get(local.LOCAL_PROCESS_TOKEN_ENV)
                != proc.identity.token
                for child in evaluator_children
            )
            and time.time() < deadline
        ):
            time.sleep(0.01)
            evaluator_children = supervisor.children()
        assert all(
            child.environ().get(local.LOCAL_PROCESS_TOKEN_ENV) == proc.identity.token
            for child in evaluator_children
        )
        assert proc.poll() is None

        assert proc.wait(timeout=5) == 7
        try:
            child = local.psutil.Process(child_pid)
        except local.psutil.NoSuchProcess:
            pass
        else:
            assert child.status() == local.psutil.STATUS_ZOMBIE
    finally:
        if proc.poll() is None:
            terminated = proc.kill()
            if not terminated and child_pid is not None:
                os.kill(child_pid, signal.SIGKILL)
        proc.cleanup_logging()


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
