"""Regression tests for job-status detection and process-group teardown.

Covers:
- A local job still waiting for a GPU (popen is None) reports as running, not
  done (previously it read as completed and results were loaded too early).
- A squeue failure for a departed job is resolved via sacct instead of being
  treated as "still running forever".
- Killing a wrapped local process tears down the whole process group, so
  wrapper-spawned children are not orphaned.
"""

import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from shinka.launch import slurm
from shinka.launch.local import submit


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
            return _FakeCompleted("TIMEOUT\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    # Departed/failed job resolved to done ("") instead of hanging as running.
    assert slurm.get_job_status("999") == ""


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


def test_env_exports_shell_quote_injection():
    """A malicious eval_env value must be shell-quoted, not spliced as commands."""
    from shinka.launch.slurm import _render_env_exports, _render_env_docker_flags

    exports = _render_env_exports({"FOO": "a; rm -rf /tmp/x"})
    # The dangerous ';' is inside a single-quoted value, not a command separator.
    assert exports == "export FOO='a; rm -rf /tmp/x'"

    flags = _render_env_docker_flags({"BAR": "b $(whoami)"})
    assert flags == "-e 'BAR=b $(whoami)'"


@pytest.mark.parametrize(
    "renderer",
    [
        pytest.param("exports", id="shell-exports"),
        pytest.param("docker", id="docker-flags"),
    ],
)
def test_env_renderers_reject_shell_metacharacters_in_names(renderer):
    from shinka.launch.slurm import _render_env_docker_flags, _render_env_exports

    render = _render_env_exports if renderer == "exports" else _render_env_docker_flags

    with pytest.raises(ValueError, match="Invalid environment variable name"):
        render({"SAFE; touch /tmp/injected; #": "value"})


def test_strip_provider_secrets_opt_in(monkeypatch):
    """SHINKA_STRIP_EVAL_SECRETS removes provider credentials from eval env."""
    from shinka.launch.local import (
        _strip_provider_secrets,
        _should_strip_eval_secrets,
    )

    env = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-secret",
        "ANTHROPIC_API_KEY": "sk-ant",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "HF_TOKEN": "hf",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    stripped = _strip_provider_secrets(env)
    assert stripped == {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"}

    monkeypatch.delenv("SHINKA_STRIP_EVAL_SECRETS", raising=False)
    assert _should_strip_eval_secrets() is False
    monkeypatch.setenv("SHINKA_STRIP_EVAL_SECRETS", "1")
    assert _should_strip_eval_secrets() is True


def test_local_submit_strips_inherited_secrets_then_applies_overrides(
    monkeypatch, tmp_path
):
    from shinka.launch import local

    captured = {}

    class FakeProcess:
        pid = 123
        returncode = None
        stdout = object()
        stderr = object()

    class FakeThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    def popen(command, **kwargs):
        captured.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setenv("SHINKA_STRIP_EVAL_SECRETS", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-openai")
    monkeypatch.setenv("SAFE_SETTING", "preserved")
    monkeypatch.setattr(local.subprocess, "Popen", popen)
    monkeypatch.setattr(local.threading, "Thread", FakeThread)

    process = local.submit(
        str(tmp_path),
        ["python", "evaluate.py"],
        env_overrides={"OPENAI_API_KEY": "explicit-openai"},
    )
    process.cleanup_logging()

    assert "ANTHROPIC_API_KEY" not in captured
    assert captured["OPENAI_API_KEY"] == "explicit-openai"
    assert captured["SAFE_SETTING"] == "preserved"
