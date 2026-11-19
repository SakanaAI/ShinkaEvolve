"""Helpers shared by the job lifecycle tests."""

import os
import signal
import subprocess
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, Callable

import psutil
import pytest

PURGED_JOB_ERROR = "slurm_load_jobs error: Invalid job id specified\n"


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds
        # Fail instead of spinning forever when the code under test never stops waiting.
        assert self.slept < 86400.0, "slept for more than a day of fake time"

    def install(self, monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
        for module in modules:
            monkeypatch.setattr(
                module,
                "time",
                SimpleNamespace(monotonic=self.monotonic, sleep=self.sleep),
            )


def fake_squeue(
    monkeypatch: pytest.MonkeyPatch, reply: Callable[[list[str]], Any]
) -> list[dict[str, Any]]:
    """Answer squeue with ``reply(job_ids)``: active ids, or an exception to raise."""
    calls: list[dict[str, Any]] = []

    def run(cmd, **kwargs):
        assert cmd[0] == "squeue"
        flag = "--jobs" if "--jobs" in cmd else "-j"
        job_ids = cmd[cmd.index(flag) + 1].split(",")
        calls.append(
            {"cmd": cmd, "job_ids": job_ids, "timeout": kwargs.get("timeout")}
        )
        outcome = reply(job_ids)
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(outcome), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def squeue_failure(stderr: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(1, ["squeue"], stderr=stderr)


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5.0
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def is_alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def kill_leftovers(*pids: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_evaluator_with_worker(directory: Path, *, runs_for: float = 30.0) -> Path:
    """An evaluator that starts a worker, as `conda run` or a pool would.

    The worker outlives an evaluator that runs for less than 30 seconds.
    """
    path = directory / "evaluate.py"
    path.write_text(
        dedent(f"""
            import subprocess
            import sys
            import time
            from pathlib import Path

            results = Path(sys.argv[sys.argv.index("--results_dir") + 1])
            worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            pending = results / "worker.tmp"
            pending.write_text(str(worker.pid))
            pending.rename(results / "worker.pid")
            time.sleep({runs_for})
            """)
    )
    return path


def worker_pid(results_dir: Path) -> int:
    pid_file = results_dir / "worker.pid"
    assert wait_until(pid_file.exists)
    return int(pid_file.read_text())
