import os
import signal
import sys
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock

import psutil
import pytest

from shinka.launch import local


def _wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5.0
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _is_alive(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_local_timeout_kills_evaluator_descendants_and_reaps_root(
    tmp_path: Path,
) -> None:
    child_script = tmp_path / "child.py"
    child_script.write_text(
        dedent("""
            import subprocess
            import sys
            import time
            from pathlib import Path

            grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            pid_file = Path(sys.argv[1])
            pending = pid_file.with_suffix(".tmp")
            pending.write_text(str(grandchild.pid))
            pending.rename(pid_file)
            time.sleep(30)
            """)
    )
    pid_file = tmp_path / "grandchild.pid"
    root = local.submit(
        str(tmp_path),
        [
            sys.executable,
            "-c",
            "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))",
            sys.executable,
            str(child_script),
            str(pid_file),
        ],
        timeout="00:00:00",
    )
    descendants = []
    try:
        assert _wait_until(pid_file.exists)
        grandchild_pid = int(pid_file.read_text())
        descendants = psutil.Process(root.pid).children(recursive=True)
        assert len(descendants) == 2
        assert grandchild_pid in {process.pid for process in descendants}

        with pytest.raises(TimeoutError, match="00:00:00"):
            local.monitor(root, str(tmp_path), timeout="00:00:00")

        assert root.returncode == -signal.SIGKILL
        with pytest.raises(ChildProcessError):
            os.waitpid(root.pid, os.WNOHANG)
        assert _wait_until(lambda: all(not _is_alive(p) for p in descendants))
    finally:
        if root.poll() is None:
            descendants = psutil.Process(root.pid).children(recursive=True)
            root.kill()
            root.wait()
        for process in descendants:
            if _is_alive(process):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        root.cleanup_logging()


@pytest.mark.parametrize("finished_at", [0.99, 1.01], ids=["before", "after"])
def test_local_monitor_accepts_only_completion_observed_by_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finished_at: float
) -> None:
    elapsed = 0.0

    def advance_clock(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    def wait(*, timeout: float) -> int:
        nonlocal elapsed
        assert timeout == pytest.approx(1.0)
        # Simulate a wait that observes an exit despite waking after its deadline.
        elapsed = finished_at
        return 0

    monkeypatch.setattr(
        local,
        "time",
        SimpleNamespace(
            time=lambda: elapsed, monotonic=lambda: elapsed, sleep=advance_clock
        ),
    )
    (tmp_path / "correct.json").write_text('{"correct": true}')
    (tmp_path / "metrics.json").write_text('{"combined_score": 42}')
    process = Mock(returncode=None)
    process.poll.side_effect = lambda: 0 if elapsed >= finished_at else None
    process.wait.side_effect = wait

    if finished_at <= 1.0:
        result = local.monitor(
            process, str(tmp_path), poll_interval=2.0, timeout="00:00:01"
        )
        assert result["metrics"]["combined_score"] == 42
        process.kill_for_timeout.assert_not_called()
    else:
        with pytest.raises(TimeoutError, match="00:00:01"):
            local.monitor(process, str(tmp_path), poll_interval=2.0, timeout="00:00:01")
        process.kill_for_timeout.assert_called_once_with()
    process.cleanup_logging.assert_called_once_with()
