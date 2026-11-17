import os
import signal
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Callable

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
