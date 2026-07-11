from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Iterator, TextIO, cast
from unittest.mock import Mock

import psutil
import pytest

from shinka.launch import local
from shinka.launch.scheduler import JobScheduler, LocalJobConfig

_GRANDCHILD_CODE = """
import os
from pathlib import Path
import signal
import sys

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
signal.pause()
"""

_CHILD_CODE = """
import subprocess
import sys

grandchild = subprocess.Popen(
    [sys.executable, "-c", sys.argv[2], sys.argv[1]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
grandchild.wait()
"""


class _ExpiredClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        self.now += 2.0
        return self.now

    def sleep(self, _seconds: float) -> None:
        raise AssertionError("timeout should be detected before sleeping")


def _wait_for_ready_pid(path: Path, process: local.ProcessWithLogging) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="ascii"))
        except (FileNotFoundError, ValueError):
            if process.poll() is not None:
                raise AssertionError("child exited before its grandchild was ready")
            time.sleep(0.01)
    raise AssertionError("grandchild did not signal readiness")


def _wait_for_process_exit(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.01)
    raise AssertionError(f"descendant process {pid} is still running")


@pytest.fixture
def process_tree(
    tmp_path: Path,
) -> Iterator[tuple[local.ProcessWithLogging, int, Path]]:
    if os.name != "posix":
        pytest.skip("POSIX process groups are required")

    log_dir = tmp_path / "logs"
    ready_path = tmp_path / "grandchild.ready"
    process = local.submit(
        str(log_dir),
        [sys.executable, "-c", _CHILD_CODE, str(ready_path), _GRANDCHILD_CODE],
    )
    grandchild_pid = _wait_for_ready_pid(ready_path, process)

    try:
        yield process, grandchild_pid, log_dir
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.process.kill()
        process.wait(timeout=5)
        process.cleanup_logging()


def test_direct_child_signals_delegate_to_popen() -> None:
    process_mock = Mock(spec=subprocess.Popen)
    process = local.ProcessWithLogging(
        cast(subprocess.Popen[str], process_mock),
        cast(tuple[TextIO, TextIO], (Mock(), Mock())),
        cast(tuple[threading.Thread, threading.Thread], (Mock(), Mock())),
    )

    process.kill()
    process.terminate()

    process_mock.kill.assert_called_once_with()
    process_mock.terminate.assert_called_once_with()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_completed_process_ignores_repeated_group_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = local.submit(str(tmp_path), [sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    process.cleanup_logging()
    group_signals: list[tuple[int, int]] = []

    def record_group_signal(process_group: int, sig: int) -> None:
        group_signals.append((process_group, sig))

    monkeypatch.setattr(local.os, "killpg", record_group_signal)

    process.kill()
    process.terminate()
    process.kill()

    assert group_signals == []


def test_scheduler_cancellation_reaps_process_tree(
    process_tree: tuple[local.ProcessWithLogging, int, Path],
) -> None:
    process, grandchild_pid, _log_dir = process_tree
    scheduler = JobScheduler("local", LocalJobConfig())

    try:
        cancelled = asyncio.run(scheduler.cancel_job_async(process))
    finally:
        scheduler.shutdown()

    assert cancelled is True
    assert process.returncode is not None
    assert all(not thread.is_alive() for thread in process.log_threads)
    assert all(file_handle.closed for file_handle in process.log_files)
    _wait_for_process_exit(grandchild_pid)


def test_timeout_kills_descendant_processes(
    process_tree: tuple[local.ProcessWithLogging, int, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process, grandchild_pid, log_dir = process_tree
    assert os.getpgid(process.pid) == process.pid
    monkeypatch.setattr(local, "time", _ExpiredClock())

    local.monitor(
        process,
        str(log_dir),
        poll_interval=0,
        timeout="00:00:01",
    )

    process.wait(timeout=5)
    _wait_for_process_exit(grandchild_pid)


def test_terminate_cancellation_kills_descendant_processes(
    process_tree: tuple[local.ProcessWithLogging, int, Path],
) -> None:
    process, grandchild_pid, _log_dir = process_tree
    assert os.getpgid(process.pid) == process.pid

    process.terminate()

    process.wait(timeout=5)
    process.cleanup_logging()
    _wait_for_process_exit(grandchild_pid)
