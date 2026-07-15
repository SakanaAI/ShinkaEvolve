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

_TERM_RESISTANT_CODE = """
from pathlib import Path
import signal
import sys

signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).touch()
signal.pause()
"""


_LEADER_EXITS_CODE = """
import subprocess
from pathlib import Path
import sys
import time

ready_path = Path(sys.argv[1])
subprocess.Popen(
    [sys.executable, "-c", sys.argv[2], sys.argv[1]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
deadline = time.monotonic() + 5
while not ready_path.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("grandchild did not become ready")
    time.sleep(0.01)
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
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
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
def test_scheduler_raw_status_api_accepts_process_wrapper(tmp_path: Path) -> None:
    process = local.submit(
        str(tmp_path),
        [sys.executable, "-c", "import signal; signal.pause()"],
    )
    scheduler = JobScheduler("local", LocalJobConfig())

    try:
        assert asyncio.run(scheduler.check_job_id_status_async(process)) is True

        process.kill()
        process.wait(timeout=5)

        assert asyncio.run(scheduler.check_job_id_status_async(process)) is False
    finally:
        process.cleanup_logging()
        scheduler.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_completed_process_relinquishes_group_during_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = local.submit(str(tmp_path), [sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    process.cleanup_logging()
    group_signals: list[tuple[int, int]] = []

    def record_group_signal(process_group: int, sig: int) -> None:
        group_signals.append((process_group, sig))
        raise ProcessLookupError

    monkeypatch.setattr(local.os, "killpg", record_group_signal)

    process.kill()
    process.terminate()
    process.kill()

    assert group_signals == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_terminate_can_escalate_when_evaluator_ignores_signal(tmp_path: Path) -> None:
    ready_path = tmp_path / "ready"
    process = local.submit(
        str(tmp_path / "logs"),
        [sys.executable, "-c", _TERM_RESISTANT_CODE, str(ready_path)],
    )
    deadline = time.monotonic() + 5
    while not ready_path.exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            raise AssertionError("evaluator did not signal readiness")
        time.sleep(0.01)

    try:
        process.terminate()
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
        assert process._process_group == process.pid

        process.kill()
        process.wait(timeout=5)
        assert process._process_group is None
    finally:
        process.cleanup_logging()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_group_signal_reaps_descendant_after_leader_exits(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    ready_path = tmp_path / "grandchild.ready"
    process = local.submit(
        str(log_dir),
        [
            sys.executable,
            "-c",
            _LEADER_EXITS_CODE,
            str(ready_path),
            _GRANDCHILD_CODE,
        ],
    )
    grandchild_pid = _wait_for_ready_pid(ready_path, process)

    try:
        process.wait(timeout=5)
        assert process.returncode == 0
        assert process.process.poll() is None
        assert os.getpgid(process.pid) == process.pid
        assert os.getpgid(grandchild_pid) == process.pid

        process.kill()

        _wait_for_process_exit(grandchild_pid)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.cleanup_logging()


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


def test_send_signal_sigterm_reaches_process_group(
    process_tree: tuple[local.ProcessWithLogging, int, Path],
) -> None:
    process, grandchild_pid, _log_dir = process_tree

    process.send_signal(signal.SIGTERM)

    process.wait(timeout=5)
    process.cleanup_logging()
    _wait_for_process_exit(grandchild_pid)


def test_send_signal_sigkill_reaches_process_group(
    process_tree: tuple[local.ProcessWithLogging, int, Path],
) -> None:
    process, grandchild_pid, _log_dir = process_tree

    process.send_signal(signal.SIGKILL)

    process.wait(timeout=5)
    assert process._process_group is None
    process.cleanup_logging()
    _wait_for_process_exit(grandchild_pid)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_abnormal_supervisor_exit_relinquishes_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_path = tmp_path / "evaluator.ready"
    process = local.submit(
        str(tmp_path / "logs"),
        [sys.executable, "-c", _GRANDCHILD_CODE, str(ready_path)],
    )
    evaluator_pid = _wait_for_ready_pid(ready_path, process)
    group_signals: list[tuple[int, int]] = []
    killpg = os.killpg

    try:
        process.process.kill()
        process.process.wait(timeout=5)

        assert process.poll() == -signal.SIGKILL
        assert process._process_group is None
        _wait_for_process_exit(evaluator_pid)

        monkeypatch.setattr(
            local.os,
            "killpg",
            lambda process_group, sig: group_signals.append((process_group, sig)),
        )
        process.cleanup_logging()
        assert group_signals == []
    finally:
        try:
            killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.cleanup_logging()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file descriptors are required")
def test_completion_pipe_avoids_closed_standard_fds(tmp_path: Path) -> None:
    result_path = tmp_path / "returncode"
    harness = f"""
import os
from pathlib import Path
import sys
from shinka.launch import local

os.close(0)
os.close(1)
process = local.submit(
    {str(tmp_path / "logs")!r},
    [sys.executable, "-c", "raise SystemExit(7)"],
)
completion_fd = process._completion_fd
completion_fd_inheritable = os.get_inheritable(completion_fd)
returncode = process.wait(timeout=5)
process.cleanup_logging()
Path({str(result_path)!r}).write_text(
    f"{{completion_fd}},{{completion_fd_inheritable}},{{returncode}}",
    encoding="ascii",
)
"""

    completed = subprocess.run([sys.executable, "-c", harness], timeout=10)

    assert completed.returncode == 0
    completion_fd, inheritable, returncode = result_path.read_text(
        encoding="ascii"
    ).split(",")
    assert int(completion_fd) >= 3
    assert inheritable == "False"
    assert returncode == "7"
