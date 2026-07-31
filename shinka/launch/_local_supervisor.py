"""Trusted local-evaluation supervisor that retains durable ownership state."""

import ctypes
import os
import select
import signal
import subprocess
import sys
import time

import psutil

LOCAL_PROCESS_TOKEN_ENV = "SHINKA_LOCAL_JOB_TOKEN"
TERMINATION_GRACE_SECONDS = 1.0
GROUP_POLL_INTERVAL_SECONDS = 0.1
GROUP_QUIESCENCE_SECONDS = 0.2
GROUP_GUARD_START_TIMEOUT_SECONDS = 1.0
GROUP_GUARD_STOP_TIMEOUT_SECONDS = 1.0
PR_SET_PDEATHSIG = 1


def _report_launch_status(status_fd: int, message: str) -> bool:
    try:
        os.write(status_fd, message.encode("utf-8", errors="replace"))
        return True
    except OSError:
        return False
    finally:
        os.close(status_fd)


def _signal_group_members(signum: int, ignored_pid: int | None = None) -> bool:
    supervisor_pid = os.getpid()
    process_group_id = os.getpgrp()
    all_signaled = True
    for process in psutil.process_iter(["pid"]):
        if process.pid in (supervisor_pid, ignored_pid):
            continue
        try:
            if os.getpgid(process.pid) == process_group_id:
                process.send_signal(signum)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError, OSError):
            all_signaled = False
    return all_signaled


def _group_has_members(ignored_pid: int | None = None) -> bool | None:
    supervisor_pid = os.getpid()
    process_group_id = os.getpgrp()
    for process in psutil.process_iter(["pid"]):
        if process.pid in (supervisor_pid, ignored_pid):
            continue
        try:
            if (
                os.getpgid(process.pid) == process_group_id
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                return True
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError, OSError):
            return None
    return False


def _terminate_group(signum: int, ignored_pid: int | None = None) -> bool:
    _signal_group_members(signum, ignored_pid)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        group_has_members = _group_has_members(ignored_pid)
        if group_has_members is False:
            return True
        time.sleep(0.01)
    _signal_group_members(signal.SIGKILL, ignored_pid)
    time.sleep(0.01)
    return _group_has_members(ignored_pid) is False


def _start_group_guard(status_fd: int) -> tuple[int, int]:
    """Keep a token-bearing member that kills the group on supervisor death."""
    read_fd, write_fd = os.pipe()
    ready_read_fd, ready_write_fd = os.pipe()
    try:
        guard_pid = os.fork()
    except OSError:
        for file_descriptor in (
            read_fd,
            write_fd,
            ready_read_fd,
            ready_write_fd,
        ):
            os.close(file_descriptor)
        raise
    if guard_pid == 0:
        os.close(write_fd)
        os.close(ready_read_fd)
        os.close(status_fd)
        if os.getpgrp() != os.getppid():
            os._exit(127)
        os.write(ready_write_fd, b"A")
        os.close(ready_write_fd)
        try:
            clean_shutdown = os.read(read_fd, 1) == b"C"
        except OSError:
            clean_shutdown = False
        finally:
            os.close(read_fd)
        if not clean_shutdown:
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except ProcessLookupError:
                pass
        os._exit(0)
    os.close(read_fd)
    os.close(ready_write_fd)
    readable, _, _ = select.select(
        [ready_read_fd], [], [], GROUP_GUARD_START_TIMEOUT_SECONDS
    )
    armed = bool(readable) and os.read(ready_read_fd, 1) == b"A"
    os.close(ready_read_fd)
    if not armed:
        _stop_group_guard(guard_pid, write_fd)
        raise RuntimeError("Local process-group guard did not arm")
    return guard_pid, write_fd


def _stop_group_guard(guard_pid: int, write_fd: int) -> bool:
    clean_shutdown_sent = False
    try:
        clean_shutdown_sent = os.write(write_fd, b"C") == 1
    except OSError:
        pass
    finally:
        os.close(write_fd)
    deadline = time.monotonic() + GROUP_GUARD_STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            waited_pid, status = os.waitpid(guard_pid, os.WNOHANG)
        except ChildProcessError:
            return False
        if waited_pid == guard_pid:
            return clean_shutdown_sent and os.waitstatus_to_exitcode(status) == 0
        time.sleep(0.01)

    try:
        os.kill(guard_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    kill_deadline = time.monotonic() + GROUP_GUARD_STOP_TIMEOUT_SECONDS
    while time.monotonic() < kill_deadline:
        try:
            waited_pid, _ = os.waitpid(guard_pid, os.WNOHANG)
        except ChildProcessError:
            return False
        if waited_pid == guard_pid:
            return False
        time.sleep(0.01)
    return False


def _group_guard_exited(guard_pid: int) -> bool:
    try:
        waited_pid, _ = os.waitpid(guard_pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return waited_pid == guard_pid


def _parent_death_kills_evaluator(supervisor_pid: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "Could not set evaluator parent-death signal")
    if os.getppid() != supervisor_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def main(status_fd: int, command: list[str]) -> int:
    if not command:
        _report_launch_status(status_fd, "ERROR:missing evaluator command")
        return 127
    evaluator_env = os.environ.copy()
    evaluator_env.pop(LOCAL_PROCESS_TOKEN_ENV, None)
    received_signal = None

    def request_termination(signum, _frame):
        nonlocal received_signal
        received_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_termination)

    supervisor_pid = os.getpid()
    try:
        guard_pid, guard_write_fd = _start_group_guard(status_fd)
    except (OSError, RuntimeError) as error:
        error_number = error.errno if isinstance(error, OSError) else 1
        _report_launch_status(status_fd, f"ERROR:{error_number or 1}:{error}")
        return 127
    if _group_guard_exited(guard_pid):
        os.close(guard_write_fd)
        _report_launch_status(status_fd, "ERROR:1:Local process-group guard exited")
        return 127
    try:
        evaluator = subprocess.Popen(
            command,
            env=evaluator_env,
            preexec_fn=lambda: _parent_death_kills_evaluator(supervisor_pid),
        )
    except OSError as error:
        _stop_group_guard(guard_pid, guard_write_fd)
        _report_launch_status(status_fd, f"ERROR:{error.errno or 1}:{error}")
        return 127
    _report_launch_status(status_fd, "READY")

    evaluator_returncode = None
    group_empty_since = None
    while True:
        if _group_guard_exited(guard_pid):
            os.close(guard_write_fd)
            _terminate_group(signal.SIGKILL)
            return 127
        if received_signal is not None:
            if _terminate_group(received_signal, guard_pid):
                evaluator.poll()
                if not _stop_group_guard(guard_pid, guard_write_fd):
                    return 127
                return 128 + received_signal
            time.sleep(0.1)
            continue
        if evaluator_returncode is None:
            evaluator_returncode = evaluator.poll()
        if evaluator_returncode is not None:
            if _group_has_members(guard_pid) is False:
                if group_empty_since is None:
                    group_empty_since = time.monotonic()
                elif time.monotonic() - group_empty_since >= GROUP_QUIESCENCE_SECONDS:
                    if not _stop_group_guard(guard_pid, guard_write_fd):
                        return 127
                    return evaluator_returncode
            else:
                group_empty_since = None
            time.sleep(GROUP_POLL_INTERVAL_SECONDS)
            continue
        time.sleep(0.01)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(127)
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2:]))
