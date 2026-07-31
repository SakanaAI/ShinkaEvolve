"""Trusted local-evaluation supervisor that retains durable ownership state."""

import ctypes
import os
import signal
import subprocess
import sys
import time

import psutil

LOCAL_PROCESS_TOKEN_ENV = "SHINKA_LOCAL_JOB_TOKEN"
TERMINATION_GRACE_SECONDS = 1.0
GROUP_POLL_INTERVAL_SECONDS = 0.1
GROUP_QUIESCENCE_SECONDS = 0.2
PR_SET_PDEATHSIG = 1


def _report_launch_status(status_fd: int, message: str) -> bool:
    try:
        os.write(status_fd, message.encode("utf-8", errors="replace"))
        return True
    except OSError:
        return False
    finally:
        os.close(status_fd)


def _signal_group_members(signum: int) -> bool:
    supervisor_pid = os.getpid()
    process_group_id = os.getpgrp()
    all_signaled = True
    for process in psutil.process_iter(["pid"]):
        if process.pid == supervisor_pid:
            continue
        try:
            if os.getpgid(process.pid) == process_group_id:
                process.send_signal(signum)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError, OSError):
            all_signaled = False
    return all_signaled


def _group_has_members() -> bool | None:
    supervisor_pid = os.getpid()
    process_group_id = os.getpgrp()
    for process in psutil.process_iter(["pid"]):
        if process.pid == supervisor_pid:
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


def _terminate_group(signum: int) -> bool:
    _signal_group_members(signum)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        group_has_members = _group_has_members()
        if group_has_members is False:
            return True
        time.sleep(0.01)
    _signal_group_members(signal.SIGKILL)
    time.sleep(0.01)
    return _group_has_members() is False


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
        evaluator = subprocess.Popen(
            command,
            env=evaluator_env,
            preexec_fn=lambda: _parent_death_kills_evaluator(supervisor_pid),
        )
    except OSError as error:
        _report_launch_status(status_fd, f"ERROR:{error.errno or 1}:{error}")
        return 127
    _report_launch_status(status_fd, "READY")

    evaluator_returncode = None
    group_empty_since = None
    while True:
        if received_signal is not None:
            if _terminate_group(received_signal):
                evaluator.poll()
                return 128 + received_signal
            time.sleep(0.1)
            continue
        if evaluator_returncode is None:
            evaluator_returncode = evaluator.poll()
        if evaluator_returncode is not None:
            if _group_has_members() is False:
                if group_empty_since is None:
                    group_empty_since = time.monotonic()
                elif time.monotonic() - group_empty_since >= GROUP_QUIESCENCE_SECONDS:
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
