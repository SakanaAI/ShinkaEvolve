"""Exec one command that the Linux kernel kills when its parent exits."""

import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1


def main(parent_pid: int, command: list[str]) -> int:
    if not sys.platform.startswith("linux") or not command:
        return 127
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "Could not set parent-death signal")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(127)
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2:]))
