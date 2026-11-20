"""What SIGTERM and SIGHUP do to a process that runs evaluations."""

import os
import signal
import threading
import time
from typing import Optional, Tuple

from .local import kill_live_evaluators_without_reaping

# Python runs a signal handler in the main thread, and only once that thread is
# back in the interpreter. The kernel may give a signal to any thread: of two
# sent back to back, Linux gives the second to another thread (numpy's BLAS
# pool has some), which then takes both, and a main thread blocked in a sleep
# or a wait is not interrupted. So Python reports every signal it receives to
# a pipe, and a relay thread sends the fatal ones to the main thread itself.
# The pipe's read end and write end, once the relay runs.
_signal_reports: Optional[Tuple[int, int]] = None
# The relay's own signal is reported too, so it repeats at this pace until the
# handler has started. That also reaches a main thread which was about to block.
_RELAY_SECONDS = 0.05


def _forget_the_parent_after_fork() -> None:
    """Drop the signal reports a forked child inherits from its parent."""
    global _signal_reports
    if _signal_reports is None:
        return
    # The parent's relay would take this process's signals for its own.
    read_fd, write_fd = _signal_reports
    _signal_reports = None
    reported_to = signal.set_wakeup_fd(-1)
    if reported_to != write_fd:
        signal.set_wakeup_fd(reported_to)
    os.close(read_fd)
    os.close(write_fd)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_the_parent_after_fork)


def kill_evaluators_on_fatal_signals() -> None:
    """Take running evaluators along when SIGTERM or SIGHUP ends this process.

    Neither signal runs exit handlers, and evaluators in their own process
    group would otherwise outlive the run. Only default dispositions are
    replaced: a handler the application installed, or ``nohup``, stays in charge.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return
    for signum in _fatal_signals():
        if signal.getsignal(signum) is signal.SIG_DFL:
            signal.signal(signum, _die_with_evaluators)
    if any(_is_hooked(signum) for signum in _fatal_signals()):
        _relay_fatal_signals_to_the_main_thread()


def _fatal_signals() -> Tuple[int, int]:
    return signal.SIGTERM, signal.SIGHUP


def _is_hooked(signum: int) -> bool:
    return signal.getsignal(signum) is _die_with_evaluators


def _relay_fatal_signals_to_the_main_thread() -> None:
    """Have signals reported to the relay, started on the first call.

    Call from the main thread.
    """
    global _signal_reports
    if _signal_reports is None:
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        threading.Thread(
            target=_relay_fatal_signals,
            args=(read_fd, threading.main_thread().ident),
            name="fatal-signal-relay",
            daemon=True,
        ).start()
        _signal_reports = (read_fd, write_fd)
    write_fd = _signal_reports[1]
    reported_to = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    if reported_to not in (-1, write_fd):
        # An event loop has signals reported to itself, and wakes up for them.
        signal.set_wakeup_fd(reported_to)


def _relay_fatal_signals(read_fd: int, main_thread_id: int) -> None:
    while True:
        report = os.read(read_fd, 1)
        if not report:
            return
        signum = report[0]
        if signum in _fatal_signals() and _is_hooked(signum):
            signal.pthread_kill(main_thread_id, signum)
            time.sleep(_RELAY_SECONDS)


def _die_with_evaluators(signum, frame) -> None:
    # A closing terminal can send a second signal while this one is handled;
    # it must not start the handler again inside itself.
    for fatal in _fatal_signals():
        signal.signal(fatal, signal.SIG_IGN)
    try:
        kill_live_evaluators_without_reaping()
    finally:
        # Then end the way the signal would have ended this process.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
