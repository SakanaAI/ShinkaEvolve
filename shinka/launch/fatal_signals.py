"""What SIGTERM and SIGHUP do to a process that runs evaluations.

Neither signal runs exit handlers, so evaluators in their own process group,
or on a cluster, would outlive the process. A run that has asked for it is
stopped by the first signal the way Ctrl-C stops it, and dies of the signal
once it has cleaned up. Without such a run, or when a later signal shows that
nobody will wait, local evaluators are killed and the process dies at once.
"""

import logging
import os
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .local import kill_live_evaluators_without_reaping

logger = logging.getLogger(__name__)

# Python runs a signal handler in the main thread, and only once that thread is
# back in the interpreter. The kernel may give a signal to any thread: of two
# sent back to back, Linux gives the second to another thread (numpy's BLAS
# pool has some), which then takes both, and a main thread blocked in a sleep
# or a wait is not interrupted. So Python reports every signal it receives to
# a pipe, and a relay thread sends the fatal ones to the main thread itself.
# The pipe's read end and write end, once the relay runs.
_signal_reports: Optional[Tuple[int, int]] = None
# The relay repeats its signal at this pace until the handler has run. That
# also reaches a main thread which was about to block.
_RELAY_SECONDS = 0.05
# How often the handler has run for each signal. The relay's own signal is
# reported like any other; this count tells the relay that the handler has run
# since, so that it does not answer its own signal with another one.
_handler_runs: Dict[int, int] = {}

# Runs that the first fatal signal asks to stop.
_stops: "List[SignalStop]" = []
# The signal that asked them, and when. They die of it once all have stopped.
_asked_to_stop: Optional[Tuple[int, float]] = None
# A closing terminal sends SIGHUP twice within milliseconds, a service manager
# may send SIGHUP right after SIGTERM, and the relay may repeat a signal once
# more than needed: signals this soon after the first are the same request.
# A later one comes from somebody who will not wait for the run to stop.
_SAME_REQUEST_SECONDS = 1.0
_warned_that_nothing_is_hooked = False


def _forget_the_parent_after_fork() -> None:
    """Drop what a forked child inherits but its parent owns."""
    global _signal_reports, _asked_to_stop
    _stops.clear()
    _asked_to_stop = None
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

    Only default dispositions are replaced: a handler the application
    installed, or ``nohup``, stays in charge. Python installs signal handlers
    from the main thread only: elsewhere this warns, once, when a signal still
    has its default disposition.
    """
    if os.name != "posix":
        return
    if threading.current_thread() is not threading.main_thread():
        _warn_once_that_nothing_is_hooked()
        return
    for signum in _fatal_signals():
        if signal.getsignal(signum) is signal.SIG_DFL:
            signal.signal(signum, _on_fatal_signal)
    if any(_is_hooked(signum) for signum in _fatal_signals()):
        _relay_fatal_signals_to_the_main_thread()


def _warn_once_that_nothing_is_hooked() -> None:
    global _warned_that_nothing_is_hooked
    unhooked = [
        signal.Signals(signum).name
        for signum in _fatal_signals()
        if signal.getsignal(signum) is signal.SIG_DFL
    ]
    if _warned_that_nothing_is_hooked or not unhooked:
        return
    _warned_that_nothing_is_hooked = True
    logger.warning(
        "%s would end this process and leave its evaluators running: Python "
        "installs signal handlers from the main thread only. Call shinka."
        "launch.fatal_signals.kill_evaluators_on_fatal_signals() from the main "
        "thread.",
        " and ".join(unhooked),
    )


class SignalStop:
    """A run that the first fatal signal stops, and that then dies of it."""

    def __init__(self, ask_to_stop: Callable[[signal.Signals], None]) -> None:
        self._ask_to_stop = ask_to_stop

    def finish(self) -> None:
        """Say that the run has ended and cleaned up.

        Does not return if a signal asked the run to stop: the process then
        dies of that signal, so that its exit status shows what ended it.
        """
        if self not in _stops:
            return
        _stops.remove(self)
        if _asked_to_stop is None or _stops:
            return
        if isinstance(sys.exc_info()[1], Exception):
            logger.error("The run failed while a signal was stopping it", exc_info=True)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # A terminal that hung up takes no more output.
                pass
        _die_with_evaluators(_asked_to_stop[0])


def stop_on_fatal_signals(ask_to_stop: Callable[[signal.Signals], None]) -> SignalStop:
    """Have the first SIGTERM or SIGHUP ask a run to stop, as Ctrl-C would.

    ``ask_to_stop`` is called from the signal handler, between any two
    instructions of the main thread: it should only pass the request on, such
    as with ``loop.call_soon_threadsafe``. Signals within a second of the
    first are the same request. A later one kills the local evaluators and
    ends the process at once, whatever the run has cleaned up by then.

    Call ``finish`` on the result when the run has ended and cleaned up.
    Nothing is hooked off the main thread, or for a signal the application
    handles itself; the result is then inert.
    """
    stop = SignalStop(ask_to_stop)
    kill_evaluators_on_fatal_signals()
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return stop
    if not any(_is_hooked(signum) for signum in _fatal_signals()):
        return stop
    _stops.append(stop)
    if _asked_to_stop is not None:
        # Started while other runs are stopping for a signal.
        ask_to_stop(signal.Signals(_asked_to_stop[0]))
    return stop


def _fatal_signals() -> Tuple[int, int]:
    return signal.SIGTERM, signal.SIGHUP


def _is_hooked(signum: int) -> bool:
    return signal.getsignal(signum) is _on_fatal_signal


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
    handled = dict(_handler_runs)
    while True:
        report = os.read(read_fd, 1)
        if not report:
            return
        signum = report[0]
        if signum not in _fatal_signals() or not _is_hooked(signum):
            continue
        runs = _handler_runs.get(signum, 0)
        if runs != handled.get(signum, 0):
            # The handler has run since the last report, for this one too.
            handled[signum] = runs
            continue
        signal.pthread_kill(main_thread_id, signum)
        time.sleep(_RELAY_SECONDS)


def _on_fatal_signal(signum, frame) -> None:
    _handler_runs[signum] = _handler_runs.get(signum, 0) + 1
    if not _runs_are_stopping_for(signum):
        _die_with_evaluators(signum)


def _runs_are_stopping_for(signum: int) -> bool:
    """Ask the runs to stop on the first signal; say if that covers this one."""
    global _asked_to_stop
    if not _stops:
        return False
    now = time.monotonic()
    if _asked_to_stop is not None:
        return now - _asked_to_stop[1] < _SAME_REQUEST_SECONDS
    _asked_to_stop = (signum, now)
    try:
        for stop in list(_stops):
            stop._ask_to_stop(signal.Signals(signum))
    except Exception:
        return False  # Nobody would stop that run.
    return True


def _die_with_evaluators(signum: int) -> None:
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
