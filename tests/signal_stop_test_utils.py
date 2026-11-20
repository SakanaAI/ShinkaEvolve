"""Helpers shared by the tests of what fatal signals do to a run."""

import os
import signal
from contextlib import contextmanager
from typing import Iterator

import pytest

from shinka.launch import fatal_signals, local

FATAL_SIGNALS = [signal.SIGTERM, signal.SIGHUP]


@contextmanager
def recorded_deaths(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Fatal signals as a fresh process has them; yields what would have ended it."""
    died_of: list[int] = []
    previous = {signum: signal.getsignal(signum) for signum in FATAL_SIGNALS}
    monkeypatch.setattr(fatal_signals, "_stops", [])
    monkeypatch.setattr(fatal_signals, "_asked_to_stop", None)
    monkeypatch.setattr(local, "_LIVE_PROCESSES", set())
    # Without the relay, only the signals a test raises reach the handler.
    monkeypatch.setattr(
        fatal_signals, "_relay_fatal_signals_to_the_main_thread", lambda: None
    )
    reported_to = signal.set_wakeup_fd(-1)
    monkeypatch.setattr(os, "kill", lambda pid, signum: died_of.append(signum))
    for signum in FATAL_SIGNALS:
        signal.signal(signum, signal.SIG_DFL)
    try:
        yield died_of
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.set_wakeup_fd(reported_to)


def raise_hooked(signum: int) -> None:
    """Send this process a signal whose handler is the hook."""
    # Unhooked, the signal would end the test run.
    assert fatal_signals._is_hooked(signum)
    signal.raise_signal(signum)
