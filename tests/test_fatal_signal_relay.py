import os
import signal
import time
from typing import Sequence

import pytest

from shinka.launch import fatal_signals
from signal_stop_test_utils import FATAL_SIGNALS

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")


def _relay(
    monkeypatch: pytest.MonkeyPatch,
    reports: Sequence[int],
    handler_runs_on_relay: Sequence[bool],
) -> list[int]:
    """Signals the relay sends for `reports`; each may let the handler run or not."""
    runs = iter(handler_runs_on_relay)
    relayed: list[int] = []

    def relay(thread_id: int, signum: int) -> None:
        relayed.append(signum)
        if next(runs):
            fatal_signals._handler_runs[signum] += 1

    monkeypatch.setattr(signal, "pthread_kill", relay)
    monkeypatch.setattr(fatal_signals, "_RELAY_SECONDS", 0.0)
    # As a process has them that has handled signals before.
    monkeypatch.setattr(
        fatal_signals, "_handler_runs", {signum: 3 for signum in FATAL_SIGNALS}
    )
    read_fd, write_fd = os.pipe()
    previous = signal.signal(signal.SIGTERM, fatal_signals._on_fatal_signal)
    try:
        os.write(write_fd, bytes(reports))
        os.close(write_fd)
        fatal_signals._relay_fatal_signals(read_fd, 1234)
    finally:
        signal.signal(signal.SIGTERM, previous)
        os.close(read_fd)
    return relayed


def test_relay_does_not_answer_its_own_signal_with_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second report is that of the relay's own signal. The handler stays
    # installed while a run stops, so nothing else would end the exchange.
    relayed = _relay(monkeypatch, [signal.SIGTERM, signal.SIGTERM], [True])

    assert relayed == [signal.SIGTERM]


def test_relay_notes_when_a_signal_arrived_that_the_handler_has_not_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fatal_signals, "_unhandled_since", {})
    before = time.monotonic()

    _relay(monkeypatch, [signal.SIGTERM, signal.SIGTERM], [False, False])

    assert before <= fatal_signals._unhandled_since[signal.SIGTERM] <= time.monotonic()


def test_relay_repeats_a_signal_until_the_handler_has_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A main thread busy in C code takes the signal but runs no handler yet.
    reports = [signal.SIGTERM] * 4
    relayed = _relay(monkeypatch, reports, [False, False, True])

    assert relayed == [signal.SIGTERM] * 3


def test_relay_sends_a_later_signal_to_the_main_thread_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First signal, the relay's own, then the one of somebody who will not wait.
    reports = [signal.SIGTERM] * 4
    relayed = _relay(monkeypatch, reports, [True, True])

    assert relayed == [signal.SIGTERM] * 2
