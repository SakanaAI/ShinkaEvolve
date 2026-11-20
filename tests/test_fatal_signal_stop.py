import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Iterator, Sequence
from unittest.mock import MagicMock

import pytest

from job_lifecycle_test_utils import (
    FakeClock,
    is_alive,
    kill_leftovers,
    wait_until,
    worker_pid,
    write_evaluator_with_worker,
)
from shinka.launch import ProcessWithLogging, fatal_signals, local
from signal_stop_test_utils import FATAL_SIGNALS, raise_hooked, recorded_deaths

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")


@pytest.fixture
def deaths(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    with recorded_deaths(monkeypatch) as died_of:
        yield died_of


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_first_fatal_signal_asks_a_run_to_stop(deaths: list[int], signum: int) -> None:
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    raise_hooked(signum)

    assert asked == [signum]
    assert deaths == []

    stop.finish()

    assert deaths == [signum]
    assert signal.getsignal(signum) is signal.SIG_DFL


def test_signals_right_after_the_first_are_the_same_request(deaths: list[int]) -> None:
    # A closing terminal sends SIGHUP twice; a service manager adds SIGTERM.
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    raise_hooked(signal.SIGHUP)
    raise_hooked(signal.SIGHUP)
    raise_hooked(signal.SIGTERM)

    assert asked == [signal.SIGHUP]
    assert deaths == []

    stop.finish()

    assert deaths == [signal.SIGHUP]


def test_later_signal_ends_the_process_at_once_with_its_evaluators(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    clock.install(monkeypatch, fatal_signals)
    evaluator = ProcessWithLogging(
        MagicMock(spec=subprocess.Popen, pid=4321, returncode=None),
        (MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
        process_group_id=4321,
    )
    evaluator.process.wait.side_effect = AssertionError("a signal handler must not wait")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(local, "_LIVE_PROCESSES", {evaluator})
    monkeypatch.setattr(os, "killpg", lambda group, signum: killed.append((group, signum)))
    asked: list[int] = []
    fatal_signals.stop_on_fatal_signals(asked.append)

    raise_hooked(signal.SIGTERM)
    clock.sleep(fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)

    assert asked == [signal.SIGTERM]
    assert killed == [(4321, signal.SIGKILL)]
    assert deaths == [signal.SIGTERM]


def test_run_that_no_signal_stopped_finishes_and_returns(deaths: list[int]) -> None:
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    stop.finish()
    stop.finish()

    assert deaths == []

    # No run is left to stop: the process ends at once, as without one.
    raise_hooked(signal.SIGTERM)

    assert asked == []
    assert deaths == [signal.SIGTERM]


def test_process_dies_once_every_run_has_stopped(deaths: list[int]) -> None:
    asked: list[str] = []
    first = fatal_signals.stop_on_fatal_signals(lambda received: asked.append("first"))
    second = fatal_signals.stop_on_fatal_signals(lambda received: asked.append("second"))

    raise_hooked(signal.SIGTERM)
    # Started while the others are stopping: asked at once.
    third = fatal_signals.stop_on_fatal_signals(lambda received: asked.append("third"))
    first.finish()
    third.finish()

    assert asked == ["first", "second", "third"]
    assert deaths == []

    second.finish()

    assert deaths == [signal.SIGTERM]


def test_request_that_cannot_be_passed_on_ends_the_process(deaths: list[int]) -> None:
    def closed_loop(received: signal.Signals) -> None:
        raise RuntimeError("Event loop is closed")

    fatal_signals.stop_on_fatal_signals(closed_loop)

    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]


def test_application_that_handles_the_signals_stops_its_runs_itself(
    deaths: list[int],
) -> None:
    def application_handler(signum, frame) -> None:
        pass

    for signum in FATAL_SIGNALS:
        signal.signal(signum, application_handler)
    stop = fatal_signals.stop_on_fatal_signals(lambda received: None)

    assert fatal_signals._stops == []
    for signum in FATAL_SIGNALS:
        assert signal.getsignal(signum) is application_handler

    stop.finish()

    assert deaths == []


def test_run_off_the_main_thread_leaves_the_signals_alone(deaths: list[int]) -> None:
    def run() -> None:
        stop = fatal_signals.stop_on_fatal_signals(lambda received: None)
        stop.finish()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert fatal_signals._stops == []
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert deaths == []


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


@contextmanager
def _run(tmp_path: Path, *, start: str = "", cleanup: str = "") -> Iterator[subprocess.Popen]:
    """A process whose run waits for a fatal signal, cleans up and finishes.

    It prints a line once it waits, and leaves the files "stopping" and
    "cleaned" behind as it gets that far.
    """
    script = tmp_path / "run.py"
    script.write_text(
        dedent("""
            import os
            import signal
            import sys
            import threading
            import time
            from pathlib import Path
            from shinka.launch import fatal_signals, local

            here = Path(sys.argv[1])
            asked, ask = os.pipe()
            stop = fatal_signals.stop_on_fatal_signals(lambda received: os.write(ask, b"!"))
            """)
        + dedent(start)
        + dedent("""
            print("waiting", flush=True)
            os.read(asked, 1)
            (here / "stopping").touch()
            """)
        + dedent(cleanup)
        + dedent("""
            (here / "cleaned").touch()
            stop.finish()
            """)
    )
    run = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    try:
        assert run.stdout is not None
        assert run.stdout.readline() == "waiting\n"
        yield run
    finally:
        run.kill()
        run.wait()


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_signalled_run_cleans_up_and_dies_of_the_signal(
    tmp_path: Path, signum: int
) -> None:
    with _run(tmp_path) as run:
        run.send_signal(signum)

        assert run.wait(timeout=10.0) == -signum
        assert (tmp_path / "cleaned").exists()


def test_closing_terminal_does_not_cut_the_cleanup_short(tmp_path: Path) -> None:
    with _run(tmp_path, cleanup="time.sleep(0.3)") as run:
        run.send_signal(signal.SIGHUP)
        run.send_signal(signal.SIGHUP)

        assert run.wait(timeout=10.0) == -signal.SIGHUP
        assert (tmp_path / "cleaned").exists()


def test_cleanup_is_not_ended_by_the_relays_own_signals(tmp_path: Path) -> None:
    # Another thread takes the signal, so the relay passes it on. Its signal
    # must not count as a second one once the cleanup has taken a while.
    take_the_signal = """
        threading.Timer(
            0.2, lambda: signal.pthread_kill(threading.get_ident(), signal.SIGTERM)
        ).start()
        """
    with _run(tmp_path, start=take_the_signal, cleanup="time.sleep(2.0)") as run:
        assert run.wait(timeout=10.0) == -signal.SIGTERM
        assert (tmp_path / "cleaned").exists()


def test_second_signal_after_a_while_ends_the_run_with_its_evaluators(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    evaluator = write_evaluator_with_worker(tmp_path)
    launch_an_evaluator = f"""
        process = local.submit(
            {str(results_dir)!r},
            [sys.executable, {str(evaluator)!r}, "--results_dir", {str(results_dir)!r}],
        )
        """
    script = _run(tmp_path, start=launch_an_evaluator, cleanup="time.sleep(30)")
    evaluator_pid = worker = 0
    try:
        with script as run:
            worker = worker_pid(results_dir)
            evaluator_pid = os.getpgid(worker)  # The evaluator leads its group.
            run.send_signal(signal.SIGTERM)
            assert wait_until((tmp_path / "stopping").exists)
            time.sleep(fatal_signals._SAME_REQUEST_SECONDS + 0.2)
            assert is_alive(evaluator_pid)

            run.send_signal(signal.SIGTERM)

            assert run.wait(timeout=10.0) == -signal.SIGTERM
            assert not (tmp_path / "cleaned").exists()
            assert wait_until(lambda: not is_alive(evaluator_pid))
            assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(*(pid for pid in (evaluator_pid, worker) if pid))
