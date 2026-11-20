import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from job_lifecycle_test_utils import (
    FakeClock,
)
from shinka.launch import (
    JobScheduler,
    LocalJobConfig,
    ProcessWithLogging,
    SlurmCondaJobConfig,
)
from shinka.launch import fatal_signals, local
from shinka.launch import scheduler as scheduler_module
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


@pytest.mark.parametrize("first", FATAL_SIGNALS)
def test_later_hangup_does_not_end_a_stopping_run(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch, first: int
) -> None:
    # A closing terminal hangs up a second time when its shell exits, however
    # long that takes: with an exit hook, or a slow history file, seconds.
    clock = FakeClock()
    clock.install(monkeypatch, fatal_signals)
    stop = fatal_signals.stop_on_fatal_signals(lambda received: None)

    raise_hooked(first)
    clock.sleep(3 * fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGHUP)

    assert deaths == []

    # SIGTERM is what somebody sends who will not wait.
    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]
    stop.finish()


def test_same_request_is_counted_from_the_first_signal(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    clock.install(monkeypatch, fatal_signals)
    fatal_signals.stop_on_fatal_signals(lambda received: None)

    raise_hooked(signal.SIGTERM)
    clock.sleep(0.6 * fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)

    assert deaths == []

    clock.sleep(0.6 * fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]


def test_signal_handled_late_counts_from_when_it_arrived(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The twin of the first signal arrives while the main thread is busy in C
    # code for seconds. The relay saw it arrive; its handler runs afterwards.
    clock = FakeClock()
    clock.install(monkeypatch, fatal_signals)
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    raise_hooked(signal.SIGTERM)
    fatal_signals._unhandled_since[signal.SIGTERM] = clock.monotonic() + 0.002
    clock.sleep(3 * fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)

    assert deaths == []
    assert fatal_signals._unhandled_since == {}

    # The next one is judged by its own arrival again.
    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]
    stop.finish()


def test_reports_queued_before_a_late_handler_are_the_same_request(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first signal and its twin both arrive while the main thread is busy
    # in C code for seconds. The handler runs once for the two; the relay then
    # answers the twin's report, which waited in its pipe, with a signal of now.
    clock = FakeClock()
    clock.install(monkeypatch, fatal_signals)
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    fatal_signals._unhandled_since[signal.SIGTERM] = clock.monotonic()
    clock.sleep(3 * fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)
    raise_hooked(signal.SIGTERM)

    assert asked == [signal.SIGTERM]
    assert deaths == []

    # A second after the run was asked, a SIGTERM is somebody who will not wait.
    clock.sleep(fatal_signals._SAME_REQUEST_SECONDS)
    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]
    stop.finish()


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


def test_process_outlives_its_stopped_runs_for_whoever_asked_it_to(
    deaths: list[int],
) -> None:
    until_the_loop_has_closed = fatal_signals.outlive_stopping_runs()
    stop = fatal_signals.stop_on_fatal_signals(lambda received: None)

    raise_hooked(signal.SIGTERM)
    stop.finish()

    assert deaths == []

    until_the_loop_has_closed.finish()

    assert deaths == [signal.SIGTERM]


def test_signal_that_finds_no_run_to_ask_ends_the_process_at_once(
    deaths: list[int],
) -> None:
    # The run has ended by itself; only its event loop is still closing.
    fatal_signals.outlive_stopping_runs()

    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]


@pytest.mark.parametrize("stopping_a_run", [False, True])
def test_first_process_of_a_container_exits_when_the_signal_cannot_kill_it(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch, stopping_a_run: bool
) -> None:
    # The kernel does not deliver a signal with its default action to pid 1.
    exits: list[int] = []
    monkeypatch.setattr(os, "getpid", lambda: 1)
    monkeypatch.setattr(os, "_exit", exits.append)
    fatal_signals.kill_evaluators_on_fatal_signals()
    stop = fatal_signals.stop_on_fatal_signals(lambda received: None)
    if not stopping_a_run:
        stop.finish()

    raise_hooked(signal.SIGTERM)
    stop.finish()

    assert deaths == [signal.SIGTERM]
    assert exits == [128 + signal.SIGTERM]


def test_request_that_cannot_be_passed_on_ends_the_process(deaths: list[int]) -> None:
    def closed_loop(received: signal.Signals) -> None:
        raise RuntimeError("Event loop is closed")

    fatal_signals.stop_on_fatal_signals(closed_loop)

    raise_hooked(signal.SIGTERM)

    assert deaths == [signal.SIGTERM]


def test_run_that_cannot_be_asked_does_not_keep_the_others_from_stopping(
    deaths: list[int],
) -> None:
    # A run that ended without saying so: its event loop is closed.
    def closed_loop(received: signal.Signals) -> None:
        raise RuntimeError("Event loop is closed")

    asked: list[int] = []
    fatal_signals.stop_on_fatal_signals(closed_loop)
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    raise_hooked(signal.SIGTERM)

    assert asked == [signal.SIGTERM]
    assert deaths == []

    # Nobody waits for the run that could not be asked.
    stop.finish()

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


def test_run_under_nohup_is_stopped_by_the_signal_that_is_left(
    deaths: list[int],
) -> None:
    # `nohup` ignores SIGHUP, which stays ignored; SIGTERM still stops the run.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN

    raise_hooked(signal.SIGTERM)

    assert asked == [signal.SIGTERM]
    assert deaths == []

    stop.finish()

    assert deaths == [signal.SIGTERM]


def test_forked_child_starts_without_its_parents_runs_and_request(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without the relay's pipe, which the child would close: it is this
    # process's own here.
    monkeypatch.setattr(fatal_signals, "_signal_reports", None)
    fatal_signals.stop_on_fatal_signals(lambda received: None)
    raise_hooked(signal.SIGTERM)
    fatal_signals._unhandled_since[signal.SIGHUP] = time.monotonic()

    fatal_signals._forget_the_parent_after_fork()  # What a forked child runs.

    assert fatal_signals._stops == []
    assert fatal_signals._unhandled_since == {}

    # A run that the child starts is not asked to stop for its parent's signal.
    asked: list[int] = []
    stop = fatal_signals.stop_on_fatal_signals(asked.append)

    assert asked == []

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


def _build_a_scheduler_in_a_thread() -> None:
    def build() -> None:
        JobScheduler("local", LocalJobConfig(eval_program_path="evaluate.py")).shutdown()

    thread = threading.Thread(target=build)
    thread.start()
    thread.join()


def test_scheduler_built_off_the_main_thread_says_that_nothing_is_hooked(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Python installs signal handlers from the main thread only.
    with caplog.at_level(logging.WARNING, logger=fatal_signals.__name__):
        _build_a_scheduler_in_a_thread()
        _build_a_scheduler_in_a_thread()

    [warning] = caplog.messages
    assert "SIGTERM and SIGHUP would end this process" in warning
    assert "kill_evaluators_on_fatal_signals() from the main thread" in warning
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_scheduler_built_off_the_main_thread_is_quiet_when_the_signals_are_hooked(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fatal_signals.kill_evaluators_on_fatal_signals()

    with caplog.at_level(logging.WARNING, logger=fatal_signals.__name__):
        _build_a_scheduler_in_a_thread()

    assert caplog.messages == []


@pytest.mark.parametrize("start", ["submit_async", "submit_async_nonblocking", "run"])
def test_evaluation_started_on_the_main_thread_hooks_the_signals_again(
    deaths: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, start: str
) -> None:
    scheduler = JobScheduler("local", LocalJobConfig(eval_program_path="evaluate.py"))
    # An application that adds and removes a handler of its own leaves the
    # default behind, not the hook it replaced.
    loop = asyncio.new_event_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: None)
    loop.remove_signal_handler(signal.SIGTERM)
    loop.close()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    evaluator = MagicMock(spec=ProcessWithLogging)
    evaluator.runtime_seconds.return_value = 0.1
    monkeypatch.setattr(scheduler_module, "submit_local", lambda *a, **k: evaluator)
    monkeypatch.setattr(
        scheduler_module, "monitor_local_evaluation", lambda *a, **k: {}
    )
    try:
        started = getattr(scheduler, start)("main.py", str(tmp_path))
        if asyncio.iscoroutine(started):
            # The runner's way: the evaluator is launched from another thread.
            asyncio.run(started)
    finally:
        scheduler.shutdown()

    assert fatal_signals._is_hooked(signal.SIGTERM)


def test_cancellation_of_a_slurm_job_is_out_of_the_terminals_reach(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A closing terminal hangs up its whole foreground process group. The
    # `scancel` of a stopping run must not be in it, or the job stays.
    report = tmp_path / "scancel-ran-in"
    scancel = tmp_path / "scancel"
    scancel.write_text(
        "#!/bin/sh\n"
        # The process group of this very process, and the job it was given.
        f'echo "$(ps -o pgid= -p $$ | tr -d " ") $1" > {report}\n'
    )
    scancel.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    scheduler = JobScheduler(
        "slurm_conda", SlurmCondaJobConfig(eval_program_path="evaluate.py")
    )
    try:
        assert scheduler._cancel_job("4321") is True
    finally:
        scheduler.shutdown()

    group, job = report.read_text().split()
    assert job == "4321"
    assert int(group) != os.getpgrp()
