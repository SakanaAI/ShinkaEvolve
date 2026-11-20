import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Iterator

import pytest

from signal_stop_test_utils import (
    CLEANUP,
    FATAL_SIGNALS,
    raise_hooked,
    recorded_deaths,
    waiting_runner,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")


@pytest.fixture
def deaths(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    with recorded_deaths(monkeypatch) as died_of:
        yield died_of


@pytest.fixture
def cleanup_events() -> list[str]:
    return []


def _signal_a_waiting_run(signum: int, events: list[str], *, during: str) -> None:
    """Run the runner's loop and raise the signal when `during` is recorded."""
    raised = False

    def record(event: str) -> None:
        nonlocal raised
        events.append(event)
        if event == during and not raised:
            raised = True
            raise_hooked(signum)

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.ensure_future(waiting_runner(record)._run_async())

    asyncio.run(asyncio.wait_for(run(), timeout=10.0))


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_fatal_signal_stops_a_run_the_way_ctrl_c_does(
    deaths: list[int], cleanup_events: list[str], signum: int
) -> None:
    _signal_a_waiting_run(signum, cleanup_events, during="waiting")

    # Charges saved, database closed, evaluations cancelled; then the signal.
    assert cleanup_events == ["setup", "waiting", *CLEANUP]
    assert deaths == [signum]


def test_fatal_signal_during_cleanup_does_not_cut_it_short(
    deaths: list[int], cleanup_events: list[str]
) -> None:
    async def run() -> None:
        def record(event: str) -> None:
            cleanup_events.append(event)
            if event == "flush_charges":
                raise_hooked(signal.SIGTERM)

        runner = waiting_runner(record)
        task = asyncio.ensure_future(runner._run_async())
        while "waiting" not in cleanup_events:
            await asyncio.sleep(0)
        task.cancel()  # Ctrl-C, or the end of the run.
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(run(), timeout=10.0))

    assert cleanup_events == ["setup", "waiting", *CLEANUP]
    assert deaths == [signal.SIGTERM]


def _killed_runner_events(
    tmp_path: Path, signum: int, runner_script: str, *, signal_at: str = "waiting"
) -> list[str]:
    """Events a runner process records; the signal reaches it at `signal_at`."""
    events = tmp_path / "events"
    script = tmp_path / "runner.py"
    script.write_text(
        dedent("""
            import asyncio
            import sys
            import threading
            import time
            from signal_stop_test_utils import waiting_runner

            def record(event):
                with open(sys.argv[1], "a") as events:
                    events.write(event + "\\n")
            """)
        + dedent(runner_script)
    )
    tests = Path(__file__).resolve().parent
    runner = subprocess.Popen(
        [sys.executable, str(script), str(events)],
        env={**os.environ, "PYTHONPATH": f"{tests.parent}{os.pathsep}{tests}"},
    )
    try:
        deadline = time.monotonic() + 60.0
        while not (events.exists() and signal_at in events.read_text()):
            assert runner.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)

        runner.send_signal(signum)

        assert runner.wait(timeout=20.0) == -signum
        return events.read_text().split()
    finally:
        runner.kill()
        runner.wait()


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_killed_runner_cleans_up_and_exits_with_the_signals_status(
    tmp_path: Path, signum: int
) -> None:
    events = _killed_runner_events(tmp_path, signum, "waiting_runner(record).run()")

    assert events == ["setup", "waiting", *CLEANUP]


def test_killed_runner_lets_the_event_loops_threads_finish(tmp_path: Path) -> None:
    # Closing the loop waits for work it handed to threads, as after Ctrl-C.
    work_in_a_thread = """
        cleaned_up = threading.Event()

        def record_cleanup(event):
            record(event)
            if event == "scheduler.shutdown":
                cleaned_up.set()

        def write_something():
            cleaned_up.wait()
            time.sleep(0.3)
            record("thread_finished")

        runner = waiting_runner(record_cleanup)
        setup = runner._setup_async

        async def setup_with_work_in_a_thread():
            asyncio.get_running_loop().run_in_executor(None, write_something)
            await setup()

        runner._setup_async = setup_with_work_in_a_thread
        runner.run()
        """
    events = _killed_runner_events(tmp_path, signal.SIGTERM, work_in_a_thread)

    assert events == ["setup", "waiting", *CLEANUP, "thread_finished"]


def test_signal_ends_a_runner_at_once_whose_run_is_over(tmp_path: Path) -> None:
    # The run failed by itself and has cleaned up. The loop is closing and
    # waits for a thread; nothing is left that the signal could ask to stop.
    fail_with_a_thread_running = """
        runner = waiting_runner(record)
        setup = runner._setup_async

        async def setup_with_a_stuck_thread():
            asyncio.get_running_loop().run_in_executor(None, time.sleep, 30)
            await setup()

        async def fail():
            raise RuntimeError("the run fails by itself")

        runner._setup_async = setup_with_a_stuck_thread
        runner._verify_database_ready = fail
        runner.run()
        """
    events = _killed_runner_events(
        tmp_path, signal.SIGTERM, fail_with_a_thread_running, signal_at="scheduler.shutdown"
    )

    assert events == ["setup", *CLEANUP]
