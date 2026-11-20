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


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_killed_runner_cleans_up_and_exits_with_the_signals_status(
    tmp_path: Path, signum: int
) -> None:
    events = tmp_path / "events"
    script = tmp_path / "runner.py"
    script.write_text(
        dedent("""
            import sys
            from signal_stop_test_utils import waiting_runner

            def record(event):
                with open(sys.argv[1], "a") as events:
                    events.write(event + "\\n")

            waiting_runner(record).run()
            """)
    )
    tests = Path(__file__).resolve().parent
    runner = subprocess.Popen(
        [sys.executable, str(script), str(events)],
        env={**os.environ, "PYTHONPATH": f"{tests.parent}{os.pathsep}{tests}"},
    )
    try:
        deadline = time.monotonic() + 60.0
        while not (events.exists() and "waiting" in events.read_text()):
            assert runner.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)

        runner.send_signal(signum)

        assert runner.wait(timeout=20.0) == -signum
        assert events.read_text().split() == ["setup", "waiting", *CLEANUP]
    finally:
        runner.kill()
        runner.wait()
