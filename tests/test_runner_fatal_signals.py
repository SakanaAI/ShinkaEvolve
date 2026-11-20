import asyncio
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Callable, Iterator

import pytest

from job_lifecycle_test_utils import (
    is_alive,
    kill_leftovers,
    wait_until,
    worker_pid,
    write_evaluator_with_worker,
)
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.billing import BillingPersistenceError
from signal_stop_test_utils import (
    CLEANUP,
    FATAL_SIGNALS,
    FRESH_SIGNALS,
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


def _signal_a_waiting_run(
    signum: int,
    events: list[str],
    *,
    during: str,
    prepare: Callable[[ShinkaEvolveRunner], None] = lambda runner: None,
) -> None:
    """Run the runner's loop and raise the signal when `during` is recorded."""
    raised = False

    def record(event: str) -> None:
        nonlocal raised
        events.append(event)
        if event == during and not raised:
            raised = True
            raise_hooked(signum)

    async def run() -> None:
        runner = waiting_runner(record)
        prepare(runner)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.ensure_future(runner._run_async())

    asyncio.run(asyncio.wait_for(run(), timeout=10.0))


@contextmanager
def _database_with_a_log(path: Path) -> Iterator[SimpleNamespace]:
    """A database of a run with a program that is in its write-ahead log only."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL;")
    connection.execute("CREATE TABLE programs (id TEXT)")
    connection.execute("INSERT INTO programs VALUES ('latest')")
    connection.commit()
    try:
        # Open until the process ends, as the runner's connection is.
        yield SimpleNamespace(
            conn=connection,
            config=SimpleNamespace(db_path=str(path)),
            # What the cleanup asks of the two databases of a run.
            get_all_programs=lambda: [],
            recompute_all_percentiles=lambda scores, score_of: None,
        )
    finally:
        connection.close()


@pytest.fixture
def database(tmp_path: Path) -> Iterator[SimpleNamespace]:
    with _database_with_a_log(tmp_path / "programs.sqlite") as database:
        yield database


@pytest.fixture
def prompt_database(tmp_path: Path) -> Iterator[SimpleNamespace]:
    with _database_with_a_log(tmp_path / "prompts.sqlite") as database:
        yield database


def _programs_in_the_file_alone(database: SimpleNamespace, tmp_path: Path) -> list[str]:
    """What somebody finds who copies `programs.sqlite` without its `-wal`."""
    copy = tmp_path / "copy.sqlite"
    shutil.copy(database.config.db_path, copy)
    connection = sqlite3.connect(copy)
    try:
        return [row[0] for row in connection.execute("SELECT id FROM programs")]
    except sqlite3.OperationalError:  # Not even the table is in the file.
        return []
    finally:
        connection.close()


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_signalled_run_leaves_its_programs_in_the_database_file(
    deaths: list[int],
    cleanup_events: list[str],
    database: SimpleNamespace,
    tmp_path: Path,
    signum: int,
) -> None:
    # A process that dies of a signal never closes its connection, which is
    # what folds the write-ahead log into the file after Ctrl-C.
    def with_database(runner: ShinkaEvolveRunner) -> None:
        runner.db = database

    assert _programs_in_the_file_alone(database, tmp_path) == []

    _signal_a_waiting_run(signum, cleanup_events, during="waiting", prepare=with_database)

    assert deaths == [signum]
    assert _programs_in_the_file_alone(database, tmp_path) == ["latest"]
    assert Path(f"{database.config.db_path}-wal").stat().st_size == 0


def test_signalled_run_leaves_its_prompts_in_their_database_file_too(
    deaths: list[int],
    cleanup_events: list[str],
    database: SimpleNamespace,
    prompt_database: SimpleNamespace,
    tmp_path: Path,
) -> None:
    def with_databases(runner: ShinkaEvolveRunner) -> None:
        runner.db = database
        runner.prompt_db = prompt_database

    _signal_a_waiting_run(
        signal.SIGTERM, cleanup_events, during="waiting", prepare=with_databases
    )

    assert deaths == [signal.SIGTERM]
    for stored in (database, prompt_database):
        assert _programs_in_the_file_alone(stored, tmp_path) == ["latest"]


def test_run_that_no_signal_stopped_leaves_the_log_to_the_closing_connection(
    deaths: list[int],
    cleanup_events: list[str],
    database: SimpleNamespace,
    tmp_path: Path,
) -> None:
    async def fail() -> None:
        raise RuntimeError("the run fails by itself")

    async def run() -> None:
        runner = waiting_runner(cleanup_events.append)
        runner.db = database
        runner._verify_database_ready = fail
        with pytest.raises(RuntimeError, match="fails by itself"):
            await asyncio.ensure_future(runner._run_async())

    asyncio.run(asyncio.wait_for(run(), timeout=10.0))

    assert deaths == []
    assert _programs_in_the_file_alone(database, tmp_path) == []


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_fatal_signal_stops_a_run_the_way_ctrl_c_does(
    deaths: list[int],
    cleanup_events: list[str],
    caplog: pytest.LogCaptureFixture,
    signum: int,
) -> None:
    with caplog.at_level(logging.ERROR, logger="shinka.launch.fatal_signals"):
        _signal_a_waiting_run(signum, cleanup_events, during="waiting")

    # Charges saved, database closed, evaluations cancelled; then the signal.
    assert cleanup_events == ["setup", "waiting", *CLEANUP]
    assert deaths == [signum]
    # The cancellation that stopped the run is no failure of the run.
    assert caplog.messages == []


def test_signalled_run_cleans_up_when_an_evaluator_cannot_be_killed(
    deaths: list[int], cleanup_events: list[str]
) -> None:
    def with_an_evaluator_of_somebody_else(runner: ShinkaEvolveRunner) -> None:
        def refuse() -> None:
            raise PermissionError("Operation not permitted")

        runner.running_jobs[0].job_id.kill_without_reaping = refuse

    _signal_a_waiting_run(
        signal.SIGTERM,
        cleanup_events,
        during="waiting",
        prepare=with_an_evaluator_of_somebody_else,
    )

    assert cleanup_events == ["setup", "waiting", *CLEANUP[1:]]
    assert deaths == [signal.SIGTERM]


def test_fatal_signal_during_the_cleanup_of_a_run_that_ended_lets_it_finish(
    deaths: list[int], cleanup_events: list[str]
) -> None:
    def record(event: str) -> None:
        cleanup_events.append(event)
        if event == "flush_charges":
            raise_hooked(signal.SIGTERM)

    async def fail() -> None:
        record("waiting")
        raise RuntimeError("the run fails by itself")

    async def run() -> None:
        runner = waiting_runner(record)
        runner._verify_database_ready = fail
        # Its own error leaves the run, not a cancellation by the signal.
        with pytest.raises(RuntimeError, match="fails by itself"):
            await asyncio.ensure_future(runner._run_async())

    asyncio.run(asyncio.wait_for(run(), timeout=10.0))

    assert cleanup_events == ["setup", "waiting", *CLEANUP]
    assert deaths == [signal.SIGTERM]


def test_signalled_run_dies_of_the_signal_when_its_cleanup_fails(
    deaths: list[int], cleanup_events: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    async def fail_to_save_charges() -> None:
        raise BillingPersistenceError("journal unavailable")

    def record(event: str) -> None:
        cleanup_events.append(event)
        if event == "waiting":
            raise_hooked(signal.SIGTERM)

    async def run() -> None:
        runner = waiting_runner(record)
        runner._flush_unwritten_charges = fail_to_save_charges
        with pytest.raises(BillingPersistenceError):
            await asyncio.ensure_future(runner._run_async())

    with caplog.at_level(logging.ERROR, logger="shinka.launch.fatal_signals"):
        asyncio.run(asyncio.wait_for(run(), timeout=10.0))

    assert cleanup_events[-4:] == CLEANUP[-4:]
    assert deaths == [signal.SIGTERM]
    # Nothing else reports the error of a process that dies here.
    assert "journal unavailable" in caplog.text


def _killed_runner_events(
    tmp_path: Path, signum: int, runner_script: str, *, signal_at: str = "waiting"
) -> list[str]:
    """Events a runner process records; the signal reaches it at `signal_at`."""
    events = tmp_path / "events"
    script = tmp_path / "runner.py"
    script.write_text(
        FRESH_SIGNALS
        + dedent("""
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
        deadline = time.monotonic() + 120.0
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

        run_async = runner._run_async

        async def run_and_say_when_it_is_over():
            # The end of the cleanup is not yet the end of the run: the run
            # is over once it no longer waits for a signal.
            try:
                await run_async()
            finally:
                record("run_over")

        runner._setup_async = setup_with_a_stuck_thread
        runner._verify_database_ready = fail
        runner._run_async = run_and_say_when_it_is_over
        runner.run()
        """
    events = _killed_runner_events(
        tmp_path, signal.SIGTERM, fail_with_a_thread_running, signal_at="run_over"
    )

    assert events == ["setup", *CLEANUP, "run_over"]


def test_stopping_runner_kills_its_local_evaluators_before_it_cleans_up(
    tmp_path: Path,
) -> None:
    # Whoever sent SIGTERM may follow it with SIGKILL before a slow cleanup
    # reaches the cancellation of the evaluations.
    results_dir = tmp_path / "results"
    evaluator = write_evaluator_with_worker(tmp_path)
    events = tmp_path / "events"
    script = tmp_path / "runner.py"
    script.write_text(
        FRESH_SIGNALS
        + dedent(f"""
            import asyncio
            import sys
            from shinka.launch import local
            from signal_stop_test_utils import waiting_runner

            def record(event):
                with open({str(events)!r}, "a") as events:
                    events.write(event + "\\n")

            async def slow_cleanup_step():
                record("flush_charges")
                await asyncio.sleep(60)

            runner = waiting_runner(record)
            runner.running_jobs[0].job_id = local.submit(
                {str(results_dir)!r},
                [sys.executable, {str(evaluator)!r}, "--results_dir", {str(results_dir)!r}],
            )
            runner._flush_unwritten_charges = slow_cleanup_step
            runner.run()
            """)
    )
    tests = Path(__file__).resolve().parent
    runner = subprocess.Popen(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": f"{tests.parent}{os.pathsep}{tests}"},
    )
    evaluator_pid = worker = 0
    try:
        # The runner's imports alone take seconds on a busy machine.
        deadline = time.monotonic() + 120.0
        while not (events.exists() and "waiting" in events.read_text()):
            assert runner.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        while not (results_dir / "worker.pid").exists():
            assert runner.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        worker = worker_pid(results_dir)
        evaluator_pid = os.getpgid(worker)  # The evaluator leads its group.

        runner.send_signal(signal.SIGTERM)

        assert wait_until(lambda: not is_alive(evaluator_pid))
        assert wait_until(lambda: not is_alive(worker))
        assert wait_until(lambda: "flush_charges" in events.read_text())
        assert runner.poll() is None  # Still cleaning up.
    finally:
        runner.kill()
        runner.wait()
        kill_leftovers(*(pid for pid in (evaluator_pid, worker) if pid))
