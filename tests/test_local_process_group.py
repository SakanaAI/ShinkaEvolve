import gc
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from job_lifecycle_test_utils import (
    is_alive,
    kill_leftovers,
    wait_until,
    worker_pid,
    write_evaluator_with_worker,
)
from shinka.launch import ProcessWithLogging, local

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="requires POSIX process groups"
)


def _submit_without_timeout(tmp_path: Path) -> tuple[ProcessWithLogging, int]:
    """Launch the way the asynchronous path does: no timeout is passed."""
    results_dir = tmp_path / "results"
    evaluator = write_evaluator_with_worker(tmp_path)
    process = local.submit(
        str(results_dir),
        [sys.executable, str(evaluator), "--results_dir", str(results_dir)],
    )
    return process, worker_pid(results_dir)


def test_killing_an_evaluator_takes_its_workers_with_it(tmp_path: Path) -> None:
    process, worker = _submit_without_timeout(tmp_path)
    try:
        assert os.getpgid(process.pid) == process.pid
        assert os.getpgid(worker) == process.pid

        assert process.kill_group() is True

        assert process.returncode == -signal.SIGKILL
        assert process.was_killed is True
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(process.pid, worker)


def test_interpreter_exit_kills_evaluators_that_are_still_running(
    tmp_path: Path,
) -> None:
    process, worker = _submit_without_timeout(tmp_path)
    try:
        local._kill_live_evaluators()

        assert process.returncode == -signal.SIGKILL
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(process.pid, worker)


def test_exit_hook_reaches_an_evaluator_nobody_kept_a_handle_on(
    tmp_path: Path,
) -> None:
    process, worker = _submit_without_timeout(tmp_path)
    root = process.pid
    del process
    gc.collect()
    try:
        local._kill_live_evaluators()

        assert wait_until(lambda: not is_alive(root))
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(root, worker)


def test_finished_evaluators_are_not_kept_alive_in_memory(tmp_path: Path) -> None:
    process = local.submit(str(tmp_path), [sys.executable, "-c", "pass"])
    assert process in local._LIVE_PROCESSES
    assert process.wait(timeout=5.0) == 0

    assert process.has_exited() is True

    assert process not in local._LIVE_PROCESSES


@contextmanager
def _orchestrator(
    tmp_path: Path, then: str
) -> Iterator[tuple[subprocess.Popen, int, int]]:
    """A process with the fatal-signal hook and one evaluator, then running `then`.

    Yields the process, the evaluator's pid and its worker's pid.
    """
    results_dir = tmp_path / "results"
    evaluator = write_evaluator_with_worker(tmp_path)
    runner_script = tmp_path / "runner.py"
    runner_script.write_text(
        dedent(f"""
            import sys
            import time
            from shinka.launch import local

            local.kill_evaluators_on_fatal_signals()
            results_dir = {str(results_dir)!r}
            process = local.submit(
                results_dir,
                [sys.executable, {str(evaluator)!r}, "--results_dir", results_dir],
            )
            print(process.pid, flush=True)
            {then}
            """)
    )
    runner = subprocess.Popen(
        [sys.executable, str(runner_script)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    evaluator_pid = worker = 0
    try:
        assert runner.stdout is not None
        evaluator_pid = int(runner.stdout.readline())
        worker = worker_pid(results_dir)
        yield runner, evaluator_pid, worker
    finally:
        runner.kill()
        runner.wait()
        kill_leftovers(*(pid for pid in (evaluator_pid, worker) if pid))


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_fatal_signal_takes_evaluators_along(tmp_path: Path, signum: int) -> None:
    with _orchestrator(tmp_path, then="time.sleep(30)") as (runner, evaluator, worker):
        runner.send_signal(signum)

        assert runner.wait(timeout=10.0) == -signum
        assert wait_until(lambda: not is_alive(evaluator))
        assert wait_until(lambda: not is_alive(worker))


def test_fatal_signal_ends_a_process_that_is_waiting_on_its_evaluator(
    tmp_path: Path,
) -> None:
    # The interrupted wait holds the lock that reaping the evaluator needs.
    with _orchestrator(tmp_path, then="process.wait()") as (runner, evaluator, worker):
        runner.send_signal(signal.SIGTERM)

        assert runner.wait(timeout=10.0) == -signal.SIGTERM
        assert wait_until(lambda: not is_alive(evaluator))
        assert wait_until(lambda: not is_alive(worker))


def test_second_fatal_signal_does_not_interrupt_the_first(tmp_path: Path) -> None:
    # A closing terminal can deliver SIGHUP twice within milliseconds.
    with _orchestrator(tmp_path, then="time.sleep(30)") as (runner, evaluator, worker):
        runner.send_signal(signal.SIGHUP)
        runner.send_signal(signal.SIGTERM)

        assert runner.wait(timeout=10.0) in (-signal.SIGHUP, -signal.SIGTERM)
        assert wait_until(lambda: not is_alive(evaluator))
        assert wait_until(lambda: not is_alive(worker))


def test_fatal_signal_handler_cannot_be_reentered_and_never_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispositions_at_kill: list[tuple[object, object]] = []
    evaluator = MagicMock(spec=subprocess.Popen, pid=4321, returncode=None)
    evaluator.wait.side_effect = AssertionError("a signal handler must not wait")
    evaluator.poll.side_effect = AssertionError("a signal handler must not reap")
    process = ProcessWithLogging(
        evaluator,
        (MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
        process_group_id=4321,
    )

    def record_kill(group_id: int, signum: int) -> None:
        assert (group_id, signum) == (4321, signal.SIGKILL)
        dispositions_at_kill.append(
            (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP))
        )

    resent: list[tuple[int, object]] = []
    monkeypatch.setattr(local, "_LIVE_PROCESSES", {process})
    monkeypatch.setattr(local.os, "killpg", record_kill)
    monkeypatch.setattr(
        local.os,
        "kill",
        lambda pid, signum: resent.append((signum, signal.getsignal(signum))),
    )
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        local._die_with_evaluators(signal.SIGTERM, None)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    assert dispositions_at_kill == [(signal.SIG_IGN, signal.SIG_IGN)]
    assert resent == [(signal.SIGTERM, signal.SIG_DFL)]


def test_fatal_signal_hook_leaves_application_handlers_alone() -> None:
    def application_handler(signum, frame) -> None:
        pass

    previous = signal.signal(signal.SIGTERM, application_handler)
    hangup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        local.kill_evaluators_on_fatal_signals()

        assert signal.getsignal(signal.SIGTERM) is application_handler
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGHUP, hangup)


def test_an_exited_evaluator_is_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = local.submit(str(tmp_path), [sys.executable, "-c", "pass"])
    assert process.wait(timeout=5.0) == 0

    def refuse(*args):
        raise AssertionError("a reaped process group id may belong to someone else")

    monkeypatch.setattr(local.os, "killpg", refuse)

    assert process.kill_group() is False
    assert process.was_killed is False
    assert process.has_exited() is True
    assert 0.0 <= process.runtime_seconds() < 5.0
