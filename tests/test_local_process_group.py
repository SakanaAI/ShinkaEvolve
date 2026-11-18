import gc
import logging
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


def _submit_without_timeout(
    tmp_path: Path, *, runs_for: float = 30.0
) -> tuple[ProcessWithLogging, int]:
    """Launch the way the asynchronous path does: no timeout is passed."""
    results_dir = tmp_path / "results"
    evaluator = write_evaluator_with_worker(tmp_path, runs_for=runs_for)
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


needs_exit_peek = pytest.mark.skipif(
    not hasattr(os, "waitid"), reason="needs os.waitid to see an exit before reaping"
)


@needs_exit_peek
def test_workers_left_behind_by_an_exited_evaluator_are_killed(tmp_path: Path) -> None:
    process, worker = _submit_without_timeout(tmp_path, runs_for=0.0)
    try:
        assert wait_until(lambda: not is_alive(process.pid))
        assert is_alive(worker)

        assert process.has_exited() is True

        assert process.returncode == 0
        assert process.was_killed is False
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(process.pid, worker)


@needs_exit_peek
def test_waiting_for_an_evaluator_kills_the_workers_it_left_behind(
    tmp_path: Path,
) -> None:
    process, worker = _submit_without_timeout(tmp_path, runs_for=0.0)
    try:
        local.monitor(process, str(tmp_path / "results"), poll_interval=0.01)

        assert process.returncode == 0
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
            import os
            import sys
            import threading
            import time
            from shinka.launch import local

            local.kill_evaluators_on_fatal_signals()
            results_dir = {str(results_dir)!r}
            process = local.submit(
                results_dir,
                [sys.executable, {str(evaluator)!r}, "--results_dir", results_dir],
            )
            print(process.pid, flush=True)
            """)
        + dedent(then)
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


def test_evaluators_do_not_outlive_a_process_that_exits_normally(
    tmp_path: Path,
) -> None:
    exit_once_the_worker_runs = """
        while not os.path.exists(os.path.join(results_dir, "worker.pid")):
            time.sleep(0.01)
        """
    with _orchestrator(tmp_path, then=exit_once_the_worker_runs) as (
        runner,
        evaluator,
        worker,
    ):
        assert runner.wait(timeout=10.0) == 0
        assert wait_until(lambda: not is_alive(evaluator))
        assert wait_until(lambda: not is_alive(worker))


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


@pytest.mark.parametrize(
    "child_ends", ["sys.exit(0)", "os.kill(os.getpid(), signal.SIGTERM)"]
)
def test_forked_child_leaves_its_parents_evaluators_alone(
    tmp_path: Path, child_ends: str
) -> None:
    # The child inherits the exit hook, the signal hook and, here, a wait lock
    # that a thread of the parent held at the moment of the fork.
    fork_a_child = f"""
        import signal

        threading.Thread(target=process.wait, daemon=True).start()
        time.sleep(0.2)
        child = os.fork()
        if child == 0:
            {child_ends}
        print(child, flush=True)
        time.sleep(30)
        """
    with _orchestrator(tmp_path, then=fork_a_child) as (runner, evaluator, worker):
        assert runner.stdout is not None
        child = int(runner.stdout.readline())
        try:
            assert wait_until(lambda: not is_alive(child))
            assert is_alive(evaluator)
            assert is_alive(worker)
        finally:
            kill_leftovers(child)


def _evaluator_stub(**popen_attributes) -> ProcessWithLogging:
    evaluator = MagicMock(spec=subprocess.Popen, pid=4321, **popen_attributes)
    return ProcessWithLogging(
        evaluator,
        (MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
        process_group_id=4321,
    )


def test_evaluator_that_survives_the_kill_does_not_block_its_caller(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A process in an uninterruptible wait (a dead network mount) stays for a while.
    process = _evaluator_stub(returncode=None)
    process.process.poll.return_value = None
    process.process.wait.side_effect = subprocess.TimeoutExpired(["evaluate"], 30)
    monkeypatch.setattr(local.os, "killpg", lambda group_id, signum: None)
    monkeypatch.setattr(local, "_has_exited_unreaped", lambda pid: False)
    monkeypatch.setattr(local, "_LIVE_PROCESSES", {process})

    with caplog.at_level(logging.WARNING, logger=local.__name__):
        assert process.kill_group() is True

    assert process.process.wait.call_args.kwargs["timeout"] > 0
    assert "has not exited" in caplog.text
    # Still alive, so the exit hook gets another go at it.
    assert process in local._LIVE_PROCESSES


def test_group_whose_members_all_exited_is_killed_through_its_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # macOS answers a kill of such a group with a permission error.
    process = _evaluator_stub(returncode=None)
    process.process.poll.return_value = None
    monkeypatch.setattr(local, "_has_exited_unreaped", lambda pid: False)

    def refuse(group_id: int, signum: int) -> None:
        raise PermissionError(group_id)

    monkeypatch.setattr(local.os, "killpg", refuse)

    assert process.kill_group() is True

    process.process.kill.assert_called_once_with()


def test_fatal_signal_handler_cannot_be_reentered_and_never_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispositions_at_kill: list[tuple[object, object]] = []
    process = _evaluator_stub(returncode=None)
    process.process.wait.side_effect = AssertionError("a signal handler must not wait")
    process.process.poll.side_effect = AssertionError("a signal handler must not reap")

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
