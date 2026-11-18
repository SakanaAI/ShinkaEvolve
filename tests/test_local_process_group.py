import gc
import os
import signal
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

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


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_fatal_signal_takes_evaluators_along(tmp_path: Path, signum: int) -> None:
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
            time.sleep(30)
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

        runner.send_signal(signum)

        assert runner.wait(timeout=10.0) == -signum
        assert wait_until(lambda: not is_alive(evaluator_pid))
        assert wait_until(lambda: not is_alive(worker))
    finally:
        runner.kill()
        runner.wait()
        kill_leftovers(*(pid for pid in (evaluator_pid, worker) if pid))


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
