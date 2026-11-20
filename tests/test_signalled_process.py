import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Iterator

import pytest

from job_lifecycle_test_utils import (
    is_alive,
    kill_leftovers,
    wait_until,
    worker_pid,
    write_evaluator_with_worker,
)
from shinka.launch import fatal_signals
from signal_stop_test_utils import FATAL_SIGNALS, FRESH_SIGNALS

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")


@contextmanager
def _run(tmp_path: Path, *, start: str = "", cleanup: str = "") -> Iterator[subprocess.Popen]:
    """A process whose run waits for a fatal signal, cleans up and finishes.

    It prints a line once it waits, and leaves the files "stopping" and
    "cleaned" behind as it gets that far.
    """
    script = tmp_path / "run.py"
    script.write_text(
        FRESH_SIGNALS
        + dedent("""
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
    never_started = threading.Timer(60.0, run.kill)
    never_started.start()
    try:
        assert run.stdout is not None
        assert run.stdout.readline() == "waiting\n"
        never_started.cancel()
        yield run
    finally:
        never_started.cancel()
        run.kill()
        run.wait()


@pytest.mark.parametrize("signum", FATAL_SIGNALS)
def test_signalled_run_cleans_up_and_dies_of_the_signal(
    tmp_path: Path, signum: int
) -> None:
    # Its last words sit in a buffer: the pipe is not a terminal.
    with _run(tmp_path, cleanup='print("cleaned up")') as run:
        run.send_signal(signum)

        assert run.wait(timeout=10.0) == -signum
        assert (tmp_path / "cleaned").exists()
        assert run.stdout is not None
        assert run.stdout.read() == "cleaned up\n"


def test_closing_terminal_does_not_cut_the_cleanup_short(tmp_path: Path) -> None:
    with _run(tmp_path, cleanup="time.sleep(0.3)") as run:
        run.send_signal(signal.SIGHUP)
        time.sleep(0.01)  # Far enough apart for the handler to run twice.
        run.send_signal(signal.SIGHUP)

        assert run.wait(timeout=10.0) == -signal.SIGHUP
        assert (tmp_path / "cleaned").exists()


def test_terminal_whose_shell_exits_slowly_does_not_cut_the_cleanup_short(
    tmp_path: Path,
) -> None:
    # The shell hangs up its jobs, and the kernel hangs up again once the
    # shell has exited: seconds later when the shell has an exit hook.
    with _run(tmp_path, cleanup="time.sleep(2.5)") as run:
        run.send_signal(signal.SIGHUP)
        assert wait_until((tmp_path / "stopping").exists)
        time.sleep(fatal_signals._SAME_REQUEST_SECONDS + 0.2)
        run.send_signal(signal.SIGHUP)

        assert run.wait(timeout=10.0) == -signal.SIGHUP
        assert (tmp_path / "cleaned").exists()


# One call into C code that no signal interrupts and that leaves the
# interpreter to other threads, such as a slow commit.
BUSY_STEP = """
    import hashlib

    def rounds_for(seconds):
        rounds, started = 50_000, time.perf_counter()
        hashlib.pbkdf2_hmac("sha256", b"password", b"salt", rounds)
        per_second = rounds / (time.perf_counter() - started)
        return int(per_second * seconds)

    def busy(seconds, rounds=None):
        hashlib.pbkdf2_hmac("sha256", b"password", b"salt", rounds or rounds_for(seconds))
    """


def test_twin_signal_handled_after_a_busy_cleanup_step_is_the_same_request(
    tmp_path: Path,
) -> None:
    # SIGTERM, which a later one of would end the run: a hangup never does.
    # The process sends itself the twin, by its own clock: a test that
    # stalls for a second would send a later signal instead.
    twin_during_a_busy_step = """
        rounds = rounds_for(2.0)
        threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
        started = time.perf_counter()
        busy(2.0, rounds)
        (here / "busy").write_text(str(time.perf_counter() - started))
        """
    with _run(tmp_path, start=BUSY_STEP, cleanup=twin_during_a_busy_step) as run:
        run.send_signal(signal.SIGTERM)

        assert run.wait(timeout=20.0) == -signal.SIGTERM
        assert (tmp_path / "cleaned").exists()
        # Otherwise the twin was handled inside the window by any clock.
        busy_seconds = float((tmp_path / "busy").read_text())
        assert busy_seconds > fatal_signals._SAME_REQUEST_SECONDS + 0.3


def test_twin_signals_that_find_the_main_thread_busy_are_one_request(
    tmp_path: Path,
) -> None:
    # Both signals arrive during one busy step, so the handler runs for them
    # seconds later, once. Each has left a report with the relay, which
    # answers the second report after the handler ran: more than a second
    # after the signals arrived, and still no later signal.
    twins_while_busy = dedent(BUSY_STEP) + dedent("""
        def twins():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.002)
            os.kill(os.getpid(), signal.SIGTERM)

        rounds = rounds_for(3.0)
        threading.Timer(0.2, twins).start()
        started = time.perf_counter()
        busy(3.0, rounds)
        (here / "busy").write_text(str(time.perf_counter() - started))
        """)
    with _run(tmp_path, start=twins_while_busy, cleanup="time.sleep(0.5)") as run:
        assert run.wait(timeout=20.0) == -signal.SIGTERM
        assert (tmp_path / "cleaned").exists()
        # Otherwise the handler ran inside the window and nothing was tested.
        busy_seconds = float((tmp_path / "busy").read_text())
        assert busy_seconds > fatal_signals._SAME_REQUEST_SECONDS + 0.3


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


def test_forked_child_of_a_run_dies_of_its_own_signal(tmp_path: Path) -> None:
    # The child inherits the hook and the run's request, which would write
    # to the parent's pipe: the parent would stop and the child live on.
    fork_a_child = """
        if os.fork() == 0:
            # Written by the child: by now it has dropped what it inherited.
            (here / "child.tmp").write_text(str(os.getpid()))
            (here / "child.tmp").rename(here / "child")
            time.sleep(30)
            os._exit(0)
        """
    with _run(tmp_path, start=fork_a_child) as run:
        assert wait_until((tmp_path / "child").exists)
        child = int((tmp_path / "child").read_text())
        try:
            os.kill(child, signal.SIGTERM)

            assert wait_until(lambda: not is_alive(child))
            with pytest.raises(subprocess.TimeoutExpired):
                run.wait(timeout=0.5)
            assert not (tmp_path / "stopping").exists()
        finally:
            kill_leftovers(child)
