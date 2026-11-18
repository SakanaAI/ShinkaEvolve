import atexit
from contextlib import ExitStack
import subprocess
import time
import threading
import os
import signal
from pathlib import Path
from typing import Optional, Tuple, TextIO, Dict
from shinka.utils import load_results, parse_time_to_seconds
import logging

logger = logging.getLogger(__name__)

# Evaluators lead their own process group, so signals sent to this process's
# group (Ctrl-C, a terminal hangup) no longer reach them. Whatever is still
# running when this process ends is killed from here instead. Entries are held
# until their exit is seen: a launched evaluator nobody kept a handle on is
# exactly the one that needs killing.
_LIVE_PROCESSES: "set[ProcessWithLogging]" = set()


class ProcessWithLogging:
    """Wrapper for subprocess.Popen with real-time logging capabilities."""

    def __init__(
        self,
        process: subprocess.Popen,
        log_files: Tuple[TextIO, TextIO],
        log_threads: Tuple[threading.Thread, threading.Thread],
        *,
        process_group_id: Optional[int] = None,
        launched_at: Optional[float] = None,
    ):
        self.process = process
        self.log_files = log_files
        self.log_threads = log_threads
        self._process_group_id = process_group_id
        self.launched_at = time.monotonic() if launched_at is None else launched_at
        self._exit_observed_at: Optional[float] = None
        self.was_killed = False

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped process."""
        return getattr(self.process, name)

    def __str__(self):
        """Return a string representation showing the PID."""
        return f"ProcessWithLogging(PID: {self.process.pid})"

    def __repr__(self):
        """Return a detailed string representation."""
        return f"ProcessWithLogging(PID: {self.process.pid}, returncode: {self.process.returncode})"

    def kill_for_timeout(self) -> None:
        """Kill the evaluator's isolated group when available and reap the root."""
        self.was_killed = True
        _kill_and_reap(self.process, self._process_group_id)
        _LIVE_PROCESSES.discard(self)

    def kill_group(self) -> bool:
        """Kill a running evaluator together with its descendants.

        Returns False when the evaluator had already exited. Its group id is
        then free for reuse, so nothing is signalled.
        """
        if self.has_exited():
            return False
        self.was_killed = True
        _kill_and_reap(self.process, self._process_group_id)
        _LIVE_PROCESSES.discard(self)
        return True

    def kill_without_reaping(self) -> None:
        """Kill the evaluator's group and return at once, for a signal handler.

        Reaping takes a lock that the interrupted code may hold. An evaluator
        nobody reaped still owns its group id, so nothing else is signalled.
        """
        if self.process.returncode is None:
            _kill_group(self.process, self._process_group_id)

    def has_exited(self) -> bool:
        """Poll the evaluator and remember when its exit was first seen."""
        self._kill_group_of_exited_leader()
        if self.process.poll() is None:
            return False
        if self._exit_observed_at is None:
            self._exit_observed_at = time.monotonic()
        _LIVE_PROCESSES.discard(self)
        return True

    def _kill_group_of_exited_leader(self) -> None:
        """Kill what an evaluator that exited on its own left running.

        Only while the evaluator is unreaped: its pid, and with it the group
        id, cannot have passed to another process until then.
        """
        if self.process.returncode is not None or self._process_group_id is None:
            return
        if _has_exited_unreaped(self.process.pid):
            _kill_group(self.process, self._process_group_id)

    def runtime_seconds(self) -> float:
        """Seconds from launch until the exit was observed, or until now."""
        end = self._exit_observed_at
        return (time.monotonic() if end is None else end) - self.launched_at

    def cleanup_logging(self):
        """Clean up logging threads and files."""
        if self.process.returncode is not None:
            _LIVE_PROCESSES.discard(self)
        # Wait for logging threads to finish
        for thread in self.log_threads:
            thread.join(timeout=1.0)

        # Close log files
        for file_handle in self.log_files:
            try:
                file_handle.close()
            except Exception as e:
                logger.error(f"Error closing log file: {e}")


def _has_exited_unreaped(pid: int) -> bool:
    """Whether a child has exited and nobody has collected its status yet."""
    waitid = getattr(os, "waitid", None)
    if waitid is None:  # macOS before Python 3.13
        return False
    try:
        return waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return False  # Another thread collected it.


def _kill_group(process: subprocess.Popen, group_id: Optional[int]) -> None:
    try:
        if group_id is None:
            process.kill()
        else:
            os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS reports a group whose members have all exited this way.
        process.kill()


def _kill_and_reap(process: subprocess.Popen, group_id: Optional[int]) -> None:
    _kill_group(process, group_id)
    process.wait()


@atexit.register
def _kill_live_evaluators() -> None:
    for process in list(_LIVE_PROCESSES):
        try:
            process.kill_group()
        except Exception as e:
            logger.warning(f"Could not kill evaluator {process}: {e}")


def kill_evaluators_on_fatal_signals() -> None:
    """Take running evaluators along when SIGTERM or SIGHUP ends this process.

    Neither signal runs exit handlers, and evaluators in their own process
    group would otherwise outlive the run. Only default dispositions are
    replaced: a handler the application installed, or ``nohup``, stays in charge.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return
    for signum in _fatal_signals():
        if signal.getsignal(signum) is signal.SIG_DFL:
            signal.signal(signum, _die_with_evaluators)


def _fatal_signals() -> Tuple[int, int]:
    return signal.SIGTERM, signal.SIGHUP


def _die_with_evaluators(signum, frame) -> None:
    # A closing terminal can send a second signal while this one is handled;
    # it must not start the handler again inside itself.
    for fatal in _fatal_signals():
        signal.signal(fatal, signal.SIG_IGN)
    try:
        for process in list(_LIVE_PROCESSES):
            process.kill_without_reaping()
    finally:
        # Then end the way the signal would have ended this process.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _stream_output(pipe, file_handle, verbose_prefix=None):
    """
    Read from a pipe and write to a file handle in real-time.

    Args:
        pipe: The subprocess pipe to read from
        file_handle: The file handle to write to
        verbose_prefix: Optional prefix for verbose logging
    """
    try:
        for line in iter(pipe.readline, ""):
            if line:
                file_handle.write(line)
                file_handle.flush()  # Force immediate write to disk
                if verbose_prefix:
                    logger.debug(f"{verbose_prefix}: {line.strip()}")
    except Exception as e:
        logger.error(f"Error in stream output thread: {e}")
    finally:
        pipe.close()


def _start_log_threads(
    process: subprocess.Popen,
    files: Tuple[TextIO, TextIO],
    verbose: bool,
    group_id: Optional[int],
) -> Tuple[threading.Thread, threading.Thread]:
    stdout, stderr = process.stdout, process.stderr
    assert stdout is not None and stderr is not None
    pipes = (stdout, stderr)
    threads: list[threading.Thread] = []
    try:
        for pipe, file, prefix in zip(pipes, files, ("STDOUT", "STDERR")):
            thread = threading.Thread(
                target=_stream_output,
                args=(pipe, file, prefix if verbose else None),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
    except BaseException:
        _kill_and_reap(process, group_id)
        for thread in threads:
            thread.join(timeout=1.0)
        for pipe in pipes[len(threads) :]:
            pipe.close()
        raise
    return threads[0], threads[1]


def submit(
    log_dir: str,
    cmd: list[str],
    verbose: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
    *,
    timeout: Optional[str] = None,
):
    """
    Submits a command for local execution with real-time logging.

    Args:
        log_dir: The directory to store logs.
        cmd: The command and its arguments as a list of strings.
        verbose: Whether to enable verbose logging.
        timeout: Accepted for callers that monitor with a timeout. Every
            evaluator leads its own process group, with or without one, so a
            timeout or cancellation can take its descendants down with it.

    Returns:
        ProcessWithLogging: Wrapper containing the Popen object and logging.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir_path / "job_log.out"
    stderr_path = log_dir_path / "job_log.err"

    # Set up environment to force unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Force Python to be unbuffered
    env["PYTHONIOENCODING"] = "utf-8"  # Ensure proper encoding
    if env_overrides:
        env.update(env_overrides)

    start_new_session = os.name == "posix"
    with ExitStack() as cleanup:
        files = (
            cleanup.enter_context(open(stdout_path, "w", buffering=1)),
            cleanup.enter_context(open(stderr_path, "w", buffering=1)),
        )
        launched_at = time.monotonic()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=start_new_session,
        )
        group_id = process.pid if start_new_session else None
        threads = _start_log_threads(process, files, verbose, group_id)
        wrapped_process = ProcessWithLogging(
            process, files, threads, process_group_id=group_id, launched_at=launched_at
        )
        cleanup.pop_all()
    _LIVE_PROCESSES.add(wrapped_process)

    if verbose:
        logger.info(f"Submitted local process with PID: {process.pid}")
        logger.info(f"Launched local command: {' '.join(cmd)}")
    return wrapped_process


def monitor(
    process: ProcessWithLogging,
    results_dir: str,
    poll_interval: float = 0.5,
    verbose: bool = False,
    timeout: Optional[str] = None,
):
    """Monitor an ordinary Local job and load any available results."""
    _wait_for_completion(process, poll_interval, verbose, timeout)
    return load_results(results_dir)


def monitor_evaluation(
    process: ProcessWithLogging, results_dir: str, *, timeout: Optional[str]
) -> dict:
    """Load seed artifacts only after the evaluator exits successfully."""
    _wait_for_completion(process, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Local evaluator exited with status {process.returncode}")
    return load_results(results_dir)


def _wait_for_completion(
    process: ProcessWithLogging,
    poll_interval: float = 0.5,
    verbose: bool = False,
    timeout: Optional[str] = None,
) -> None:
    if verbose:
        logger.info(f"Monitoring local process with PID: {process.pid}...")

    timeout_seconds = parse_time_to_seconds(timeout) if timeout is not None else None

    try:
        if timeout_seconds is None:
            while not process.has_exited():
                if verbose:
                    logger.info(f"Process {process.pid} is still running...")
                time.sleep(poll_interval)
        elif process.returncode is None:
            deadline = process.launched_at + timeout_seconds
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                expired = True
            else:
                # A wait can observe an exit after its timeout if it wakes late.
                expired = time.monotonic() > deadline
            if expired:
                process.kill_for_timeout()
                raise TimeoutError(f"Local evaluator exceeded timeout of {timeout}")
    finally:
        process.cleanup_logging()

    return_code = process.returncode
    if verbose:
        logger.info(f"Process {process.pid} completed with return code: {return_code}")
