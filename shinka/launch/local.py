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
        _kill_and_reap(self.process, self._process_group_id)

    def cleanup_logging(self):
        """Clean up logging threads and files."""
        # Wait for logging threads to finish
        for thread in self.log_threads:
            thread.join(timeout=1.0)

        # Close log files
        for file_handle in self.log_files:
            try:
                file_handle.close()
            except Exception as e:
                logger.error(f"Error closing log file: {e}")


def _kill_and_reap(process: subprocess.Popen, group_id: Optional[int]) -> None:
    try:
        if group_id is None:
            process.kill()
        else:
            os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


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
        timeout: Set for evaluations that will be monitored with a timeout.

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

    start_new_session = os.name == "posix" and timeout is not None
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
    """
    Monitors a local subprocess until completion and loads its results.

    Args:
        process: The ProcessWithLogging object to monitor.
        results_dir: The directory where results will be stored.
        poll_interval: Time in seconds between status checks for untimed jobs.
        verbose: Whether to enable verbose logging.
        timeout: Optional timeout in `hh:mm:ss` format.

    Returns:
        dict: Dictionary containing job results.

    Raises:
        TimeoutError: The process was killed after exceeding the timeout.
    """
    if verbose:
        logger.info(f"Monitoring local process with PID: {process.pid}...")

    timeout_seconds = parse_time_to_seconds(timeout) if timeout is not None else None

    try:
        if timeout_seconds is None:
            while process.poll() is None:
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

    return load_results(results_dir)
