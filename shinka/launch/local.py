import subprocess
import time
import threading
import os
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
    ):
        self.process = process
        self.log_files = log_files
        self.log_threads = log_threads

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped process."""
        return getattr(self.process, name)

    def __str__(self):
        """Return a string representation showing the PID."""
        return f"ProcessWithLogging(PID: {self.process.pid})"

    def __repr__(self):
        """Return a detailed string representation."""
        return f"ProcessWithLogging(PID: {self.process.pid}, returncode: {self.process.returncode})"

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


def submit(
    log_dir: str,
    cmd: list[str],
    verbose: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
):
    """
    Submits a command for local execution with real-time logging.

    Args:
        log_dir: The directory to store logs.
        cmd: The command and its arguments as a list of strings.
        verbose: Whether to enable verbose logging.

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

    # Use PIPE to capture output and redirect to files in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        env=env,
    )

    # Open log files for writing with line buffering
    stdout_file = open(stdout_path, "w", buffering=1)
    stderr_file = open(stderr_path, "w", buffering=1)

    # Start threads to stream output to files in real-time
    stdout_thread = threading.Thread(
        target=_stream_output,
        args=(process.stdout, stdout_file, "STDOUT" if verbose else None),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_output,
        args=(process.stderr, stderr_file, "STDERR" if verbose else None),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    # Create wrapper with logging capabilities
    wrapped_process = ProcessWithLogging(
        process, (stdout_file, stderr_file), (stdout_thread, stderr_thread)
    )

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
        poll_interval: Time in seconds between status checks.
        verbose: Whether to enable verbose logging.
        timeout: Optional timeout in `hh:mm:ss` format.

    Returns:
        dict: Dictionary containing job results.
    """
    if verbose:
        logger.info(f"Monitoring local process with PID: {process.pid}...")

    start_time = time.time()
    timeout_seconds = parse_time_to_seconds(timeout) if timeout is not None else None

    while process.poll() is None:
        if timeout_seconds and (time.time() - start_time) > timeout_seconds:
            if verbose:
                logger.info(
                    f"Process {process.pid} exceeded timeout of {timeout}. Killing."
                )
            process.kill()
            break

        if verbose:
            logger.info(f"Process {process.pid} is still running...")
        time.sleep(poll_interval)

    # Clean up logging resources
    process.cleanup_logging()

    return_code = process.returncode
    if verbose:
        logger.info(f"Process {process.pid} completed with return code: {return_code}")

    return load_results(results_dir)
