import subprocess
import time
import threading
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple, TextIO, Dict
from shinka.utils import load_results, parse_time_to_seconds
import logging

_SUPERVISOR_CODE = """
import os
import signal
import subprocess
import sys

def ignore_termination(_signum, _frame):
    pass


result_fd = int(sys.argv[1])
signal.signal(signal.SIGTERM, ignore_termination)
child = subprocess.Popen(sys.argv[2:])
return_code = child.wait()
os.write(result_fd, f"{return_code}\\n".encode("ascii"))
os.close(result_fd)
while True:
    signal.pause()
"""
logger = logging.getLogger(__name__)


def _move_fd_above_standard_streams(fd: int) -> int:
    """Return a close-on-exec duplicate whose descriptor is at least 3."""
    if fd >= 3:
        os.set_inheritable(fd, False)
        return fd

    import fcntl

    moved_fd = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    os.close(fd)
    return moved_fd


class ProcessWithLogging:
    """Wrapper for subprocess.Popen with real-time logging capabilities."""

    def __init__(
        self,
        process: subprocess.Popen,
        log_files: Tuple[TextIO, TextIO],
        log_threads: Tuple[threading.Thread, threading.Thread],
        process_group: Optional[int] = None,
        completion_fd: Optional[int] = None,
    ):
        self.process = process
        self.log_files = log_files
        self.log_threads = log_threads
        self._process_group = process_group
        self._completion_fd = completion_fd
        self._completion_buffer = b""
        self._returncode: Optional[int] = None

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped process."""
        return getattr(self.process, name)

    def __str__(self):
        """Return a string representation showing the PID."""
        return f"ProcessWithLogging(PID: {self.process.pid})"

    def __repr__(self):
        """Return a detailed string representation."""
        return f"ProcessWithLogging(PID: {self.process.pid}, returncode: {self.returncode})"

    @property
    def returncode(self) -> Optional[int]:
        """Return the evaluator's exit code rather than the supervisor's."""
        if self._returncode is not None:
            return self._returncode
        if self._completion_fd is None:
            return self.process.returncode
        self._read_completion()
        return self._returncode

    def _read_completion(self) -> None:
        """Read an evaluator exit code from the supervisor without blocking."""
        if self._completion_fd is None or self._returncode is not None:
            return

        try:
            chunk = os.read(self._completion_fd, 64)
        except BlockingIOError:
            return

        if chunk:
            self._completion_buffer += chunk
            if b"\n" in self._completion_buffer:
                line, _, _ = self._completion_buffer.partition(b"\n")
                self._returncode = int(line)
                os.close(self._completion_fd)
                self._completion_fd = None
            return

        os.close(self._completion_fd)
        self._completion_fd = None
        supervisor_returncode = self.process.poll()
        if supervisor_returncode is not None:
            # Once the supervisor exits it no longer reserves the PGID. Do not
            # retain an identifier that the kernel may subsequently reuse.
            self._process_group = None
            self._returncode = supervisor_returncode

    def poll(self) -> Optional[int]:
        """Poll the evaluator while a supervisor keeps its process group alive."""
        if self._completion_fd is None and self._returncode is None:
            return self.process.poll()

        self._read_completion()
        if self._returncode is not None:
            return self._returncode

        supervisor_returncode = self.process.poll()
        if supervisor_returncode is not None:
            self._read_completion()
            if self._returncode is None:
                self._process_group = None
                self._returncode = supervisor_returncode
        return self._returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for the evaluator, leaving its supervisor alive for cleanup."""
        if self._completion_fd is None and self._returncode is None:
            return self.process.wait(timeout=timeout)

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            if deadline is not None and time.monotonic() >= deadline:
                assert timeout is not None
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            time.sleep(0.01)

    def _signal_process_group(self, process_group: int, sig: int) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            self._process_group = None
        else:
            # SIGTERM leaves the supervisor alive so callers can escalate. Once
            # SIGKILL releases it, never retain a PGID the kernel may reuse.
            if sig == signal.SIGKILL:
                self._process_group = None

    def kill(self) -> None:
        """Kill the process group on POSIX, or only the direct child elsewhere."""
        process_group = self._process_group
        if process_group is None:
            self.process.kill()
            return
        self._signal_process_group(process_group, signal.SIGKILL)

    def send_signal(self, sig: int) -> None:
        """Signal the owned process group, or the direct child when ungrouped."""
        process_group = self._process_group
        if process_group is None:
            self.process.send_signal(sig)
            return
        self._signal_process_group(process_group, sig)

    def terminate(self) -> None:
        """Terminate the process group on POSIX, or only the direct child elsewhere."""
        process_group = self._process_group
        if process_group is None:
            self.process.terminate()
            return
        self._signal_process_group(process_group, signal.SIGTERM)

    def cleanup_logging(self):
        """Clean up logging threads and files."""
        if self._completion_fd is not None or self._returncode is not None:
            if self.process.poll() is None:
                self.kill()
            else:
                self._process_group = None
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
            self._process_group = None
            if self._completion_fd is not None:
                os.close(self._completion_fd)
                self._completion_fd = None

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

    # A live supervisor keeps the process group allocated after the evaluator
    # exits, so descendants can still be signalled without risking PGID reuse.
    completion_fd = None
    if os.name == "posix":
        completion_fd, completion_write_fd = os.pipe()
        try:
            completion_fd = _move_fd_above_standard_streams(completion_fd)
            completion_write_fd = _move_fd_above_standard_streams(
                completion_write_fd
            )
            os.set_blocking(completion_fd, False)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _SUPERVISOR_CODE,
                    str(completion_write_fd),
                    *cmd,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=env,
                start_new_session=True,
                pass_fds=(completion_write_fd,),
            )
        except Exception:
            os.close(completion_fd)
            raise
        finally:
            os.close(completion_write_fd)
    else:
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
        process,
        (stdout_file, stderr_file),
        (stdout_thread, stderr_thread),
        process_group=process.pid if os.name == "posix" else None,
        completion_fd=completion_fd,
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
