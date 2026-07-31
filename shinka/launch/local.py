import subprocess
import time
import threading
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TextIO, Dict
import psutil
from shinka.utils import load_results, parse_time_to_seconds
import logging

logger = logging.getLogger(__name__)
LOCAL_PROCESS_TOKEN_ENV = "SHINKA_LOCAL_JOB_TOKEN"
LOCAL_PROCESS_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")


def _process_group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


def create_local_process_token() -> str:
    """Create the durable token persisted before a local job launch."""
    return uuid.uuid4().hex


def find_local_process_identities(
    token: str,
) -> list["LocalProcessIdentity"]:
    """Find live process groups carrying one preallocated local-job token."""
    if LOCAL_PROCESS_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError("Invalid local process token")
    if not hasattr(os, "getpgid"):
        raise RuntimeError("Local process recovery requires process groups")

    process_group_ids = set()
    current_user_id = os.geteuid()
    same_user_inspection_blocked = False
    for process in psutil.process_iter(["pid"]):
        try:
            if process.uids().effective != current_user_id:
                continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, PermissionError, OSError) as error:
            raise RuntimeError(
                "Could not inspect process owners during local recovery"
            ) from error

        try:
            if process.environ().get(LOCAL_PROCESS_TOKEN_ENV) != token:
                continue
            process_group_ids.add(os.getpgid(process.pid))
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError, OSError):
            same_user_inspection_blocked = True

    if same_user_inspection_blocked:
        raise RuntimeError(
            "Could not inspect same-user processes during local recovery"
        )

    return [
        LocalProcessIdentity(pid=process_group_id, token=token)
        for process_group_id in sorted(process_group_ids)
    ]


@dataclass(frozen=True)
class LocalProcessIdentity:
    """Durable identity for one local evaluation process group."""

    pid: int
    token: str

    @classmethod
    def from_storage(cls, job_id: object, job_name: object) -> "LocalProcessIdentity":
        if not isinstance(job_id, str) or not job_id.isdecimal():
            raise ValueError("Invalid local process ID")
        if (
            not isinstance(job_name, str)
            or LOCAL_PROCESS_TOKEN_PATTERN.fullmatch(job_name) is None
        ):
            raise ValueError("Invalid local process token")
        pid = int(job_id)
        if pid <= 0:
            raise ValueError("Invalid local process ID")
        return cls(pid=pid, token=job_name)

    def to_storage(self) -> tuple[str, str]:
        return str(self.pid), self.token

    def _process_group_members(
        self,
    ) -> tuple[list[psutil.Process], bool, bool]:
        """Return live group members, token presence, and inspection ambiguity."""
        if not hasattr(os, "getpgid"):
            return [], False, True

        tokenless_members = []
        token_members = []
        inspection_blocked = False
        for process in psutil.process_iter(["pid"]):
            try:
                process_group_id = os.getpgid(process.pid)
            except ProcessLookupError:
                continue
            except (PermissionError, OSError):
                inspection_blocked = True
                continue

            if process_group_id != self.pid:
                continue

            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                process_token = process.environ().get(LOCAL_PROCESS_TOKEN_ENV)
                if process_token == self.token:
                    token_members.append(process)
                else:
                    tokenless_members.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, PermissionError, OSError):
                inspection_blocked = True

        tokenless_members.sort(key=lambda process: process.pid == self.pid)
        token_members.sort(key=lambda process: process.pid == self.pid)
        group_members = [*tokenless_members, *token_members]
        return group_members, bool(token_members), inspection_blocked

    def kill(self) -> bool:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            group_members, token_present, inspection_blocked = (
                self._process_group_members()
            )
            if inspection_blocked:
                return False
            if not group_members:
                return True
            if not token_present:
                return False

            # Authenticate every snapshot, then kill tokenless descendants
            # before token-bearing members so the proof survives until last.
            for process in group_members:
                try:
                    # psutil rechecks the process creation time immediately
                    # before signaling and refuses a PID already known as reused.
                    process.kill()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (psutil.AccessDenied, OSError):
                    return False
            time.sleep(0.01)
        return self.is_terminated()

    def is_terminated(self) -> bool:
        group_members, _token_present, inspection_blocked = (
            self._process_group_members()
        )
        return not group_members and not inspection_blocked

    def __str__(self) -> str:
        return f"LocalProcessIdentity(PID: {self.pid})"


class AmbiguousLocalSubmissionError(RuntimeError):
    """Raised when a post-launch failure cannot confirm process termination."""

    def __init__(self, process: "ProcessWithLogging"):
        self.cancel_target = process
        super().__init__(
            f"Local submission outcome remains ambiguous for PID {process.pid}"
        )


class ProcessWithLogging:
    """Wrapper for subprocess.Popen with real-time logging capabilities."""

    def __init__(
        self,
        process: subprocess.Popen,
        log_files: Tuple[TextIO, TextIO],
        log_threads: Tuple[threading.Thread, ...],
        identity: Optional[LocalProcessIdentity] = None,
    ):
        self.process = process
        self.log_files = log_files
        self.log_threads = log_threads
        self.identity = identity

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped process."""
        return getattr(self.process, name)

    def kill(self) -> bool:
        """Kill token-verified processes belonging to this local evaluation.

        Eval commands are often wrappers (``conda run -n env python …`` or
        ``bash -lc "… docker run …"``) that fork the real worker. The durable
        identity authenticates each group snapshot before signaling its stable
        process handles, narrowing PID/PGID reuse races while still terminating
        inherited-token children.
        """
        pid = self.process.pid
        if self.identity is None:
            if self.is_terminated():
                return True
            logger.error(f"Refusing to kill process {pid} without durable identity")
            return False
        if not self.identity.kill():
            logger.error(f"Could not verify ownership while killing process {pid}")
            return False

        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            logger.error(f"Process {pid} did not exit after SIGKILL")
            return False
        except Exception as e:
            logger.error(f"Could not confirm process {pid} termination: {e}")
            return False

        return self.is_terminated()

    def is_terminated(self) -> bool:
        """Return whether both the leader and its process group are gone."""
        if self.identity is None:
            return self.process.poll() is not None and not self._process_group_exists()
        return self.process.poll() is not None and self.identity.is_terminated()

    def _process_group_exists(self) -> bool:
        return _process_group_exists(self.process.pid)

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
            try:
                thread.join(timeout=1.0)
            except RuntimeError:
                # A post-launch setup failure may leave a thread unstarted.
                pass

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
    """Submit a local command with a newly allocated durable token."""
    return submit_with_token(
        log_dir,
        cmd,
        create_local_process_token(),
        verbose=verbose,
        env_overrides=env_overrides,
    )


def submit_with_token(
    log_dir: str,
    cmd: list[str],
    local_job_token: str,
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
    if LOCAL_PROCESS_TOKEN_PATTERN.fullmatch(local_job_token) is None:
        raise ValueError("Invalid local process token")
    env[LOCAL_PROCESS_TOKEN_ENV] = local_job_token

    stdout_file = None
    stderr_file = None
    process = None
    wrapped_process = None
    try:
        # Open logs before launch so filesystem failures cannot orphan a child.
        stdout_file = open(stdout_path, "w", buffering=1)
        stderr_file = open(stderr_path, "w", buffering=1)

        # start_new_session=True puts the child in its own process group.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            start_new_session=True,
        )
        wrapped_process = ProcessWithLogging(
            process,
            (stdout_file, stderr_file),
            (),
            identity=LocalProcessIdentity(process.pid, local_job_token),
        )

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
        wrapped_process.log_threads = (stdout_thread, stderr_thread)
        stdout_thread.start()
        stderr_thread.start()
    except Exception:
        if wrapped_process is not None:
            if not wrapped_process.kill():
                raise AmbiguousLocalSubmissionError(wrapped_process) from None
            wrapped_process.cleanup_logging()
        else:
            if stdout_file is not None:
                stdout_file.close()
            if stderr_file is not None:
                stderr_file.close()
        raise

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
            if not process.kill():
                raise RuntimeError(
                    f"Could not confirm termination of process group {process.pid}"
                )
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
