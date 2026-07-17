import logging
import subprocess
import time
import asyncio
import shlex
import sys
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from .local import submit as submit_local, monitor as monitor_local
from .local import ProcessWithLogging
from .slurm import (
    get_job_status,
    submit_docker as submit_slurm_docker,
    submit_conda as submit_slurm_conda,
    monitor as monitor_slurm,
)
from shinka.utils import parse_time_to_seconds

logger = logging.getLogger(__name__)

_SLURM_CANCEL_TIMEOUT_SECONDS = 5.0
_SLURM_CANCEL_POLL_INTERVAL_SECONDS = 0.1
_SLURM_COMMAND_TIMEOUT_SECONDS = 5.0
_SHUTDOWN_CLEANUP_RETRY_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_CLEANUP_RETRY_INITIAL_DELAY_SECONDS = 0.05
_SHUTDOWN_CLEANUP_RETRY_MAX_DELAY_SECONDS = 1.0


class SchedulerSubmissionCleanupError(RuntimeError):
    """Report whether a scheduler-owned shutdown cleanup reached a terminal state."""

    def __init__(
        self,
        token: object,
        job_id: Union[str, ProcessWithLogging],
        cleanup_succeeded: bool,
    ):
        self.token = token
        self.job_id = job_id
        self.cleanup_succeeded = cleanup_succeeded
        state = "completed" if cleanup_succeeded else "remains owned"
        super().__init__(f"Evaluation submission cleanup {state} during shutdown")


def _has_value(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


def _validate_activation_config(
    conda_env: Optional[str],
    activate_script: Optional[str],
) -> None:
    if _has_value(conda_env) and _has_value(activate_script):
        raise ValueError("conda_env and activate_script are mutually exclusive")


def _numeric_thread_env(numeric_threads: Optional[int]) -> Dict[str, str]:
    """Numeric-library thread-cap env vars for a given per-process thread limit.

    Returns an empty dict when ``numeric_threads`` is None (no capping).
    """
    if numeric_threads is None:
        return {}
    thread_value = str(max(1, int(numeric_threads)))
    return {
        "OMP_NUM_THREADS": thread_value,
        "OMP_THREAD_LIMIT": thread_value,
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": thread_value,
        "MKL_NUM_THREADS": thread_value,
        "MKL_DYNAMIC": "FALSE",
        "NUMEXPR_NUM_THREADS": thread_value,
        "NUMEXPR_MAX_THREADS": thread_value,
        "VECLIB_MAXIMUM_THREADS": thread_value,
        "BLIS_NUM_THREADS": thread_value,
        "GOTO_NUM_THREADS": thread_value,
    }


@dataclass
class JobConfig:
    """Base job configuration"""

    eval_program_path: Optional[str] = "evaluate.py"
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)
    eval_verbose: bool = True  # emit per-run eval banners to stdout
    # Cap numeric-library threads (OMP/BLAS/MKL/...) per eval subprocess.
    # None = leave unset. Applied for both local and SLURM jobs.
    numeric_threads_per_job: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        job_to_dict = asdict(self)
        return {k: v for k, v in job_to_dict.items() if v is not None}


@dataclass
class LocalJobConfig(JobConfig):
    """Configuration for local jobs"""

    time: Optional[str] = None
    conda_env: Optional[str] = None
    activate_script: Optional[str] = None
    python_executable: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_activation_config(self.conda_env, self.activate_script)


@dataclass
class SlurmDockerJobConfig(JobConfig):
    """Configuration for SLURM jobs using Docker"""

    image: str = "ubuntu:latest"
    image_tar_path: Optional[str] = None
    docker_flags: str = ""
    partition: str = "gpu"
    time: str = "01:00:00"
    cpus: int = 1
    gpus: int = 1
    mem: Optional[str] = "8G"


@dataclass
class SlurmCondaJobConfig(JobConfig):
    """Configuration for SLURM jobs using Conda environment"""

    conda_env: str = ""
    activate_script: Optional[str] = None
    modules: Optional[List[str]] = None
    partition: str = "gpu"
    time: str = "01:00:00"
    cpus: int = 1
    gpus: int = 1
    mem: Optional[str] = "8G"

    def __post_init__(self):
        _validate_activation_config(self.conda_env, self.activate_script)
        if self.modules is None:
            self.modules = []


SlurmEnvJobConfig = SlurmCondaJobConfig


class JobScheduler:
    def __init__(
        self,
        job_type: str,
        config: Union[
            LocalJobConfig,
            SlurmDockerJobConfig,
            SlurmCondaJobConfig,
        ],
        verbose: bool = False,
        max_workers: int = 4,
    ):
        self.job_type = job_type
        self.config = config
        self.verbose = verbose
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._submission_lock = threading.Lock()
        self._shutdown_requested = False
        self._unclaimed_submissions: Dict[
            object, Union[str, ProcessWithLogging]
        ] = {}
        self._shutdown_submissions: Dict[object, Union[str, ProcessWithLogging]] = {}
        self._shutdown_cleanup_futures: Dict[object, Any] = {}
        if self.job_type == "slurm_env":
            self.job_type = "slurm_conda"

        if self.job_type == "local":
            self.monitor = monitor_local
        elif self.job_type in ["slurm_docker", "slurm_conda"]:
            self.monitor = monitor_slurm
        else:
            raise ValueError(
                f"Unknown job type: {job_type}. "
                f"Must be 'local', 'slurm_docker', 'slurm_conda', or 'slurm_env'"
            )

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        python_executable = "python"
        if self.job_type == "local" and isinstance(self.config, LocalJobConfig):
            if not (
                _has_value(self.config.conda_env)
                or _has_value(self.config.activate_script)
            ):
                python_executable = (
                    self.config.python_executable or sys.executable or "python"
                )

        if self.job_type == "slurm_docker":
            assert isinstance(self.config, SlurmDockerJobConfig)
            python_cmd = [
                "python",
                f"/workspace/{self.config.eval_program_path}",
                "--program_path",
                f"/workspace/{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        else:
            python_cmd = [
                python_executable,
                f"{self.config.eval_program_path}",
                "--program_path",
                f"{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                python_cmd.extend([f"--{k}", str(v)])

        if self.job_type == "local" and isinstance(self.config, LocalJobConfig):
            if _has_value(self.config.conda_env):
                return [
                    "conda",
                    "run",
                    "-n",
                    self.config.conda_env.strip(),
                    *python_cmd,
                ]
            if _has_value(self.config.activate_script):
                activate_script = self.config.activate_script.strip().replace('"', '\\"')
                return [
                    "bash",
                    "-lc",
                    f'set -e; source "{activate_script}"; exec {shlex.join(python_cmd)}',
                ]

        return python_cmd

    def _build_eval_env(self) -> Dict[str, str]:
        """Environment for the eval subprocess: verbosity + numeric-thread caps.

        Shared by both the local path (Popen env overrides) and the SLURM path
        (exported inside the batch script), so behaviour is identical across
        job types.
        """
        env: Dict[str, str] = {
            "SHINKA_EVAL_VERBOSE": "1" if self.config.eval_verbose else "0",
        }
        env.update(_numeric_thread_env(self.config.numeric_threads_per_job))
        return env

    def _build_local_env_overrides(self) -> Optional[Dict[str, str]]:
        """Build environment overrides for local evaluation subprocesses."""
        if self.job_type != "local" or not isinstance(self.config, LocalJobConfig):
            return None
        return self._build_eval_env()

    def run(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Tuple[Dict[str, Any], float]:
        job_id: Union[str, ProcessWithLogging]
        cmd = self._build_command(exec_fname_t, results_dir_t)
        start_time = time.time()

        if self.job_type == "local":
            assert isinstance(self.config, LocalJobConfig)
            job_id = submit_local(
                results_dir_t,
                cmd,
                verbose=self.verbose,
                env_overrides=self._build_local_env_overrides(),
            )
        elif self.job_type == "slurm_docker":
            assert isinstance(self.config, SlurmDockerJobConfig)
            job_id = submit_slurm_docker(
                results_dir_t,
                cmd,
                self.config.time,
                self.config.partition,
                self.config.cpus,
                self.config.gpus,
                self.config.mem,
                self.config.docker_flags,
                self.config.image,
                image_tar_path=self.config.image_tar_path,
                verbose=self.verbose,
                eval_env=self._build_eval_env(),
            )
        elif self.job_type == "slurm_conda":
            assert isinstance(self.config, SlurmCondaJobConfig)
            job_id = submit_slurm_conda(
                results_dir_t,
                cmd,
                self.config.time,
                self.config.partition,
                self.config.cpus,
                self.config.gpus,
                self.config.mem,
                self.config.conda_env,
                self.config.activate_script,
                self.config.modules,
                verbose=self.verbose,
                eval_env=self._build_eval_env(),
            )
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

        if isinstance(job_id, str):
            results = monitor_slurm(job_id, results_dir_t)
        else:
            results = monitor_local(job_id, results_dir_t)

        end_time = time.time()
        rtime = end_time - start_time

        # Ensure results is not None
        if results is None:
            results = {"correct": {"correct": False}, "metrics": {}}

        return results, rtime

    def submit_async(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Union[str, ProcessWithLogging]:
        """Submit a job asynchronously and return the job ID or process."""
        cmd = self._build_command(exec_fname_t, results_dir_t)
        if self.job_type == "local":
            assert isinstance(self.config, LocalJobConfig)
            return submit_local(
                results_dir_t,
                cmd,
                verbose=self.verbose,
                env_overrides=self._build_local_env_overrides(),
            )
        elif self.job_type == "slurm_docker":
            assert isinstance(self.config, SlurmDockerJobConfig)
            return submit_slurm_docker(
                results_dir_t,
                cmd,
                self.config.time,
                self.config.partition,
                self.config.cpus,
                self.config.gpus,
                self.config.mem,
                self.config.docker_flags,
                self.config.image,
                image_tar_path=self.config.image_tar_path,
                verbose=self.verbose,
                eval_env=self._build_eval_env(),
            )
        elif self.job_type == "slurm_conda":
            assert isinstance(self.config, SlurmCondaJobConfig)
            return submit_slurm_conda(
                results_dir_t,
                cmd,
                self.config.time,
                self.config.partition,
                self.config.cpus,
                self.config.gpus,
                self.config.mem,
                self.config.conda_env,
                self.config.activate_script,
                self.config.modules,
                verbose=self.verbose,
                eval_env=self._build_eval_env(),
            )
        raise ValueError(f"Unknown job type: {self.job_type}")

    def check_job_id_status(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Check whether a raw scheduler job ID is still running."""
        if self.job_type in ["slurm_docker", "slurm_conda"]:
            if isinstance(job_id, str):
                return get_job_status(job_id) != ""
            return False

        if not isinstance(job_id, ProcessWithLogging):
            return False

        try:
            return job_id.poll() is None
        except Exception as e:
            logger.warning(f"poll() failed for PID {job_id.pid}: {e}")
            try:
                import psutil

                return psutil.pid_exists(job_id.pid)
            except ImportError:
                try:
                    import os

                    os.kill(job_id.pid, 0)
                    return True
                except (OSError, ProcessLookupError):
                    return False
            except Exception as e2:
                logger.warning(
                    f"All status check methods failed for PID {job_id.pid}: {e2}"
                )
                return False

    def check_job_status(self, job) -> bool:
        """Check if job is running. Returns True if running, False if done."""
        if (
            self.job_type == "local"
            and isinstance(job.job_id, ProcessWithLogging)
            and isinstance(self.config, LocalJobConfig)
            and self.config.time
            and job.start_time
        ):
            timeout = parse_time_to_seconds(self.config.time)
            if time.time() - job.start_time > timeout:
                if self.verbose:
                    logger.warning(
                        f"Process {job.job_id.pid} exceeded "
                        f"timeout of {self.config.time}. Killing. "
                        f"=> Gen. {job.generation}"
                    )
                job.job_id.kill()
                return False

        return self.check_job_id_status(job.job_id)

    def get_job_results(
        self, job_id: Union[str, ProcessWithLogging], results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Get results from a completed job."""
        if self.job_type in ["slurm_docker", "slurm_conda"]:
            if isinstance(job_id, str):
                return monitor_slurm(job_id, results_dir, verbose=self.verbose)
        else:
            if isinstance(job_id, ProcessWithLogging):
                job_id.wait()
                return monitor_local(
                    job_id,
                    results_dir,
                    verbose=self.verbose,
                    timeout=self.config.time,
                )
        return None

    def _cleanup_unclaimed_submission(
        self, token: object, job_id: Union[str, ProcessWithLogging]
    ) -> bool:
        """Retry cleanup of a submitted job that shutdown claimed before the runner."""
        deadline = time.monotonic() + getattr(
            self,
            "_shutdown_cleanup_retry_timeout_seconds",
            _SHUTDOWN_CLEANUP_RETRY_TIMEOUT_SECONDS,
        )
        retry_delay = _SHUTDOWN_CLEANUP_RETRY_INITIAL_DELAY_SECONDS
        while True:
            if self._cancel_job_sync(job_id):
                with self._submission_lock:
                    self._shutdown_submissions.pop(token, None)
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    f"Could not clean up evaluation job {job_id} submitted during "
                    "scheduler shutdown; retaining ownership"
                )
                return False
            time.sleep(min(retry_delay, remaining))
            retry_delay = min(
                _SHUTDOWN_CLEANUP_RETRY_MAX_DELAY_SECONDS, retry_delay * 2
            )

    def _submit_tracked(
        self,
        token: object,
        exec_fname_t: str,
        results_dir_t: str,
    ) -> Union[str, ProcessWithLogging]:
        """Submit while retaining ownership until the event loop claims the job."""
        with self._submission_lock:
            if self._shutdown_requested:
                raise RuntimeError("Scheduler shutdown has started")

        job_id = self.submit_async(exec_fname_t, results_dir_t)
        with self._submission_lock:
            if self._shutdown_requested:
                self._shutdown_submissions[token] = job_id
                cleanup_here = True
            else:
                self._unclaimed_submissions[token] = job_id
                cleanup_here = False

        if cleanup_here:
            cleaned = self._cleanup_unclaimed_submission(token, job_id)
            raise SchedulerSubmissionCleanupError(token, job_id, cleaned)
        return job_id

    async def submit_async_nonblocking(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Union[str, ProcessWithLogging]:
        """Submit a job asynchronously without blocking the event loop."""
        loop = asyncio.get_event_loop()
        token = object()
        job_id = await loop.run_in_executor(
            self.executor,
            self._submit_tracked,
            token,
            exec_fname_t,
            results_dir_t,
        )

        with self._submission_lock:
            if token in self._unclaimed_submissions:
                self._unclaimed_submissions.pop(token)
                return job_id
            shutdown_job = self._shutdown_submissions.get(token)
            cleanup_future = self._shutdown_cleanup_futures.get(token)

        if shutdown_job is None:
            raise RuntimeError("Evaluation submission was lost during scheduler shutdown")

        if cleanup_future is None:
            raise SchedulerSubmissionCleanupError(token, shutdown_job, False)

        cleaned = await asyncio.wrap_future(cleanup_future)
        with self._submission_lock:
            self._shutdown_cleanup_futures.pop(token, None)
        raise SchedulerSubmissionCleanupError(token, shutdown_job, cleaned)

    def begin_shutdown(self) -> None:
        """Stop submission handoffs and asynchronously clean jobs not yet claimed."""
        with self._submission_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            unclaimed_jobs = list(self._unclaimed_submissions.items())
            self._unclaimed_submissions.clear()
            self._shutdown_submissions.update(unclaimed_jobs)
            for token, job_id in unclaimed_jobs:
                self._shutdown_cleanup_futures[token] = self.executor.submit(
                    self._cleanup_unclaimed_submission, token, job_id
                )

    async def check_job_status_async(self, job) -> bool:
        """Async version of job status checking."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(self.executor, self.check_job_status, job)

    async def check_job_id_status_async(
        self, job_id: Union[str, ProcessWithLogging]
    ) -> bool:
        """Async version of raw scheduler job-ID status checking."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            self.executor, self.check_job_id_status, job_id
        )

    async def get_job_results_async(
        self, job_id: Union[str, ProcessWithLogging], results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Async version of getting job results."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            self.executor, self.get_job_results, job_id, results_dir
        )

    async def batch_check_status_async(self, jobs: List) -> List[bool]:
        """Check status of multiple jobs concurrently."""
        tasks = [self.check_job_status_async(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _cancel_job_sync(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Cancel a running job from a scheduler worker."""
        try:
            if self.job_type in ["slurm_docker", "slurm_conda"]:
                if isinstance(job_id, str):
                    # For SLURM jobs, use scancel command
                    deadline = time.monotonic() + _SLURM_CANCEL_TIMEOUT_SECONDS
                    remaining = deadline - time.monotonic()
                    result = subprocess.run(
                        ["scancel", job_id],
                        capture_output=True,
                        text=True,
                        timeout=max(
                            0.001,
                            min(_SLURM_COMMAND_TIMEOUT_SECONDS, remaining),
                        ),
                    )
                    if result.returncode != 0:
                        return False

                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            logger.warning(
                                f"Timed out waiting for Slurm job {job_id} "
                                "to leave the queue after scancel"
                            )
                            return False
                        if (
                            get_job_status(
                                job_id,
                                timeout=max(
                                    0.001,
                                    min(
                                        _SLURM_COMMAND_TIMEOUT_SECONDS,
                                        remaining,
                                    ),
                                ),
                            )
                            == ""
                        ):
                            return True
                        time.sleep(
                            min(_SLURM_CANCEL_POLL_INTERVAL_SECONDS, remaining)
                        )
            else:
                # For local jobs, kill the process
                if isinstance(job_id, ProcessWithLogging):
                    job_id.kill()
                    try:
                        job_id.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"Timed out waiting for local job {job_id} "
                            "to exit after cancellation"
                        )
                        return False
                    job_id.cleanup_logging()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    async def cancel_job_async(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Cancel a running job asynchronously."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(self.executor, self._cancel_job_sync, job_id)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.begin_shutdown()
        self.executor.shutdown(wait=True)

    def shutdown_nowait(self):
        """Stop accepting work without waiting for in-flight scheduler commands."""
        self.begin_shutdown()
        self.executor.shutdown(wait=False, cancel_futures=False)
