import logging
import subprocess
import threading
import time
import asyncio
import shlex
import sys
from typing import Optional, Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from .local import submit as submit_local, monitor as monitor_local
from .local import monitor_evaluation as monitor_local_evaluation
from .local import ProcessWithLogging, kill_evaluators_on_fatal_signals
from .job_config import (
    JobConfig,
    LocalJobConfig,
    SlurmCondaJobConfig,
    SlurmDockerJobConfig,
    SlurmEnvJobConfig,
    _has_value,
)
from .slurm_evaluation import monitor_evaluation as monitor_slurm_evaluation
from .slurm_evaluation import CONTROL_COMMAND_TIMEOUT_SECONDS
from .slurm import (
    submit_docker as submit_slurm_docker,
    submit_conda as submit_slurm_conda,
    monitor as monitor_slurm,
)
from .slurm_status import SlurmQueuePoller, is_unknown_job_error
from .evaluation_runtimes import LongestRuntimes, reports_correct
from shinka.utils import parse_time_to_seconds

logger = logging.getLogger(__name__)
SLURM_CANCEL_TIMEOUT_SECONDS = 30
SLURM_JOB_TYPES = ("slurm_docker", "slurm_conda")
# A job whose results are being collected has left the queue or was just
# cancelled. Give a cancelled one this long to leave before reading its files.
SLURM_RESULTS_SETTLE_SECONDS = 20.0
SLURM_RESULTS_POLL_SECONDS = 2.0


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
        self._slurm_queue = SlurmQueuePoller()
        self._state_lock = threading.Lock()
        self._runtimes = LongestRuntimes()
        # Evaluations `run` is blocked on, so that `shutdown` can release it.
        self._seed_jobs: List[Union[str, ProcessWithLogging]] = []
        self._closed = False
        if self.job_type == "slurm_env":
            self.job_type = "slurm_conda"

        if self.job_type == "local":
            self.monitor = monitor_local
            kill_evaluators_on_fatal_signals()
        elif self.job_type in SLURM_JOB_TYPES:
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

    @property
    def longest_successful_runtime(self) -> Optional[float]:
        """Seconds the longest local evaluation with a correct result took."""
        return self._runtimes.successful

    @property
    def longest_finished_runtime(self) -> Optional[float]:
        """Seconds the longest local evaluation took, whatever its outcome."""
        return self._runtimes.finished

    def note_successful_runtime(self, seconds: float) -> None:
        """Record how long a successful evaluation took, e.g. in an earlier run."""
        self._runtimes.note_successful(seconds)

    def run(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Tuple[Dict[str, Any], float]:
        if self._closed:
            raise RuntimeError("JobScheduler is shut down and runs no more evaluations")
        job_id: Union[str, ProcessWithLogging]
        cmd = self._build_command(exec_fname_t, results_dir_t)
        start_time = time.time()
        timeout = None
        job_name = f"shinka-seed-{uuid4().hex}"

        if self.job_type == "local":
            assert isinstance(self.config, LocalJobConfig)
            timeout = self.config.time or None
            if timeout is not None:
                # Reject an invalid timeout before starting the subprocess.
                parse_time_to_seconds(timeout)
            job_id = submit_local(
                results_dir_t,
                cmd,
                verbose=self.verbose,
                env_overrides=self._build_local_env_overrides(),
                timeout=timeout,
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
                command_timeout=CONTROL_COMMAND_TIMEOUT_SECONDS,
                job_name=job_name,
                seed_results_dir=results_dir_t,
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
                command_timeout=CONTROL_COMMAND_TIMEOUT_SECONDS,
                job_name=job_name,
                seed_results_dir=results_dir_t,
            )
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

        with self._state_lock:
            self._seed_jobs.append(job_id)
            closed = self._closed
        try:
            if closed:
                # Shut down while the evaluation was being launched.
                self._cancel_job(job_id)
            if isinstance(job_id, str):
                results = monitor_slurm_evaluation(
                    job_id, results_dir_t, job_name=job_name
                )
            else:
                results = monitor_local_evaluation(
                    job_id, results_dir_t, timeout=timeout
                )
        finally:
            with self._state_lock:
                self._seed_jobs.remove(job_id)
            if isinstance(job_id, ProcessWithLogging):
                # Also when the seed crashed or timed out.
                self._runtimes.note_finished(job_id.runtime_seconds())

        end_time = time.time()
        rtime = end_time - start_time

        # Ensure results is not None
        if results is None:
            results = {"correct": {"correct": False}, "metrics": {}}
        if self.job_type == "local" and reports_correct(results):
            self._runtimes.note_successful(rtime)

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

    def _check_slurm_jobs(self, jobs: List) -> List[bool]:
        """Whether each job is still queued or running, from one queue query."""
        job_ids = [job.job_id for job in jobs if isinstance(job.job_id, str)]
        active = self._slurm_queue.active_jobs(job_ids) if job_ids else set()
        return [isinstance(job.job_id, str) and job.job_id in active for job in jobs]

    def check_job_status(self, job) -> bool:
        """Check if job is running. Returns True if running, False if done."""
        if self.job_type in SLURM_JOB_TYPES:
            return self._check_slurm_jobs([job])[0]
        else:
            if isinstance(job.job_id, ProcessWithLogging):
                if (
                    isinstance(self.config, LocalJobConfig)
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
                        job.job_id.kill_group()
                        return False

                # More robust status checking with exception handling
                try:
                    return not job.job_id.has_exited()
                except Exception as e:
                    # If poll() fails, try alternative methods to determine if process is running
                    logger.warning(f"poll() failed for PID {job.job_id.pid}: {e}")
                    try:
                        # Try using psutil as fallback if available
                        import psutil

                        return psutil.pid_exists(job.job_id.pid)
                    except ImportError:
                        # Fallback: check if PID exists using os.kill with signal 0
                        try:
                            import os

                            os.kill(job.job_id.pid, 0)
                            return True  # Process exists
                        except (OSError, ProcessLookupError):
                            return False  # Process doesn't exist
                    except Exception as e2:
                        logger.warning(
                            f"All status check methods failed for PID {job.job_id.pid}: {e2}"
                        )
                        # If all methods fail, assume process is dead
                        return False
            return False

    def get_job_results(
        self, job_id: Union[str, ProcessWithLogging], results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Get results from a completed job."""
        if self.job_type in SLURM_JOB_TYPES:
            if isinstance(job_id, str):
                return monitor_slurm(
                    job_id,
                    results_dir,
                    poll_interval=SLURM_RESULTS_POLL_SECONDS,
                    verbose=self.verbose,
                    max_wait=SLURM_RESULTS_SETTLE_SECONDS,
                )
        else:
            if isinstance(job_id, ProcessWithLogging):
                job_id.wait()
                results = monitor_local(
                    job_id,
                    results_dir,
                    verbose=self.verbose,
                    timeout=self.config.time or None,
                )
                job_id.has_exited()
                seconds = job_id.runtime_seconds()
                self._runtimes.note_finished(seconds)
                if not job_id.was_killed and reports_correct(results):
                    self._runtimes.note_successful(seconds)
                return results
        return None

    async def submit_async_nonblocking(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Union[str, ProcessWithLogging]:
        """Submit a job asynchronously without blocking the event loop."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            self.executor, self.submit_async, exec_fname_t, results_dir_t
        )

    async def check_job_status_async(self, job) -> bool:
        """Async version of job status checking."""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(self.executor, self.check_job_status, job)

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
        if self.job_type in SLURM_JOB_TYPES:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor, self._check_slurm_jobs, jobs
                )
            except Exception as e:
                # Reported per job, like a failed individual check.
                return [e] * len(jobs)
        tasks = [self.check_job_status_async(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _cancel_job(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Cancel a job. Blocks, so the event loop calls it through the executor."""
        try:
            if self.job_type in SLURM_JOB_TYPES:
                if isinstance(job_id, str):
                    result = subprocess.run(
                        ["scancel", job_id],
                        capture_output=True,
                        text=True,
                        timeout=SLURM_CANCEL_TIMEOUT_SECONDS,
                    )
                    # A job the controller no longer knows is already gone.
                    return result.returncode == 0 or is_unknown_job_error(
                        result.stderr
                    )
            elif isinstance(job_id, ProcessWithLogging):
                # Kill the evaluator with everything it started.
                job_id.kill_group()
                job_id.cleanup_logging()
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    async def cancel_job_async(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Cancel a running job asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._cancel_job, job_id)

    def shutdown(self):
        """Cancel evaluations `run` still waits on, then stop the executor."""
        with self._state_lock:
            self._closed = True
            seed_jobs = list(self._seed_jobs)
        for job_id in seed_jobs:
            self._cancel_job(job_id)
        self.executor.shutdown(wait=True)
