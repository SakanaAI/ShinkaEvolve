import logging
import time
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from .local import submit as submit_local, monitor as monitor_local
from .local import ProcessWithLogging
from .slurm import (
    submit_docker as submit_slurm_docker,
    submit_conda as submit_slurm_conda,
    monitor as monitor_slurm,
)
from shinka.utils import parse_time_to_seconds

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Base job configuration"""

    eval_program_path: Optional[str] = "evaluate.py"
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        job_to_dict = asdict(self)
        return {k: v for k, v in job_to_dict.items() if v is not None}


@dataclass
class LocalJobConfig(JobConfig):
    """Configuration for local jobs"""

    time: Optional[str] = None
    conda_env: Optional[str] = None


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
    modules: Optional[List[str]] = None
    partition: str = "gpu"
    time: str = "01:00:00"
    cpus: int = 1
    gpus: int = 1
    mem: Optional[str] = "8G"

    def __post_init__(self):
        if self.modules is None:
            self.modules = []


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

        if self.job_type == "local":
            self.monitor = monitor_local
        elif self.job_type in ["slurm_docker", "slurm_conda"]:
            self.monitor = monitor_slurm
        else:
            raise ValueError(
                f"Unknown job type: {job_type}. "
                f"Must be 'local', 'slurm_docker', or 'slurm_conda'"
            )

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        # Docker requires workspace to be mounted
        if self.job_type == "slurm_docker":
            assert isinstance(self.config, SlurmDockerJobConfig)
            cmd = [
                "python",
                f"/workspace/{self.config.eval_program_path}",
                "--program_path",
                f"/workspace/{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        else:
            # For local jobs, check if conda environment is specified
            if (
                self.job_type == "local"
                and isinstance(self.config, LocalJobConfig)
                and self.config.conda_env
            ):
                # Use conda run to execute in specific environment
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    self.config.conda_env,
                    "python",
                    f"{self.config.eval_program_path}",
                    "--program_path",
                    f"{exec_fname_t}",
                    "--results_dir",
                    results_dir_t,
                ]
            else:
                cmd = [
                    "python",
                    f"{self.config.eval_program_path}",
                    "--program_path",
                    f"{exec_fname_t}",
                    "--results_dir",
                    results_dir_t,
                ]
        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                # Handle boolean flags
                if isinstance(v, bool):
                    if v:  # Only append flag if True
                        cmd.append(f"--{k}")
                else:
                    # For non-boolean values, append both flag and value
                    cmd.extend([f"--{k}", str(v)])
        return cmd

    def run(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Tuple[Dict[str, Any], float]:
        job_id: Union[str, ProcessWithLogging]
        cmd = self._build_command(exec_fname_t, results_dir_t)
        start_time = time.time()

        if self.job_type == "local":
            assert isinstance(self.config, LocalJobConfig)
            job_id = submit_local(results_dir_t, cmd, verbose=self.verbose)
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
                self.config.modules,
                verbose=self.verbose,
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
            return submit_local(results_dir_t, cmd, verbose=self.verbose)
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
                self.config.modules,
                verbose=self.verbose,
            )
        raise ValueError(f"Unknown job type: {self.job_type}")

    def check_job_status(self, job) -> bool:
        """Check if job is running. Returns True if running, False if done."""
        if self.job_type in ["slurm_docker", "slurm_conda"]:
            from .slurm import get_job_status

            if isinstance(job.job_id, str):
                status = get_job_status(job.job_id)
                return status != ""
            return False  # Should not happen with slurm
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
                        job.job_id.kill()
                        return False

                # More robust status checking with exception handling
                try:
                    return job.job_id.poll() is None
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
        tasks = [self.check_job_status_async(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_job_async(self, job_id: Union[str, ProcessWithLogging]) -> bool:
        """Cancel a running job asynchronously."""
        loop = asyncio.get_event_loop()

        def cancel_job():
            """Cancel job in thread executor."""
            try:
                if self.job_type in ["slurm_docker", "slurm_conda"]:
                    if isinstance(job_id, str):
                        # For SLURM jobs, use scancel command
                        import subprocess

                        result = subprocess.run(
                            ["scancel", job_id], capture_output=True, text=True
                        )
                        return result.returncode == 0
                else:
                    # For local jobs, kill the process
                    if isinstance(job_id, ProcessWithLogging):
                        job_id.kill()
                        return True
                return False
            except Exception as e:
                logger.error(f"Error cancelling job {job_id}: {e}")
                return False

        return await loop.run_in_executor(self.executor, cancel_job)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
