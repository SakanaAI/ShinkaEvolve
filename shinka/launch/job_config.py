"""Configuration of the jobs that evaluate candidate programs."""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

from shinka.utils import parse_time_to_seconds

_UNLIMITED_SLURM_TIMES = frozenset({"unlimited", "infinite", "-1"})


def _has_value(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


def _validate_activation_config(
    conda_env: Optional[str],
    activate_script: Optional[str],
) -> None:
    if _has_value(conda_env) and _has_value(activate_script):
        raise ValueError("conda_env and activate_script are mutually exclusive")


def _validate_slurm_time(time_limit: str) -> None:
    """Reject a ``time`` that ``sbatch --time`` would not accept."""
    if str(time_limit).strip().lower() not in _UNLIMITED_SLURM_TIMES:
        parse_time_to_seconds(time_limit)


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
        if self.time:
            parse_time_to_seconds(self.time)


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

    def __post_init__(self) -> None:
        _validate_slurm_time(self.time)


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
        _validate_slurm_time(self.time)
        if self.modules is None:
            self.modules = []


SlurmEnvJobConfig = SlurmCondaJobConfig
