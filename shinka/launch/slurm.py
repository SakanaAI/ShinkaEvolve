import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import uuid
import threading
from shinka.utils import load_results
from typing import Optional, Dict
import logging

from .slurm_results import render_seed_evaluator
from .slurm_script import (
    SBATCH_CONDA_TEMPLATE,
    SBATCH_DOCKER_TEMPLATE,
    _build_activation_commands,
    _render_env_docker_flags,
    _render_env_exports,
)
from .slurm_status import STATUS_COMMAND_TIMEOUT_SECONDS, is_unknown_job_error

logger = logging.getLogger(__name__)

# Failed status lookups in a row before an unbounded ``monitor`` gives up.
MONITOR_STATUS_ERROR_LIMIT = 3


# Configuration for Docker image caching
DOCKER_CACHE_DIR = Path(os.path.expanduser("~/docker_cache"))
try:
    DOCKER_CACHE_DIR.mkdir(exist_ok=True)
except PermissionError:
    # This can happen if the module is imported in a restricted environment
    # (like a Docker container) where the user doesn't have a home directory
    # or write access to it. This is fine if we're not using the caching feature.
    pass
CACHE_MANIFEST = DOCKER_CACHE_DIR / "cache_manifest.json"

# track local jobs for status checks
LOCAL_JOBS: dict[str, dict] = {}


def load_cache_manifest():
    """Load the cache manifest file."""
    if CACHE_MANIFEST.exists():
        with open(CACHE_MANIFEST, "r") as f:
            return json.load(f)
    return {}


def save_cache_manifest(manifest):
    """Save the cache manifest file."""
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def get_local_image(image_name: str, *, command_timeout: Optional[float] = None) -> str:
    """Check if image exists locally and return the appropriate image name."""
    manifest = load_cache_manifest()

    # Check if image is in manifest
    if image_name in manifest:
        local_path = DOCKER_CACHE_DIR / manifest[image_name]
        if local_path.exists():
            # Return original image name instead of local registry
            return image_name

    # Try to pull and cache the image
    try:
        logger.info(f"Pulling and caching {image_name}...")
        subprocess.run(
            ["docker", "pull", image_name], check=True, timeout=command_timeout
        )

        # Save the image
        image_file = f"{image_name.replace('/', '_').replace(':', '_')}.tar"
        image_path = DOCKER_CACHE_DIR / image_file
        try:
            with tempfile.TemporaryDirectory(
                prefix=".image-", dir=DOCKER_CACHE_DIR
            ) as staging:
                staged_image = Path(staging) / image_file
                subprocess.run(
                    ["docker", "save", image_name, "-o", str(staged_image)],
                    check=True,
                    timeout=command_timeout,
                )
                staged_image.replace(image_path)
        except OSError as exc:
            logger.warning("Could not stage Docker cache for %s: %s", image_name, exc)
            return image_name

        # Update manifest
        manifest[image_name] = image_file
        save_cache_manifest(manifest)

        return image_name
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "Docker cache preparation timed out for %s. The batch job can pull it: %s",
            image_name,
            exc,
        )
        return image_name
    except subprocess.CalledProcessError:
        logger.info(f"Warning: Could not pull {image_name}, using as is")
        return image_name


def _sbatch_command(sbatch_path: str, job_name: Optional[str]) -> list[str]:
    # The CLI option takes precedence over SBATCH_JOB_NAME.
    if job_name is not None:
        return ["sbatch", f"--job-name={job_name}", sbatch_path]
    return ["sbatch", sbatch_path]


def _seed_submission_env(seed_results_dir: Optional[str]) -> Optional[Dict[str, str]]:
    if seed_results_dir is None or "SLURM_RESTART_COUNT" not in os.environ:
        return None
    # A new job submitted from inside a restarted batch starts at execution zero.
    env = os.environ.copy()
    env.pop("SLURM_RESTART_COUNT")
    return env


def submit_docker(
    log_dir: str,
    cmd: list[str],
    time: str,
    partition: str,
    cpus: int,
    gpus: int,
    mem: Optional[str],
    docker_flags: str,
    image: str,
    image_tar_path: Optional[str] = None,
    verbose: bool = False,
    eval_env: Optional[Dict[str, str]] = None,
    local: bool = False,
    *,
    command_timeout: Optional[float] = None,
    job_name: Optional[str] = None,
    seed_results_dir: Optional[str] = None,
    **sbatch_kwargs,
):
    if local:
        return submit_local_docker(
            log_dir=log_dir,
            cmd=cmd,
            time=time,
            partition=partition,
            cpus=cpus,
            gpus=gpus,
            mem=mem,
            docker_flags=docker_flags,
            image=image,
            image_tar_path=image_tar_path,
            verbose=verbose,
            eval_env=eval_env,
            **sbatch_kwargs,
        )
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    load_command = ""
    if image_tar_path:
        load_command = f"""
if [ -f "{image_tar_path}" ]; then
    echo "Loading image from {image_tar_path}..."
    docker load < "{image_tar_path}"
else
    echo "Image tar file not found at {image_tar_path}, exiting."
    exit 1
fi
"""
    else:
        # Fallback to existing pull/cache logic
        get_local_image(image, command_timeout=command_timeout)
        image_file = f"{image.replace('/', '_').replace(':', '_')}.tar"
        load_command = f"""
if [ -f "{DOCKER_CACHE_DIR}/{image_file}" ]; then
    echo "Loading cached image..."
    docker load < "{DOCKER_CACHE_DIR}/{image_file}"
    if ! docker image inspect {image} >/dev/null 2>&1; then
        echo "Failed to load cached image, pulling from registry..."
        docker pull {image}
    fi
else
    echo "Pulling image..."
    docker pull {image}
fi
"""

    if mem is not None:
        sbatch_kwargs["mem"] = mem

    additional_sbatch_params = ""
    for k, v in sbatch_kwargs.items():
        additional_sbatch_params += f"#SBATCH --{k}={v}"

    batch_name = job_name or f"docker-{uuid.uuid4().hex[:6]}"
    seed_setup, command, seed_finish = render_seed_evaluator(
        cmd, seed_results_dir, batch_name
    )
    sbatch_script = SBATCH_DOCKER_TEMPLATE.format(
        job_name=batch_name,
        log_dir=log_dir,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        docker_flags=docker_flags,
        image=image,
        cmd=command,
        load_command=load_command,
        eval_env_docker_flags=_render_env_docker_flags(eval_env),
        seed_setup=seed_setup,
        seed_finish=seed_finish,
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
        f.write(sbatch_script)
        sbatch_path = f.name

    result = subprocess.run(
        _sbatch_command(sbatch_path, job_name),
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=command_timeout,
        env=_seed_submission_env(seed_results_dir),
    )
    # Slurm replies: "Submitted batch job <jobid>"
    job_id = result.stdout.strip().split()[-1]
    if verbose:
        logger.info(f"Submitted Docker job {job_id}")
    return job_id


def submit_conda(
    log_dir: str,
    cmd: list[str],
    time: str,
    partition: str,
    cpus: int,
    gpus: int,
    mem: Optional[str],
    conda_env: str = "",
    activate_script: Optional[str] = None,
    modules: Optional[list[str]] = None,
    verbose: bool = False,
    eval_env: Optional[Dict[str, str]] = None,
    local: bool = False,
    *,
    command_timeout: Optional[float] = None,
    job_name: Optional[str] = None,
    seed_results_dir: Optional[str] = None,
    **sbatch_kwargs,
):
    if local:
        return submit_local_conda(
            log_dir=log_dir,
            cmd=cmd,
            time=time,
            partition=partition,
            cpus=cpus,
            gpus=gpus,
            mem=mem,
            conda_env=conda_env,
            activate_script=activate_script,
            modules=modules,
            verbose=verbose,
            eval_env=eval_env,
            **sbatch_kwargs,
        )
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if modules is None:
        modules = []

    module_load_commands = "\n".join([f"module load {module}" for module in modules])
    activation_commands = _build_activation_commands(
        conda_env=conda_env,
        activate_script=activate_script,
    )

    if mem is not None:
        sbatch_kwargs["mem"] = mem

    additional_sbatch_params = ""
    for k, v in sbatch_kwargs.items():
        additional_sbatch_params += f"#SBATCH --{k}={v}"

    batch_name = job_name or f"conda-{uuid.uuid4().hex[:6]}"
    seed_setup, command, seed_finish = render_seed_evaluator(
        cmd, seed_results_dir, batch_name
    )
    sbatch_script = SBATCH_CONDA_TEMPLATE.format(
        job_name=batch_name,
        log_dir=log_dir,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        activation_commands=activation_commands,
        module_load_commands=module_load_commands,
        cmd=command,
        eval_env_exports=_render_env_exports(eval_env),
        seed_setup=seed_setup,
        seed_finish=seed_finish,
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
        f.write(sbatch_script)
        sbatch_path = f.name

    try:
        result = subprocess.run(
            _sbatch_command(sbatch_path, job_name),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=command_timeout,
            env=_seed_submission_env(seed_results_dir),
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() if e.stderr else str(e)
        logger.info(f"Error failed to submit Conda job: {err_msg}")
        logger.info(f"Failed sbatch script: {sbatch_script}")
        raise

    # Slurm replies: "Submitted batch job <jobid>"
    job_id = result.stdout.strip().split()[-1]
    if verbose:
        logger.info(f"Submitted Conda job {job_id}")
    return job_id


def launch_local_subprocess(
    job_id: str,
    cmd: list[str],
    gpus: int,
):
    """Wait for free gpus then launch async."""

    def runner():
        while True:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            lines = res.stdout.strip().splitlines()
            free = []
            for ln in lines:
                idx, mem_used = [x.strip() for x in ln.split(",")]
                if mem_used == "0":
                    free.append(idx)
            if len(free) >= gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free[:gpus])
                proc = subprocess.Popen(cmd)
                LOCAL_JOBS[job_id]["popen"] = proc
                break
            time.sleep(5)

    LOCAL_JOBS[job_id] = {"popen": None}
    t = threading.Thread(target=runner, daemon=True)
    t.start()


def submit_local_docker(
    log_dir: str,
    cmd: list[str],
    time: str,
    partition: str,
    cpus: int,
    gpus: int,
    mem: Optional[str],
    docker_flags: str,
    image: str,
    image_tar_path: Optional[str] = None,
    verbose: bool = False,
    eval_env: Optional[Dict[str, str]] = None,
    **kwargs,
) -> str:
    """Submit a job to run locally in a Docker container."""
    job_id = f"local-{uuid.uuid4().hex[:6]}"
    log_dir_path = Path(log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    image_name = get_local_image(image)
    image_file = f"{image.replace('/', '_').replace(':', '_')}.tar"
    docker_env_flags = _render_env_docker_flags(eval_env)
    # build bash command with logging
    full = (
        f"if [ -f '{DOCKER_CACHE_DIR}/{image_file}' ]; then "
        f"docker load < '{DOCKER_CACHE_DIR}/{image_file}'; "
        f"else docker pull {image_name}; fi; "
        f"docker run --rm {docker_env_flags} "
        f"{docker_flags} {image_name} "
        f"{shlex.join(cmd)} >> {shlex.quote(f'{log_dir}/job_log.out')} "
        f"2>> {shlex.quote(f'{log_dir}/job_log.err')}"
    )
    launch_local_subprocess(job_id, ["bash", "-lc", full], gpus)
    if verbose:
        logger.info(f"Submitted local Docker job {job_id}")
    return job_id


def submit_local_conda(
    log_dir: str,
    cmd: list[str],
    time: str,
    partition: str,
    cpus: int,
    gpus: int,
    mem: Optional[str],
    conda_env: str = "",
    activate_script: Optional[str] = None,
    modules: Optional[list[str]] = None,
    verbose: bool = False,
    eval_env: Optional[Dict[str, str]] = None,
    **kwargs,
) -> str:
    """Submit local conda job."""
    job_id = f"local-conda-{uuid.uuid4().hex[:6]}"
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    modules = modules or []
    loads = "; ".join([f"module load {m}" for m in modules])
    activation_commands = _build_activation_commands(
        conda_env=conda_env,
        activate_script=activate_script,
        separator="; ",
    )
    logs = (shlex.quote(f"{log_dir}/job_log.out"), shlex.quote(f"{log_dir}/job_log.err"))
    full_cmd = "; ".join(
        segment
        for segment in [
            "module --quiet purge",
            loads,
            activation_commands,
            _render_env_exports(eval_env, separator="; "),
            f"{shlex.join(cmd)} >> {logs[0]} 2>> {logs[1]}",
        ]
        if segment
    )
    launch_local_subprocess(job_id, ["bash", "-lc", full_cmd], gpus)
    if verbose:
        logger.info(f"Submitted local Conda job {job_id}")
    return job_id


def get_job_status(
    job_id: str, *, timeout: Optional[float] = STATUS_COMMAND_TIMEOUT_SECONDS
) -> Optional[str]:
    """Get status for Slurm or local jobs.

    Returns the queue entry of an active job, ``""`` once the job has left the
    queue, and ``None`` when the status could not be determined.
    """
    if job_id.startswith("local-"):
        job = LOCAL_JOBS.get(job_id)
        if not job:
            return None
        proc = job.get("popen")
        if proc and proc.poll() is None:
            return job_id
        return ""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "--noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as error:
        # The controller purges a finished job after MinJobAge.
        if is_unknown_job_error(error.stderr):
            return ""
        logger.warning("squeue failed for job %s: %s", job_id, error.stderr or error)
        return None
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("squeue failed for job %s: %s", job_id, error)
        return None


def monitor(
    job_id,
    results_dir=None,
    poll_interval=10,
    verbose: bool = False,
    *,
    max_wait: Optional[float] = None,
):
    """
    Monitor a Slurm job until completion and load its results.

    Args:
        job_id: The Slurm job ID to monitor
        poll_interval: Time in seconds between status checks
        max_wait: Stop waiting after this many seconds, or as soon as the
            status cannot be determined, and load whatever results exist.
            Without it, the wait ends when the job leaves the queue; several
            failed status lookups in a row raise ``RuntimeError``.

    Returns:
        dict: Dictionary containing job results and metrics
    """
    if verbose:
        logger.info(f"Monitoring job {job_id}...")

    deadline = None if max_wait is None else time.monotonic() + max_wait
    consecutive_errors = 0
    while True:
        status = get_job_status(job_id, timeout=_status_timeout(deadline))
        if status == "":
            if verbose:
                logger.info("Job completed!")
            break

        if verbose:
            logger.info(f"Job {job_id} status: {status}")
        if deadline is None:
            consecutive_errors = consecutive_errors + 1 if status is None else 0
            if consecutive_errors >= MONITOR_STATUS_ERROR_LIMIT:
                raise RuntimeError(
                    f"Slurm status lookup for {job_id} failed "
                    f"{consecutive_errors} consecutive times"
                )
            pause = poll_interval
        else:
            pause = min(poll_interval, deadline - time.monotonic())
            if status is None or pause <= 0:
                logger.warning(
                    "Could not confirm that job %s left the queue; loading the "
                    "results it has",
                    job_id,
                )
                break
        time.sleep(pause)

    if results_dir is not None:
        return load_results(results_dir)


def _status_timeout(deadline: Optional[float]) -> float:
    if deadline is None:
        return STATUS_COMMAND_TIMEOUT_SECONDS
    remaining = deadline - time.monotonic()
    return max(1.0, min(STATUS_COMMAND_TIMEOUT_SECONDS, remaining))
