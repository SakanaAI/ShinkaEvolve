import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import threading
from shinka.utils import load_results
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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


def _has_value(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


def _escape_double_quotes(value: str) -> str:
    return value.replace('"', '\\"')


def _build_activation_commands(
    conda_env: Optional[str] = None,
    activate_script: Optional[str] = None,
    *,
    separator: str = "\n",
) -> str:
    if _has_value(conda_env) and _has_value(activate_script):
        raise ValueError("conda_env and activate_script are mutually exclusive")

    commands: list[str] = []
    if _has_value(activate_script):
        quoted_path = _escape_double_quotes(activate_script.strip())
        commands.extend(
            [
                f'echo "Activating sourced environment: {quoted_path}"',
                f'source "{quoted_path}"',
            ]
        )
    elif _has_value(conda_env):
        env_name = conda_env.strip()
        commands.extend(
            [
                f'echo "Activating conda environment: {env_name}"',
                "source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {env_name}",
            ]
        )
    return separator.join(commands)


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


def get_local_image(image_name):
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
        subprocess.run(["docker", "pull", image_name], check=True)

        # Save the image
        image_file = f"{image_name.replace('/', '_').replace(':', '_')}.tar"
        image_path = DOCKER_CACHE_DIR / image_file
        subprocess.run(
            ["docker", "save", image_name, "-o", str(image_path)], check=True
        )

        # Update manifest
        manifest[image_name] = image_file
        save_cache_manifest(manifest)

        return image_name
    except subprocess.CalledProcessError:
        logger.info(f"Warning: Could not pull {image_name}, using as is")
        return image_name


SBATCH_DOCKER_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/job_log.out
#SBATCH --error={log_dir}/job_log.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
{additional_sbatch_params}

# (optional) load modules or set env here
module --quiet purge

echo "Job running on $(hostname) under Slurm job $SLURM_JOB_ID"
echo "Launching Docker container…"

# Load image from tar if specified, otherwise pull from registry
{load_command}

docker run --rm \\
    {docker_flags} \\
    {image} {cmd}

exit $?
"""

SBATCH_CONDA_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/job_log.out
#SBATCH --error={log_dir}/job_log.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
{additional_sbatch_params}

# Load modules
module --quiet purge
{module_load_commands}

echo "Job running on $(hostname) under Slurm job $SLURM_JOB_ID"

# Activate Python environment when configured
{activation_commands}

{cmd}

exit $?
"""


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
    local: bool = False,
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
            **sbatch_kwargs,
        )
    job_name = f"docker-{uuid.uuid4().hex[:6]}"
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
        get_local_image(image)  # This function pulls and caches the image
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

    sbatch_script = SBATCH_DOCKER_TEMPLATE.format(
        job_name=job_name,
        log_dir=log_dir,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        docker_flags=docker_flags,
        image=image,
        cmd=" ".join(cmd),
        load_command=load_command,
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
        f.write(sbatch_script)
        sbatch_path = f.name

    result = subprocess.run(
        ["sbatch", sbatch_path], stdout=subprocess.PIPE, check=True, text=True
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
    local: bool = False,
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
            **sbatch_kwargs,
        )
    job_name = f"conda-{uuid.uuid4().hex[:6]}"
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

    sbatch_script = SBATCH_CONDA_TEMPLATE.format(
        job_name=job_name,
        log_dir=log_dir,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        activation_commands=activation_commands,
        module_load_commands=module_load_commands,
        cmd=" ".join(cmd),
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
        f.write(sbatch_script)
        sbatch_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", sbatch_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
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
    **kwargs,
) -> str:
    """Submit a job to run locally in a Docker container."""
    job_id = f"local-{uuid.uuid4().hex[:6]}"
    log_dir_path = Path(log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    image_name = get_local_image(image)
    image_file = f"{image.replace('/', '_').replace(':', '_')}.tar"
    # build bash command with logging
    full = (
        f"if [ -f '{DOCKER_CACHE_DIR}/{image_file}' ]; then "
        f"docker load < '{DOCKER_CACHE_DIR}/{image_file}'; "
        f"else docker pull {image_name}; fi; "
        f"docker run --rm {docker_flags} {image_name} "
        f"{' '.join(cmd)} >> {log_dir}/job_log.out "
        f"2>> {log_dir}/job_log.err"
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
    full_cmd = "; ".join(
        segment
        for segment in [
            "module --quiet purge",
            loads,
            activation_commands,
            f"{' '.join(cmd)} >> {log_dir}/job_log.out 2>> {log_dir}/job_log.err",
        ]
        if segment
    )
    launch_local_subprocess(job_id, ["bash", "-lc", full_cmd], gpus)
    if verbose:
        logger.info(f"Submitted local Conda job {job_id}")
    return job_id


def get_job_status(job_id: str) -> Optional[str]:
    """Get status for Slurm or local jobs."""
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
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def monitor(job_id, results_dir=None, poll_interval=10, verbose: bool = False):
    """
    Monitor a Slurm job until completion and load its results.

    Args:
        job_id: The Slurm job ID to monitor
        poll_interval: Time in seconds between status checks

    Returns:
        dict: Dictionary containing job results and metrics
    """
    if verbose:
        logger.info(f"Monitoring job {job_id}...")

    # Monitor job status
    while True:
        status = get_job_status(job_id)
        if status == "":
            if verbose:
                logger.info("Job completed!")
            break

        if verbose:
            logger.info(f"\rJob status: {status}", end="", flush=True)
        time.sleep(poll_interval)

    if results_dir is not None:
        return load_results(results_dir)
