import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import threading
from shinka.utils import load_results
from shinka.utils.security import (
    validate_safe_path,
    validate_docker_image_name,
    sanitize_command_args,
    SecurityError,
)
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


def load_cache_manifest():
    """Load the cache manifest file with proper error handling."""
    if not CACHE_MANIFEST.exists():
        return {}

    try:
        with open(CACHE_MANIFEST, "r") as f:
            data = json.load(f)

        # Validate structure
        if not isinstance(data, dict):
            logger.warning(f"Invalid cache manifest format: {CACHE_MANIFEST}")
            return {}

        return data

    except json.JSONDecodeError as e:
        logger.error(f"Malformed cache manifest: {CACHE_MANIFEST} - {e}")
        return {}
    except PermissionError:
        logger.error(f"Permission denied reading cache manifest: {CACHE_MANIFEST}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading cache manifest: {e}")
        return {}


def save_cache_manifest(manifest):
    """Save the cache manifest file."""
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def get_local_image(image_name):
    """
    Check if image exists locally and return the appropriate image name.

    Args:
        image_name: Docker image name to retrieve

    Returns:
        str: Validated image name

    Raises:
        SecurityError: If image name is invalid
    """
    # Validate image name first to prevent command injection
    validated_image = validate_docker_image_name(image_name)

    manifest = load_cache_manifest()

    # Check if image is in manifest
    if validated_image in manifest:
        local_path = DOCKER_CACHE_DIR / manifest[validated_image]
        if local_path.exists():
            # Return original image name instead of local registry
            return validated_image

    # Try to pull and cache the image
    try:
        logger.info(f"Pulling and caching {validated_image}...")
        subprocess.run(
            ["docker", "pull", validated_image],
            check=True,
            timeout=300  # 5 minute timeout
        )

        # Save the image
        image_file = f"{validated_image.replace('/', '_').replace(':', '_')}.tar"
        image_path = DOCKER_CACHE_DIR / image_file
        subprocess.run(
            ["docker", "save", validated_image, "-o", str(image_path)],
            check=True,
            timeout=300
        )

        # Update manifest
        manifest[validated_image] = image_file
        save_cache_manifest(manifest)

        return validated_image
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to pull Docker image {validated_image}: {e.stderr if hasattr(e, 'stderr') else e}"
        )
        raise RuntimeError(f"Docker image pull failed: {validated_image}") from e
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout pulling Docker image: {validated_image}")
        raise RuntimeError(f"Docker image pull timeout: {validated_image}")


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

# Activate conda environment
if [ -n "{conda_env}" ]; then
    echo "Activating conda environment: {conda_env}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate {conda_env}
fi

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

    # Secure path validation - use current working directory as base
    cwd = os.getcwd()
    try:
        safe_log_dir = validate_safe_path(cwd, log_dir)
    except SecurityError:
        # If validation fails, fall back to absolute path within cwd
        logger.warning(f"Path validation failed for {log_dir}, using sanitized path")
        safe_log_dir = Path(cwd) / "logs" / Path(log_dir).name

    log_dir_str = str(safe_log_dir)
    os.makedirs(log_dir_str, exist_ok=True)

    # Validate Docker image name
    validated_image = validate_docker_image_name(image)

    # Sanitize command arguments to prevent injection
    safe_cmd = sanitize_command_args(cmd)

    load_command = ""
    if image_tar_path:
        # Validate the tar path
        safe_tar_path = str(Path(image_tar_path).resolve())
        load_command = f"""
if [ -f "{safe_tar_path}" ]; then
    echo "Loading image from {safe_tar_path}..."
    docker load < "{safe_tar_path}"
else
    echo "Image tar file not found at {safe_tar_path}, exiting."
    exit 1
fi
"""
    else:
        # Fallback to existing pull/cache logic
        get_local_image(validated_image)  # This function pulls and caches the image
        image_file = f"{validated_image.replace('/', '_').replace(':', '_')}.tar"
        cache_dir_str = str(DOCKER_CACHE_DIR)
        load_command = f"""
if [ -f "{cache_dir_str}/{image_file}" ]; then
    echo "Loading cached image..."
    docker load < "{cache_dir_str}/{image_file}"
    if ! docker image inspect {validated_image} >/dev/null 2>&1; then
        echo "Failed to load cached image, pulling from registry..."
        docker pull {validated_image}
    fi
else
    echo "Pulling image..."
    docker pull {validated_image}
fi
"""

    if mem is not None:
        sbatch_kwargs["mem"] = mem

    additional_sbatch_params = ""
    for k, v in sbatch_kwargs.items():
        additional_sbatch_params += f"#SBATCH --{k}={v}"

    sbatch_script = SBATCH_DOCKER_TEMPLATE.format(
        job_name=job_name,
        log_dir=log_dir_str,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        docker_flags=docker_flags,
        image=validated_image,
        cmd=safe_cmd,
        load_command=load_command,
    )

    # Create temporary file with proper cleanup
    sbatch_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
            f.write(sbatch_script)
            sbatch_path = f.name

        result = subprocess.run(
            ["sbatch", sbatch_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=30
        )
        # Slurm replies: "Submitted batch job <jobid>"
        job_id = result.stdout.strip().split()[-1]
        if verbose:
            logger.info(f"Submitted Docker job {job_id}")
        return job_id
    finally:
        # Clean up temporary file
        if sbatch_path and os.path.exists(sbatch_path):
            try:
                os.unlink(sbatch_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {sbatch_path}: {e}")


def submit_conda(
    log_dir: str,
    cmd: list[str],
    time: str,
    partition: str,
    cpus: int,
    gpus: int,
    mem: Optional[str],
    conda_env: str = "",
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
            modules=modules,
            verbose=verbose,
            **sbatch_kwargs,
        )
    job_name = f"conda-{uuid.uuid4().hex[:6]}"

    # Secure path validation
    cwd = os.getcwd()
    try:
        safe_log_dir = validate_safe_path(cwd, log_dir)
    except SecurityError:
        logger.warning(f"Path validation failed for {log_dir}, using sanitized path")
        safe_log_dir = Path(cwd) / "logs" / Path(log_dir).name

    log_dir_str = str(safe_log_dir)
    os.makedirs(log_dir_str, exist_ok=True)

    if modules is None:
        modules = []

    # Sanitize module names to prevent injection
    safe_modules = []
    for module in modules:
        # Basic validation - module names should be alphanumeric with limited special chars
        if re.match(r'^[a-zA-Z0-9._/-]+$', module):
            safe_modules.append(module)
        else:
            logger.warning(f"Skipping potentially unsafe module name: {module}")

    module_load_commands = "\n".join([f"module load {module}" for module in safe_modules])

    # Sanitize command arguments
    safe_cmd = sanitize_command_args(cmd)

    if mem is not None:
        sbatch_kwargs["mem"] = mem

    additional_sbatch_params = ""
    for k, v in sbatch_kwargs.items():
        additional_sbatch_params += f"#SBATCH --{k}={v}"

    sbatch_script = SBATCH_CONDA_TEMPLATE.format(
        job_name=job_name,
        log_dir=log_dir_str,
        time=time,
        partition=partition,
        cpus=cpus,
        gpus=gpus,
        additional_sbatch_params=additional_sbatch_params,
        conda_env=conda_env,
        module_load_commands=module_load_commands,
        cmd=safe_cmd,
    )

    # Create temporary file with proper cleanup
    sbatch_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as f:
            f.write(sbatch_script)
            sbatch_path = f.name

        result = subprocess.run(
            ["sbatch", sbatch_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=30
        )

        # Slurm replies: "Submitted batch job <jobid>"
        job_id = result.stdout.strip().split()[-1]
        if verbose:
            logger.info(f"Submitted Conda job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() if e.stderr else str(e)
        logger.error(f"Failed to submit Conda job: {err_msg}")
        logger.debug(f"Failed sbatch script: {sbatch_script}")
        raise
    finally:
        # Clean up temporary file
        if sbatch_path and os.path.exists(sbatch_path):
            try:
                os.unlink(sbatch_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {sbatch_path}: {e}")


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
    """Submit a job to run locally in a Docker container (SECURITY FIXED)."""
    job_id = f"local-{uuid.uuid4().hex[:6]}"

    # Secure path validation
    cwd = os.getcwd()
    try:
        safe_log_dir = validate_safe_path(cwd, log_dir)
    except SecurityError:
        logger.warning(f"Path validation failed for {log_dir}, using sanitized path")
        safe_log_dir = Path(cwd) / "logs" / Path(log_dir).name

    os.makedirs(safe_log_dir, exist_ok=True)

    # Validate and get image
    image_name = get_local_image(image)  # This validates the image name
    image_file = f"{image_name.replace('/', '_').replace(':', '_')}.tar"

    # Sanitize command arguments
    safe_cmd = sanitize_command_args(cmd)

    # Build safer bash command using properly escaped variables
    # Still using shell for the conditional, but with proper escaping
    cache_path = str(DOCKER_CACHE_DIR / image_file)
    log_out = str(safe_log_dir / "job_log.out")
    log_err = str(safe_log_dir / "job_log.err")

    # Use shlex.quote for all dynamic values in the shell command
    import shlex
    full = (
        f"if [ -f {shlex.quote(cache_path)} ]; then "
        f"docker load < {shlex.quote(cache_path)}; "
        f"else docker pull {shlex.quote(image_name)}; fi; "
        f"docker run --rm {docker_flags} {shlex.quote(image_name)} "
        f"{safe_cmd} >> {shlex.quote(log_out)} "
        f"2>> {shlex.quote(log_err)}"
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
    modules: Optional[list[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """Submit local conda job (SECURITY FIXED)."""
    job_id = f"local-conda-{uuid.uuid4().hex[:6]}"

    # Secure path validation
    cwd = os.getcwd()
    try:
        safe_log_dir = validate_safe_path(cwd, log_dir)
    except SecurityError:
        logger.warning(f"Path validation failed for {log_dir}, using sanitized path")
        safe_log_dir = Path(cwd) / "logs" / Path(log_dir).name

    log_dir_str = str(safe_log_dir)
    os.makedirs(log_dir_str, exist_ok=True)

    modules = modules or []

    # Sanitize module names
    safe_modules = []
    for module in modules:
        if re.match(r'^[a-zA-Z0-9._/-]+$', module):
            safe_modules.append(module)
        else:
            logger.warning(f"Skipping potentially unsafe module name: {module}")

    # Sanitize command arguments
    safe_cmd = sanitize_command_args(cmd)

    # Use shlex.quote for all dynamic values
    import shlex
    loads = "; ".join([f"module load {shlex.quote(m)}" for m in safe_modules])
    log_out = str(safe_log_dir / "job_log.out")
    log_err = str(safe_log_dir / "job_log.err")

    full_cmd = (
        f"module --quiet purge; {loads}; "
        f"source $(conda info --base)/etc/profile.d/conda.sh; "
        f"conda activate {shlex.quote(conda_env)}; "
        f"{safe_cmd} >> {shlex.quote(log_out)} "
        f"2>> {shlex.quote(log_err)}"
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
