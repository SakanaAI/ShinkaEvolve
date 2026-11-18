"""Text of the batch scripts submitted to Slurm."""

from typing import Dict, Optional


def _render_env_exports(eval_env: Optional[Dict[str, str]]) -> str:
    """Render an env dict as newline-joined ``export K=V`` lines for a script."""
    if not eval_env:
        return ""
    return "\n".join(f"export {k}={v}" for k, v in eval_env.items())


def _render_env_docker_flags(eval_env: Optional[Dict[str, str]]) -> str:
    """Render an env dict as space-joined ``-e K=V`` flags for ``docker run``."""
    if not eval_env:
        return ""
    return " ".join(f"-e {k}={v}" for k, v in eval_env.items())


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

{seed_setup}
# (optional) load modules or set env here
module --quiet purge

echo "Job running on $(hostname) under Slurm job $SLURM_JOB_ID"
echo "Launching Docker container…"

# Load image from tar if specified, otherwise pull from registry
{load_command}

docker run --rm \\
    {eval_env_docker_flags} \\
    {docker_flags} \\
    {image} {cmd}

{seed_finish}
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

{seed_setup}
# Load modules
module --quiet purge
{module_load_commands}

echo "Job running on $(hostname) under Slurm job $SLURM_JOB_ID"

# Activate Python environment when configured
{activation_commands}

{eval_env_exports}
{cmd}

{seed_finish}
"""
