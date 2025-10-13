"""E2B sandbox backend for ShinkaEvolve job execution.

This module provides E2B (secure cloud sandbox) execution backend for running
evolutionary program evaluations in isolated cloud environments.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from shinka.utils import load_results

logger = logging.getLogger(__name__)

# Track active sandboxes for status checks
ACTIVE_SANDBOXES: Dict[str, Dict[str, Any]] = {}


def submit(
    log_dir: str,
    cmd: list[str],
    template: str = "base",
    timeout: int = 600,
    env_vars: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> str:
    """
    Submit a job to an E2B sandbox for execution.

    Args:
        log_dir: Directory to store logs and results
        cmd: Command to execute as list of strings
        template: E2B sandbox template name
        timeout: Timeout in seconds
        env_vars: Environment variables to set in sandbox
        verbose: Whether to enable verbose logging

    Returns:
        str: Sandbox ID (used as job_id for monitoring)
    """
    try:
        from e2b import Sandbox
    except ImportError as e:
        logger.error(
            "E2B package not found. Install with: pip install e2b>=2.2.0"
        )
        raise ImportError(
            "E2B package required for e2b job type. "
            "Install with: pip install e2b>=2.2.0"
        ) from e

    # Ensure log directory exists
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Parse command to extract program_path and results_dir
    # Expected format: python evaluate.py --program_path <path> --results_dir <path>
    program_path = None
    eval_script_path = None
    results_dir = None

    # Extract paths from command
    for i, arg in enumerate(cmd):
        if arg == "--program_path" and i + 1 < len(cmd):
            program_path = cmd[i + 1]
        elif arg == "--results_dir" and i + 1 < len(cmd):
            results_dir = cmd[i + 1]
        elif arg.endswith(".py") and eval_script_path is None:
            eval_script_path = arg

    if not program_path or not eval_script_path:
        raise ValueError(
            f"Could not parse program_path or eval_script from command: {cmd}"
        )

    if verbose:
        logger.info(f"Creating E2B sandbox with template: {template}")
        logger.info(f"Timeout: {timeout}s")
        logger.info(f"Program: {program_path}")
        logger.info(f"Eval script: {eval_script_path}")

    # Create sandbox with timeout in milliseconds
    env_vars = env_vars or {}
    sandbox = Sandbox.create(
        template=template,
        timeout=timeout * 1000,  # Convert to milliseconds
        envs=env_vars,
    )

    sandbox_id = sandbox.id

    if verbose:
        logger.info(f"Created E2B sandbox: {sandbox_id}")

    try:
        # Upload evaluation script
        if Path(eval_script_path).exists():
            with open(eval_script_path, "r", encoding="utf-8") as f:
                eval_script_content = f.read()
            sandbox.files.write(f"/workspace/{Path(eval_script_path).name}", eval_script_content)
            if verbose:
                logger.info(f"Uploaded eval script: {eval_script_path}")
        else:
            logger.warning(f"Eval script not found: {eval_script_path}")

        # Upload program file
        if Path(program_path).exists():
            with open(program_path, "r", encoding="utf-8") as f:
                program_content = f.read()
            sandbox.files.write(f"/workspace/{Path(program_path).name}", program_content)
            if verbose:
                logger.info(f"Uploaded program: {program_path}")
        else:
            raise FileNotFoundError(f"Program file not found: {program_path}")

        # Store sandbox metadata for monitoring
        ACTIVE_SANDBOXES[sandbox_id] = {
            "sandbox": sandbox,
            "log_dir": log_dir,
            "command": cmd,
            "start_time": time.time(),
            "template": template,
            "timeout": timeout,
        }

        if verbose:
            logger.info(f"Submitted E2B job {sandbox_id}")

        return sandbox_id

    except Exception as e:
        # Clean up sandbox on error
        logger.error(f"Error setting up E2B sandbox: {e}")
        try:
            sandbox.close()
        except Exception:
            pass
        raise


def monitor(
    sandbox_id: str,
    results_dir: Optional[str] = None,
    poll_interval: int = 10,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Monitor an E2B sandbox job until completion and load results.

    Args:
        sandbox_id: The E2B sandbox ID to monitor
        results_dir: Directory where results will be stored
        poll_interval: Time in seconds between status checks (not used for E2B)
        verbose: Whether to enable verbose logging

    Returns:
        dict: Dictionary containing job results and metrics
    """
    if sandbox_id not in ACTIVE_SANDBOXES:
        logger.warning(f"Sandbox {sandbox_id} not found in active sandboxes")
        # Try to reconnect to sandbox
        try:
            from e2b import Sandbox
            sandbox = Sandbox(sandbox_id)
            ACTIVE_SANDBOXES[sandbox_id] = {
                "sandbox": sandbox,
                "log_dir": "",
                "command": [],
                "start_time": time.time(),
            }
        except Exception as e:
            logger.error(f"Could not reconnect to sandbox {sandbox_id}: {e}")
            return None

    sandbox_info = ACTIVE_SANDBOXES[sandbox_id]
    sandbox = sandbox_info["sandbox"]
    log_dir = sandbox_info["log_dir"]
    cmd = sandbox_info["command"]
    timeout = sandbox_info.get("timeout", 600)

    if verbose:
        logger.info(f"Monitoring E2B sandbox: {sandbox_id}")

    try:
        # Build command string for sandbox execution
        # Modify paths to use /workspace/ prefix
        cmd_str_parts = []
        for i, arg in enumerate(cmd):
            if arg == "--program_path" and i + 1 < len(cmd):
                cmd_str_parts.append("--program_path")
                cmd_str_parts.append(f"/workspace/{Path(cmd[i + 1]).name}")
            elif arg == "--results_dir" and i + 1 < len(cmd):
                cmd_str_parts.append("--results_dir")
                cmd_str_parts.append("/workspace/results")
            elif arg.endswith(".py") and i > 0 and cmd[i - 1] != "--program_path":
                cmd_str_parts.append(f"/workspace/{Path(arg).name}")
            else:
                if i > 0 and cmd[i - 1] not in ["--program_path", "--results_dir"]:
                    cmd_str_parts.append(arg)

        cmd_str = " ".join(cmd_str_parts)

        if verbose:
            logger.info(f"Executing command in sandbox: {cmd_str}")

        # Execute the command in the sandbox
        result = sandbox.commands.run(cmd_str, timeout=timeout)

        # Write stdout and stderr to log files
        log_dir_path = Path(log_dir)
        stdout_path = log_dir_path / "job_log.out"
        stderr_path = log_dir_path / "job_log.err"

        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(result.stdout or "")

        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(result.stderr or "")

        if verbose:
            logger.info(f"Command exit code: {result.exit_code}")
            if result.exit_code != 0:
                logger.warning(f"Command failed with exit code {result.exit_code}")
                if result.stderr:
                    logger.warning(f"Stderr: {result.stderr[:500]}")

        # Download results from sandbox
        # Try to download common result files
        result_files = ["results.json", "metrics.json", "extra_data.pkl"]
        for result_file in result_files:
            try:
                content = sandbox.files.read(f"/workspace/results/{result_file}")
                if content:
                    local_results_path = log_dir_path / result_file
                    if isinstance(content, bytes):
                        with open(local_results_path, "wb") as f:
                            f.write(content)
                    else:
                        with open(local_results_path, "w", encoding="utf-8") as f:
                            f.write(content)
                    if verbose:
                        logger.info(f"Downloaded result file: {result_file}")
            except Exception as e:
                # File might not exist, which is ok
                if verbose:
                    logger.debug(f"Could not download {result_file}: {e}")

        if verbose:
            logger.info(f"Sandbox {sandbox_id} execution completed")

    except Exception as e:
        logger.error(f"Error monitoring E2B sandbox {sandbox_id}: {e}")
        # Write error to stderr log
        log_dir_path = Path(log_dir)
        stderr_path = log_dir_path / "job_log.err"
        with open(stderr_path, "a", encoding="utf-8") as f:
            f.write(f"\n\nE2B Error: {str(e)}\n")

    finally:
        # Clean up sandbox
        try:
            sandbox.close()
            if verbose:
                logger.info(f"Closed E2B sandbox: {sandbox_id}")
        except Exception as e:
            logger.warning(f"Error closing sandbox {sandbox_id}: {e}")

        # Remove from active sandboxes
        if sandbox_id in ACTIVE_SANDBOXES:
            del ACTIVE_SANDBOXES[sandbox_id]

    # Load and return results
    if results_dir:
        return load_results(results_dir)
    else:
        return load_results(log_dir)


def get_sandbox_status(sandbox_id: str) -> Optional[str]:
    """
    Get status of an E2B sandbox.

    Args:
        sandbox_id: The E2B sandbox ID

    Returns:
        str: Sandbox ID if running, empty string if completed, None if not found
    """
    if sandbox_id in ACTIVE_SANDBOXES:
        # Sandbox is still in our tracking, so it's running
        return sandbox_id
    return ""  # Completed or not found
