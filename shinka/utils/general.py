import json
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def load_configs_from_yaml(config_path: str):
    """
    Loads configs from a YAML file.
    """
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    assert "db_config" in configs, "db_config not found in config file"
    assert "evo_config" in configs, "evo_config not found in config file"

    db_cfg = DatabaseConfig(**configs["db_config"])
    evo_cfg = EvolutionConfig(**configs["evo_config"])
    return evo_cfg, db_cfg


def load_results(results_dir: str):
    """
    Loads results from the specified directory.

    Args:
        results_dir: The directory containing the results.

    Returns:
        dict: A dictionary containing the loaded results.
    """
    loaded_results = {"correct": {"correct": False}, "metrics": {}}
    results_dir_path = Path(results_dir)

    stdout_log_path = results_dir_path / "job_log.out"
    if stdout_log_path.exists():
        with open(stdout_log_path, "r") as f:
            loaded_results["stdout_log"] = f.read()
    else:
        loaded_results["stdout_log"] = ""

    stderr_log_path = results_dir_path / "job_log.err"
    if stderr_log_path.exists():
        with open(stderr_log_path, "r") as f:
            loaded_results["stderr_log"] = f.read()
    else:
        loaded_results["stderr_log"] = ""

    metrics_file_path = results_dir_path / "metrics.json"
    if metrics_file_path.exists():
        with open(metrics_file_path, "r") as f:
            try:
                loaded_results["metrics"] = json.load(f)
            except json.JSONDecodeError:
                file_path_str = str(metrics_file_path)
                warning_msg = f"Could not decode JSON from {file_path_str}"
                logger.warning(warning_msg)
                loaded_results["metrics"] = {}
    else:
        file_path_str = str(metrics_file_path)
        warning_msg = f"Metrics file not found at {file_path_str}"
        logger.warning(warning_msg)
        loaded_results["metrics"] = {}

    correct_file_path = results_dir_path / "correct.json"
    if correct_file_path.exists():
        with open(correct_file_path, "r") as f:
            loaded_results["correct"] = json.load(f)
    else:
        loaded_results["correct"] = {"correct": False}

    return loaded_results


def parse_time_to_seconds(time_str: str) -> int:
    """Converts hh:mm:ss to seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time format must be hh:mm:ss")
    h, m, s = [int(p) for p in parts]
    return h * 3600 + m * 60 + s
