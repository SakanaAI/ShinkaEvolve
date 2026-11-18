import json
from pathlib import Path
import logging
from typing import Any, Optional
import yaml

logger = logging.getLogger(__name__)


def truncate_log_tail(text: str, max_chars: Optional[int] = None) -> str:
    """Return only the last ``max_chars`` characters of ``text``.

    Tail-biased because the end of a captured log holds the final results /
    error context. ``max_chars`` of None or <= 0 returns ``text`` unchanged. On
    truncation, a one-line marker noting how many characters were dropped is
    prepended so consumers can tell the log is partial.
    """
    if not text or max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    marker = f"... [truncated {dropped} chars of stdout; full log on disk] ...\n"
    return marker + text[-max_chars:]


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


def load_result_logs(results_dir: str) -> dict[str, str]:
    """Read evaluator logs even when its result files cannot be loaded."""
    logs = {}
    for stream, filename in (
        ("stdout_log", "job_log.out"),
        ("stderr_log", "job_log.err"),
    ):
        path = Path(results_dir) / filename
        try:
            logs[stream] = path.read_text(errors="replace")
        except FileNotFoundError:
            logs[stream] = ""
        except OSError as exc:
            logger.warning("Could not read evaluator log %s: %s", path, exc)
            logs[stream] = ""
    return logs


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _unreadable_json(path: Path, exc: Exception) -> str:
    return f"Could not decode JSON from {path} ({type(exc).__name__}: {exc})"


def load_results(results_dir: str, *, log_dir: Optional[str] = None) -> dict[str, Any]:
    """
    Loads results from the specified directory.

    Args:
        results_dir: The directory containing the results.
        log_dir: The directory containing the evaluator logs, if separate.

    Returns:
        dict: A dictionary containing the loaded results.
    """
    loaded_results = {
        "correct": {"correct": False},
        "metrics": {},
        **load_result_logs(results_dir if log_dir is None else log_dir),
    }
    results_dir_path = Path(results_dir)

    metrics_file_path = results_dir_path / "metrics.json"
    if metrics_file_path.is_file():
        try:
            loaded_results["metrics"] = _load_json(metrics_file_path)
        except (OSError, ValueError) as exc:
            warning_msg = _unreadable_json(metrics_file_path, exc)
            logger.warning(warning_msg)
            loaded_results["evaluation_error"] = warning_msg
    else:
        warning_msg = f"Metrics file not found at {metrics_file_path}"
        logger.warning(warning_msg)
        loaded_results["evaluation_error"] = warning_msg

    correct_file_path = results_dir_path / "correct.json"
    if correct_file_path.is_file():
        try:
            loaded_results["correct"] = _load_json(correct_file_path)
        except (OSError, ValueError) as exc:
            # A truncated file is a failed evaluation, not a reason to lose the
            # job: keep the default verdict and report why.
            warning_msg = _unreadable_json(correct_file_path, exc)
            logger.warning(warning_msg)
            loaded_results["evaluation_error"] = warning_msg
    else:
        loaded_results["evaluation_error"] = (
            f"Correctness file not found at {correct_file_path}"
        )

    return loaded_results


def parse_time_to_seconds(time_str: str) -> int:
    """Converts hh:mm:ss to seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time format must be hh:mm:ss")
    h, m, s = [int(p) for p in parts]
    return h * 3600 + m * 60 + s
