"""YAML config loader for shinka_run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


_NAMESPACE_ALIASES: dict[str, str] = {
    "evo": "evo",
    "db": "db",
    "job": "job",
    "evo_config": "evo",
    "db_config": "db",
    "job_config": "job",
}

_RUNNER_INT_KEYS = ("max_evaluation_jobs", "max_proposal_jobs", "max_db_workers")
_RUNNER_BOOL_KEYS = ("verbose", "debug")


def _validate_runner_positive_int(key: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Config '{key}' must be a positive integer.")
    return value


def _validate_runner_bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Config '{key}' must be a boolean.")
    return value


def load_optional_yaml_config(
    *,
    task_dir: Path,
    config_fname: Optional[str],
    allowed_field_types: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load optional YAML config and return namespaced overrides + runner opts."""
    parsed: Dict[str, Dict[str, Any]] = {"evo": {}, "db": {}, "job": {}}
    runner: Dict[str, Any] = {}
    if not config_fname:
        return parsed, runner

    config_path = Path(config_fname)
    if not config_path.is_absolute():
        config_path = task_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return parsed, runner
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML object: {config_path}")

    allowed_top_keys = set(_NAMESPACE_ALIASES) | set(_RUNNER_INT_KEYS) | set(
        _RUNNER_BOOL_KEYS
    )
    unknown_top_keys = sorted(key for key in loaded if key not in allowed_top_keys)
    if unknown_top_keys:
        unknown = ", ".join(unknown_top_keys)
        raise ValueError(
            f"Unknown config top-level keys: {unknown}. "
            "Use evo/db/job, evo_config/db_config/job_config, "
            "or runner keys max_evaluation_jobs/max_proposal_jobs/max_db_workers/"
            "verbose/debug."
        )

    seen_ns_aliases: Dict[str, str] = {}
    for alias, namespace in _NAMESPACE_ALIASES.items():
        if alias not in loaded:
            continue
        if namespace in seen_ns_aliases:
            first = seen_ns_aliases[namespace]
            raise ValueError(
                f"Config contains both '{first}' and '{alias}'. Use only one."
            )
        seen_ns_aliases[namespace] = alias

        ns_values = loaded[alias]
        if ns_values is None:
            continue
        if not isinstance(ns_values, dict):
            raise ValueError(f"Config key '{alias}' must map to a YAML object.")

        valid_fields = allowed_field_types[namespace]
        unknown_fields = sorted(
            field_name for field_name in ns_values if field_name not in valid_fields
        )
        if unknown_fields:
            unknown = ", ".join(f"{namespace}.{field}" for field in unknown_fields)
            raise ValueError(f"Unknown config fields: {unknown}")
        parsed[namespace].update(ns_values)

    for key in _RUNNER_INT_KEYS:
        if key in loaded and loaded[key] is not None:
            runner[key] = _validate_runner_positive_int(key, loaded[key])
    for key in _RUNNER_BOOL_KEYS:
        if key in loaded and loaded[key] is not None:
            runner[key] = _validate_runner_bool(key, loaded[key])

    return parsed, runner
