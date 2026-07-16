from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _sha256_bytes(path.read_bytes())


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    return output[0] if output else f"exit {completed.returncode}"


def ensure_wandb_run_id(results_dir: Path, configured_id: str | None) -> str:
    path = results_dir / ".wandb_run_id"
    if configured_id:
        run_id = configured_id
    elif path.is_file() and path.read_text(encoding="utf-8").strip():
        run_id = path.read_text(encoding="utf-8").strip()
    else:
        run_id = uuid.uuid4().hex
    path.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def write_run_manifest(
    *,
    evo_config: Any,
    db_config: Any,
    job_config: Any,
    results_dir: Path,
    effective_workers: dict[str, int],
    minimum_request_demand: dict[str, int],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    repo_root_text = _command_output(["git", "rev-parse", "--show-toplevel"])
    repo_root = Path(repo_root_text) if repo_root_text else Path.cwd()
    git_commit = _command_output(["git", "rev-parse", "HEAD"], repo_root)
    dirty_status = _command_output(["git", "status", "--porcelain=v1"], repo_root) or ""
    dirty_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    ).stdout

    config_payload = {
        "evo": _jsonable(evo_config),
        "db": _jsonable(db_config),
        "job": _jsonable(job_config),
    }
    eval_path = Path(getattr(job_config, "eval_program_path", ""))
    if not eval_path.is_absolute():
        eval_path = Path.cwd() / eval_path

    headless_command = shlex.split(
        __import__("os").environ.get(
            "SHINKA_HEADLESS_COMMAND",
            "npx -y @roberttlange/headless",
        )
    )
    agents = {
        str(model).split("headless/", 1)[1].split("@", 1)[0].split("?", 1)[0]
        for model in getattr(evo_config, "llm_models", [])
        if str(model).startswith("headless/")
    }
    native_commands = {
        "antigravity": ["agy", "--version"],
        "cursor": ["agent", "--version"],
        "codex": ["codex", "--version"],
    }

    manifest = {
        "schema_version": "shinka-run-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": {
            "repo_root": str(repo_root),
            "git_commit": git_commit,
            "dirty": bool(dirty_status),
            "dirty_status_sha256": _sha256_bytes(dirty_status.encode()),
            "dirty_diff_sha256": _sha256_bytes(dirty_diff),
        },
        "config": config_payload,
        "config_sha256": _sha256_bytes(
            json.dumps(config_payload, sort_keys=True).encode()
        ),
        "evaluator": {
            "path": str(eval_path),
            "sha256": _sha256_file(eval_path),
        },
        "providers": {
            "headless": _command_output([*headless_command, "--version"]),
            "native_cli_versions": {
                agent: _command_output(native_commands[agent])
                for agent in sorted(agents)
                if agent in native_commands
            },
        },
        "proposal_timeouts": {
            "default_seconds": getattr(
                evo_config, "headless_proposal_timeout_seconds", None
            ),
            "cleanup_grace_seconds": getattr(
                evo_config, "headless_cleanup_grace_seconds", None
            ),
            "model_overrides": getattr(evo_config, "headless_model_timeouts", {}),
        },
        "quotas": {
            "rate_limits": getattr(evo_config, "llm_rate_limits", {}),
            "daily_quotas": getattr(evo_config, "llm_daily_quotas", {}),
            "minimum_request_demand": minimum_request_demand,
        },
        "effective_workers": effective_workers,
        "wandb": {
            "run_id": getattr(evo_config, "wandb_run_id", None),
            "resume": getattr(evo_config, "wandb_resume", None),
        },
    }
    path = results_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
