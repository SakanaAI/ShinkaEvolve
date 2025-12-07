"""Registry for tracking live Codex CLI sessions and their OS PIDs."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_DIR = Path.home() / ".codex" / "shinka_sessions"


def _ensure_registry_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _entry_path(key: str | int) -> Path:
    _ensure_registry_dir()
    return REGISTRY_DIR / f"{key}.json"


def register_session_process(
    pid: int,
    *,
    prompt_preview: str,
    workdir: Path,
    session_kind: str = "unknown",
    parent_id: Optional[str] = None,
    generation: Optional[int] = None,
    patch_type: Optional[str] = None,
    results_dir: Optional[str] = None,
    filename_key: Optional[str] = None,
) -> None:
    """Persist minimal metadata about a newly spawned Codex CLI process.
    
    Args:
        pid: The OS process ID to check for liveness.
        results_dir: The run's results directory (for matching sessions to runs).
        filename_key: Optional unique string for the filename. Defaults to str(pid).
                      Use this if multiple sessions might share the same PID (e.g. threads).
    """

    entry = {
        "pid": pid,
        "prompt_preview": prompt_preview.strip(),
        "workdir": str(workdir),
        "started_at": time.time(),
        "session_kind": session_kind,
        "session_id": None,
        "status": "running",
        "parent_id": parent_id,
        "generation": generation,
        "patch_type": patch_type,
        "results_dir": results_dir,
    }
    
    key = filename_key if filename_key else pid
    _entry_path(key).write_text(json.dumps(entry), encoding="utf-8")


def update_session_process(pid: int, filename_key: Optional[str] = None, **updates: Any) -> None:
    """Merge updates into an existing registry entry.
    
    Args:
        pid: Legacy argument, used as key if filename_key is None.
        filename_key: The specific file key to update.
    """
    key = filename_key if filename_key else pid
    path = _entry_path(key)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


def remove_session_process(pid: int, filename_key: Optional[str] = None) -> None:
    """Remove an entry once the Codex process exits."""
    key = filename_key if filename_key else pid
    path = _entry_path(key)
    if path.exists():
        path.unlink(missing_ok=True)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError: 
        # Handle case where pid is invalid (e.g. 0 or negative if passed incorrectly)
        return False
    else:
        return True


def list_session_processes() -> List[Dict[str, Any]]:
    """Return sanitized entries for still-running Codex processes."""

    entries: List[Dict[str, Any]] = []
    if not REGISTRY_DIR.exists():
        return entries

    for json_file in REGISTRY_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            json_file.unlink(missing_ok=True)
            continue

        pid = data.get("pid")
        if not isinstance(pid, int):
            json_file.unlink(missing_ok=True)
            continue

        if not _is_pid_alive(pid):
            json_file.unlink(missing_ok=True)
            continue

        entries.append(
            {
                "pid": pid,
                "session_id": data.get("session_id"),
                "prompt_preview": data.get("prompt_preview"),
                "workdir": data.get("workdir"),
                "started_at": data.get("started_at"),
                "session_kind": data.get("session_kind"),
                "status": data.get("status", "running"),
                "parent_id": data.get("parent_id"),
                "generation": data.get("generation"),
                "patch_type": data.get("patch_type"),
                "results_dir": data.get("results_dir"),
                "can_stop": True,
            }
        )
    return entries


def terminate_session_process(pid: int, sig: signal.Signals = signal.SIGTERM) -> None:
    """Send a termination signal to a tracked Codex process."""

    os.kill(pid, sig)
