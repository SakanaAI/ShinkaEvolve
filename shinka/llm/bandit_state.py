"""Bandit state persistence that never executes what the state file contains.

States are written as JSON. Runs saved before this format hold a pickle at the
same path; its values are still read, by `bandit_state_legacy`, without
unpickling it. That reader is scheduled for removal; its module says when.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .bandit_state_legacy import read_legacy_state

logger = logging.getLogger(__name__)

STATE_FORMAT = "shinka-bandit-state"
STATE_VERSION = 1

class BanditStateError(ValueError):
    """The bandit state file cannot be read as bandit state."""


def write_bandit_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically write a bandit state as JSON."""
    document = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "state": {key: _plain(key, value) for key, value in state.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # A plain exclusive create, not a private temporary file: the state
        # keeps the permissions the umask grants, as a direct write would.
        with open(temp_path, "x", encoding="utf-8") as handle:
            # Infinity and NaN are valid state (log-space zeros, empty ranges);
            # Python's JSON dialect round-trips them.
            json.dump(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temp_path, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_bandit_state(path: Path) -> dict[str, Any]:
    """Read a bandit state written by this module or by an older pickle writer."""
    data = path.read_bytes()
    if data.lstrip()[:1] == b"{":
        return _state_from_json(data)
    state = _state_from_legacy_pickle(data)
    logger.warning(
        "Bandit state %s is a pickle from an older Shinka version. It was read "
        "and is rewritten as JSON on the next save; a later release stops "
        "reading pickle states.",
        path,
    )
    return state


def _state_from_json(data: bytes) -> dict[str, Any]:
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BanditStateError(f"bandit state is not valid JSON: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("format") != STATE_FORMAT
        or document.get("version") != STATE_VERSION
        or not isinstance(document.get("state"), dict)
    ):
        raise BanditStateError("bandit state has an unknown format or version")
    return {key: _plain(key, value) for key, value in document["state"].items()}


def _state_from_legacy_pickle(data: bytes) -> dict[str, Any]:
    try:
        state = read_legacy_state(data)
    except Exception as exc:  # noqa: BLE001 - malformed pickles fail in many ways
        raise BanditStateError(f"cannot read legacy bandit state: {exc}") from exc
    return {key: _plain(key, value) for key, value in state.items()}


def _plain(key: str, value: Any) -> Any:
    """Reduce a state value to None, a number, or a flat list of numbers or names."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        if all(isinstance(item, str) for item in items) or all(
            isinstance(item, (bool, int, float)) for item in items
        ):
            return items
    raise BanditStateError(f"bandit state field '{key}' holds unsupported data")
