"""Bandit state persistence that never executes what the state file contains.

States are written as JSON. Runs saved before this format hold a pickle at the
same path; it is still read, through an unpickler that can only rebuild NumPy
arrays and scalars, so resuming never imports or calls anything else a
results directory may name.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STATE_FORMAT = "shinka-bandit-state"
STATE_VERSION = 1

# What NumPy 1.x and 2.x pickles of float arrays and scalars reference.
_LEGACY_PICKLE_GLOBALS = frozenset(
    (module, name)
    for package in ("numpy.core", "numpy._core")
    for module, name in (
        (f"{package}.multiarray", "_reconstruct"),
        (f"{package}.multiarray", "scalar"),
        (f"{package}.numeric", "_frombuffer"),
    )
) | {("numpy", "ndarray"), ("numpy", "dtype")}


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
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            # Infinity and NaN are valid state (log-space zeros, empty ranges);
            # Python's JSON dialect round-trips them.
            json.dump(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def read_bandit_state(path: Path) -> dict[str, Any]:
    """Read a bandit state written by this module or by an older pickle writer."""
    data = path.read_bytes()
    if data.lstrip()[:1] == b"{":
        return _state_from_json(data)
    state = _state_from_legacy_pickle(data)
    logger.info(
        "Read legacy pickle bandit state from %s; it is rewritten as JSON on the "
        "next save.",
        path,
    )
    return state


def _state_from_json(data: bytes) -> dict[str, Any]:
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BanditStateError(f"bandit state is not valid JSON: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("format") != STATE_FORMAT
        or document.get("version") != STATE_VERSION
        or not isinstance(document.get("state"), dict)
    ):
        raise BanditStateError("bandit state has an unknown format or version")
    return {key: _plain(key, value) for key, value in document["state"].items()}


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """Rebuilds NumPy arrays and scalars; refuses every other global."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _LEGACY_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"bandit state pickle references {module}.{name}, which is not allowed"
        )


def _state_from_legacy_pickle(data: bytes) -> dict[str, Any]:
    try:
        state = _ArrayOnlyUnpickler(io.BytesIO(data)).load()
    except Exception as exc:  # noqa: BLE001 - unpickling fails in many ways
        raise BanditStateError(f"cannot read legacy bandit state: {exc}") from exc
    if not isinstance(state, dict):
        raise BanditStateError("legacy bandit state is not a dictionary")
    return {str(key): _plain(str(key), value) for key, value in state.items()}


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
