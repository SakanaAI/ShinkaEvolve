"""Bandit state persistence that never executes what the state file contains.

States are written as strict JSON: a non-finite number, which JSON has no token
for, is the string "Infinity", "-Infinity" or "NaN", and `non_finite_fields`
names the state fields written that way. Version 1 files wrote those numbers
as bare tokens, which only Python's parser accepts; they are still read.
Runs saved before the JSON format hold a pickle at the same path; its values
are still read, by `bandit_state_legacy`, without unpickling it. That reader is
scheduled for removal; its module says when.
"""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .bandit_state_legacy import read_legacy_state

logger = logging.getLogger(__name__)

STATE_FORMAT = "shinka-bandit-state"
STATE_VERSION = 2
_READABLE_VERSIONS = (1, STATE_VERSION)
_NON_FINITE = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


class BanditStateError(ValueError):
    """The bandit state file cannot be read as bandit state."""


def write_bandit_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically write a bandit state as strict JSON."""
    plain = {key: _plain(key, value) for key, value in state.items()}
    non_finite_fields = [key for key, value in plain.items() if _has_non_finite(value)]
    document = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        # Only these fields hold numbers written as strings; a list of names
        # that happens to contain "NaN" is not among them and stays as it is.
        "non_finite_fields": non_finite_fields,
        "state": {
            key: _non_finite_as_strings(value) if key in non_finite_fields else value
            for key, value in plain.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # A plain exclusive create, not a private temporary file: the state
        # keeps the permissions the umask grants, as a direct write would.
        with open(temp_path, "x", encoding="utf-8") as handle:
            # Infinity and NaN are valid state (log-space zeros, empty ranges)
            # and were turned into strings above; a bare token is refused.
            json.dump(document, handle, allow_nan=False)
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
        or document.get("version") not in _READABLE_VERSIONS
        or not isinstance(document.get("state"), dict)
    ):
        raise BanditStateError("bandit state has an unknown format or version")
    # json.loads has already turned the bare tokens of a version 1 file into
    # floats; the strings of a version 2 file follow here.
    non_finite_fields = document.get("non_finite_fields", [])
    if not isinstance(non_finite_fields, list) or not all(
        isinstance(name, str) for name in non_finite_fields
    ):
        raise BanditStateError("bandit state field list 'non_finite_fields' is invalid")
    # A set: the list is looked up once per state field.
    flagged = frozenset(non_finite_fields)
    return {
        key: _plain(key, _non_finite_from_strings(value) if key in flagged else value)
        for key, value in document["state"].items()
    }


def _state_from_legacy_pickle(data: bytes) -> dict[str, Any]:
    try:
        state = read_legacy_state(data)
    except Exception as exc:  # noqa: BLE001 - malformed pickles fail in many ways
        raise BanditStateError(f"cannot read legacy bandit state: {exc}") from exc
    return {key: _plain(key, value) for key, value in state.items()}


def _has_non_finite(value: Any) -> bool:
    items = value if isinstance(value, list) else [value]
    return any(isinstance(item, float) and not math.isfinite(item) for item in items)


def _non_finite_as_strings(value: Any) -> Any:
    """Replace the non-finite floats of a plain state value by their names."""
    if isinstance(value, list):
        return [_non_finite_as_strings(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _non_finite_from_strings(value: Any) -> Any:
    """Undo `_non_finite_as_strings`; any other string is left for `_plain`.

    A state value is a scalar or a flat list. Deeper nesting is not followed:
    `_plain` refuses it, where following it could exhaust the stack first.
    """
    if isinstance(value, list):
        return [
            _NON_FINITE.get(item, item) if isinstance(item, str) else item
            for item in value
        ]
    if isinstance(value, str):
        return _NON_FINITE.get(value, value)
    return value


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
