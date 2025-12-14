"""Codex authentication helpers (headless-friendly).

This module provides a small wrapper around the Codex CLI login flows:
- OAuth device auth (`codex login --device-auth`) for headless environments
- API key auth (`codex login --with-api-key`) for non-interactive setups

We intentionally keep this logic separate from the Codex exec wrapper so that
callers can reuse it from runners, evaluators, or any future UI endpoints.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


class CodexAuthError(RuntimeError):
    """Raised when Codex authentication cannot be established."""


def _is_interactive() -> bool:
    # Avoid hanging in non-interactive contexts (CI, background jobs).
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _status_looks_authenticated(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    # Be conservative: treat explicit "not logged in"/"unauthorized" as failure.
    if "not logged" in combined:
        return False
    if "unauthorized" in combined:
        return False
    if "please login" in combined or "please log in" in combined:
        return False
    return True


def is_codex_authenticated(codex_bin: Path) -> bool:
    """Return True if Codex CLI reports an authenticated session."""

    try:
        result = subprocess.run(
            [str(codex_bin), "login", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False

    if result.returncode != 0:
        return False
    return _status_looks_authenticated(result.stdout or "", result.stderr or "")


def _login_with_api_key(
    codex_bin: Path, api_key: str, *, timeout_seconds: int
) -> bool:
    """Attempt a non-interactive login using an API key via stdin."""

    try:
        result = subprocess.run(
            [str(codex_bin), "login", "--with-api-key"],
            input=f"{api_key}\n",
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0


def _login_device_auth(codex_bin: Path, *, timeout_seconds: int) -> bool:
    """Attempt a device auth login, inheriting stdio so the user sees the code."""

    try:
        result = subprocess.run(
            [str(codex_bin), "login", "--device-auth"],
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0


def ensure_codex_authenticated(
    codex_bin: Path,
    *,
    api_key: Optional[str] = None,
    timeout_seconds: int = 900,
    allow_interactive: Optional[bool] = None,
) -> None:
    """Ensure Codex is authenticated, attempting login flows if needed.

    Order of operations:
    1) `codex login status` (fast path)
    2) If not logged in and api_key provided, attempt `codex login --with-api-key`
    3) If still not logged in and interactive, attempt `codex login --device-auth`

    Raises:
        CodexAuthError: If authentication is not available after attempts.
    """

    if is_codex_authenticated(codex_bin):
        return

    if api_key:
        if _login_with_api_key(codex_bin, api_key, timeout_seconds=timeout_seconds):
            if is_codex_authenticated(codex_bin):
                return

    interactive = _is_interactive() if allow_interactive is None else allow_interactive
    if interactive:
        if _login_device_auth(codex_bin, timeout_seconds=timeout_seconds):
            if is_codex_authenticated(codex_bin):
                return

    raise CodexAuthError(
        "Codex authentication required. Run `codex login --device-auth` "
        "or provide an OpenAI API key via OPENAI_API_KEY / ~/.shinka/credentials.json."
    )

