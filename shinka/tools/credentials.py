"""Minimal credential helpers for Shinka.

This module provides a tiny, dependency-free way to load API keys from either:
1) Environment variables (preferred)
2) A local JSON credential store at ~/.shinka/credentials.json (optional)

The intent is to reduce workflow friction for running CLI-backed agents while
keeping backward compatibility (no required setup) and avoiding accidental key
logging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_CREDENTIALS_PATH = Path.home() / ".shinka" / "credentials.json"

# Provider -> canonical environment variable name.
# NOTE: Keep this mapping small and explicit. Callers can still pass a raw env
# var name to get_api_key() for other providers.
PROVIDER_ENV_VAR_MAP: dict[str, str] = {
    "codex": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _safe_get_str(mapping: Any, key: str) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _load_credentials(path: Path) -> dict[str, Any]:
    """Load the credentials JSON document, returning an empty dict on failure."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def get_api_key(
    provider: str, *, credentials_path: Optional[Path] = None
) -> Optional[str]:
    """Return an API key for a provider, if available.

    Resolution order:
    1) Environment variable (canonical for known providers)
    2) ~/.shinka/credentials.json if present

    Supported credential file formats (examples):
      - {"OPENAI_API_KEY": "sk-..."}
      - {"codex": "sk-..."}  (provider name as key)
      - {"providers": {"codex": {"api_key": "sk-..."}}}

    Args:
        provider: Provider name (e.g. "codex") or an env var name.
        credentials_path: Optional override for the credential file path.

    Returns:
        The API key string, or None if not found.
    """

    provider_key = (provider or "").strip()
    if not provider_key:
        return None

    provider_lower = provider_key.lower()
    env_var = PROVIDER_ENV_VAR_MAP.get(provider_lower)
    if env_var is None and provider_key.isupper() and "_" in provider_key:
        env_var = provider_key

    if env_var:
        value = os.environ.get(env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()

    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    if not path.exists():
        return None

    doc = _load_credentials(path)
    if not doc:
        return None

    # Common: store keys by env var name.
    if env_var:
        value = _safe_get_str(doc, env_var)
        if value:
            return value

    # Convenience: store keys by provider name.
    value = _safe_get_str(doc, provider_lower)
    if value:
        return value

    # Nested structure: {"providers": {"codex": {"api_key": "..."} }}
    providers = doc.get("providers")
    if isinstance(providers, dict):
        provider_section = providers.get(provider_lower)
        value = _safe_get_str(provider_section, "api_key")
        if value:
            return value

    return None
