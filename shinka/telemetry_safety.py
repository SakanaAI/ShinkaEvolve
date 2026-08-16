"""Shared metadata-boundary safety helpers."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_WORDS = {
    "auth",
    "authorization",
    "cookie",
    "code",
    "credential",
    "credentials",
    "embedding",
    "embeddings",
    "password",
    "secret",
    "token",
}
_SENSITIVE_COMPOUNDS = {
    "accesskey",
    "accesstoken",
    "apikey",
    "authheader",
    "authorizationheader",
    "authtoken",
    "clientsecret",
    "privatekey",
    "secretaccesskey",
    "tasksysmsg",
}


def is_sensitive_telemetry_key(value: Any) -> bool:
    """Recognize sensitive keys across snake, kebab, camel, and acronym case."""
    text = str(value).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z0-9]+", text.lower())
    joined = "".join(words)
    return bool(set(words) & _SENSITIVE_WORDS) or any(
        compound in joined for compound in _SENSITIVE_COMPOUNDS
    )
