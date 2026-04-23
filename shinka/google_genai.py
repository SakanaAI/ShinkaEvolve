import os
from typing import Any

from google import genai


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_ENV_VALUES


def _google_genai_timeout_ms(timeout_seconds: int) -> int:
    """Convert a second-based timeout to google-genai milliseconds."""
    return int(timeout_seconds * 1000)


def google_genai_auth_mode() -> str:
    """Return the configured Google GenAI auth mode."""
    return "vertexai" if _env_flag("GOOGLE_GENAI_USE_VERTEXAI") else "api_key"


def build_google_genai_client(timeout_ms: int | None = None) -> genai.Client:
    """Build a Google GenAI client for either direct Gemini API or Vertex AI."""
    kwargs: dict[str, Any] = {}
    if timeout_ms is not None:
        kwargs["http_options"] = genai.types.HttpOptions(timeout=timeout_ms)

    if google_genai_auth_mode() == "vertexai":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI is enabled."
            )
        if not location:
            raise ValueError(
                "GOOGLE_CLOUD_LOCATION is required when GOOGLE_GENAI_USE_VERTEXAI is enabled."
            )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            **kwargs,
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY for Gemini API mode, or set "
            "GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, and "
            "GOOGLE_CLOUD_LOCATION for Vertex AI mode."
        )
    return genai.Client(api_key=api_key, **kwargs)
