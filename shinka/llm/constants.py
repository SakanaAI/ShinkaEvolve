import os

from shinka.env import load_shinka_dotenv

load_shinka_dotenv()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


TIMEOUT = _env_int("SHINKA_LLM_TIMEOUT", 1200)
MAX_RETRIES = _env_int("SHINKA_LLM_MAX_RETRIES", 3)
OPENAI_MAX_RETRIES = _env_int("SHINKA_OPENAI_MAX_RETRIES", 0, minimum=0)
BACKOFF_MAX_TRIES = _env_int("SHINKA_LLM_BACKOFF_MAX_TRIES", 20)
BACKOFF_MAX_VALUE = _env_int("SHINKA_LLM_BACKOFF_MAX_VALUE", 20)
BACKOFF_MAX_TIME_MULTIPLIER = _env_int("SHINKA_LLM_BACKOFF_MAX_TIME_MULTIPLIER", 5)
BACKOFF_MAX_TIME = _env_int(
    "SHINKA_LLM_BACKOFF_MAX_TIME",
    TIMEOUT * BACKOFF_MAX_TIME_MULTIPLIER,
)

GEMINI_THINKING_LEVEL_MODELS = frozenset(
    {
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-4-argon",
        "gemini-4-hydrogen",
    }
)
GEMINI_THINKING_LEVEL_BY_EFFORT = {
    "min": "LOW",
    "minimal": "LOW",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "max": "HIGH",
    "xhigh": "HIGH",
}

OPENAI_GPT6_MODELS = frozenset(
    {"gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"}
)
OPENAI_GPT6_REQUIRED_REASONING_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})
CLAUDE_ADAPTIVE_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    }
)
