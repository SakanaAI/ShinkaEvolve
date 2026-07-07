import os

from shinka.env import load_shinka_dotenv

load_shinka_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


TIMEOUT = _env_int("SHINKA_LLM_TIMEOUT", 1200)
MAX_RETRIES = _env_int("SHINKA_LLM_MAX_RETRIES", 3)
OPENAI_MAX_RETRIES = _env_int("SHINKA_OPENAI_MAX_RETRIES", 0)
BACKOFF_MAX_TRIES = _env_int("SHINKA_LLM_BACKOFF_MAX_TRIES", 20)
BACKOFF_MAX_VALUE = _env_int("SHINKA_LLM_BACKOFF_MAX_VALUE", 20)
BACKOFF_MAX_TIME_MULTIPLIER = _env_int("SHINKA_LLM_BACKOFF_MAX_TIME_MULTIPLIER", 5)
BACKOFF_MAX_TIME = _env_int(
    "SHINKA_LLM_BACKOFF_MAX_TIME",
    TIMEOUT * BACKOFF_MAX_TIME_MULTIPLIER,
)
