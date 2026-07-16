import json
import os
import subprocess
import sys

import pytest

from shinka.llm.constants import (
    BACKOFF_MAX_TIME,
    BACKOFF_MAX_TRIES,
    BACKOFF_MAX_VALUE,
    OPENAI_MAX_RETRIES,
    TIMEOUT,
)
from shinka.llm.providers.anthropic import MAX_TIME as ANTHROPIC_MAX_TIME
from shinka.llm.providers.anthropic import MAX_TRIES as ANTHROPIC_MAX_TRIES
from shinka.llm.providers.anthropic import MAX_VALUE as ANTHROPIC_MAX_VALUE
from shinka.llm.providers.deepseek import MAX_TIME as DEEPSEEK_MAX_TIME
from shinka.llm.providers.deepseek import MAX_TRIES as DEEPSEEK_MAX_TRIES
from shinka.llm.providers.deepseek import MAX_VALUE as DEEPSEEK_MAX_VALUE
from shinka.llm.providers.gemini import MAX_TIME as GEMINI_MAX_TIME
from shinka.llm.providers.gemini import MAX_TRIES as GEMINI_MAX_TRIES
from shinka.llm.providers.gemini import MAX_VALUE as GEMINI_MAX_VALUE
from shinka.llm.providers.local_openai import MAX_TIME as LOCAL_OPENAI_MAX_TIME
from shinka.llm.providers.local_openai import MAX_TRIES as LOCAL_OPENAI_MAX_TRIES
from shinka.llm.providers.local_openai import MAX_VALUE as LOCAL_OPENAI_MAX_VALUE
from shinka.llm.providers.openai import MAX_TIME as OPENAI_MAX_TIME
from shinka.llm.providers.openai import MAX_TRIES as OPENAI_MAX_TRIES
from shinka.llm.providers.openai import MAX_VALUE as OPENAI_MAX_VALUE


def test_llm_backoff_max_time_tracks_timeout():
    expected = TIMEOUT * 5

    assert BACKOFF_MAX_TIME == expected
    assert OPENAI_MAX_TIME == expected
    assert LOCAL_OPENAI_MAX_TIME == expected
    assert DEEPSEEK_MAX_TIME == expected
    assert ANTHROPIC_MAX_TIME == expected
    assert GEMINI_MAX_TIME == expected


def test_llm_backoff_retry_constants_are_shared():
    assert OPENAI_MAX_TRIES == BACKOFF_MAX_TRIES
    assert LOCAL_OPENAI_MAX_TRIES == BACKOFF_MAX_TRIES
    assert DEEPSEEK_MAX_TRIES == BACKOFF_MAX_TRIES
    assert ANTHROPIC_MAX_TRIES == BACKOFF_MAX_TRIES
    assert GEMINI_MAX_TRIES == BACKOFF_MAX_TRIES

    assert OPENAI_MAX_VALUE == BACKOFF_MAX_VALUE
    assert LOCAL_OPENAI_MAX_VALUE == BACKOFF_MAX_VALUE
    assert DEEPSEEK_MAX_VALUE == BACKOFF_MAX_VALUE
    assert ANTHROPIC_MAX_VALUE == BACKOFF_MAX_VALUE
    assert GEMINI_MAX_VALUE == BACKOFF_MAX_VALUE


def test_llm_retry_env_overrides():
    env = {
        **os.environ,
        "SHINKA_LLM_TIMEOUT": "7",
        "SHINKA_LLM_MAX_RETRIES": "2",
        "SHINKA_OPENAI_MAX_RETRIES": "1",
        "SHINKA_LLM_BACKOFF_MAX_TRIES": "3",
        "SHINKA_LLM_BACKOFF_MAX_VALUE": "4",
        "SHINKA_LLM_BACKOFF_MAX_TIME": "5",
    }
    code = """
import json
from shinka.llm import constants as c
print(json.dumps({
    'timeout': c.TIMEOUT,
    'max_retries': c.MAX_RETRIES,
    'openai_max_retries': c.OPENAI_MAX_RETRIES,
    'backoff_max_tries': c.BACKOFF_MAX_TRIES,
    'backoff_max_value': c.BACKOFF_MAX_VALUE,
    'backoff_max_time': c.BACKOFF_MAX_TIME,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "timeout": 7,
        "max_retries": 2,
        "openai_max_retries": 1,
        "backoff_max_tries": 3,
        "backoff_max_value": 4,
        "backoff_max_time": 5,
    }


def test_openai_sdk_retries_default_to_single_shinka_retry_layer():
    assert OPENAI_MAX_RETRIES == 0


@pytest.mark.parametrize(
    ("name", "value", "minimum"),
    [
        ("SHINKA_LLM_TIMEOUT", "0", 1),
        ("SHINKA_LLM_MAX_RETRIES", "0", 1),
        ("SHINKA_LLM_BACKOFF_MAX_TRIES", "0", 1),
        ("SHINKA_LLM_BACKOFF_MAX_VALUE", "-1", 1),
        ("SHINKA_LLM_BACKOFF_MAX_TIME_MULTIPLIER", "0", 1),
        ("SHINKA_LLM_BACKOFF_MAX_TIME", "0", 1),
        ("SHINKA_OPENAI_MAX_RETRIES", "-1", 0),
    ],
)
def test_llm_retry_env_rejects_invalid_bounds(name: str, value: str, minimum: int):
    env = {**os.environ, name: value}
    result = subprocess.run(
        [sys.executable, "-c", "import shinka.llm.constants"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"{name} must be >= {minimum}, got {value}" in result.stderr


def test_openai_sdk_retries_allows_zero_override():
    env = {**os.environ, "SHINKA_OPENAI_MAX_RETRIES": "0"}
    code = (
        "from shinka.llm.constants import OPENAI_MAX_RETRIES; print(OPENAI_MAX_RETRIES)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0"
