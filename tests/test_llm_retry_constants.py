from shinka.llm.constants import (
    BACKOFF_MAX_TIME,
    BACKOFF_MAX_TRIES,
    BACKOFF_MAX_VALUE,
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
