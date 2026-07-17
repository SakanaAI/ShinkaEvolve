"""Regression tests for sync/async (and sibling-provider) drift fixes.

Each of these bugs is a case where one copy of a duplicated twin diverged from
the other. The tests pin the copies back together:

* DeepSeek async dropped reasoning-token accounting (reported full
  ``completion_tokens`` as output and no ``thinking_tokens``) while sync
  subtracted the reasoning tokens.
* Gemini's default ``thinking_budget`` forked (sync=1024, async=0), so omitting
  the arg made sync "think" and async not.
* Gemini's backoff wrapper retried the deterministic "structured output"
  ``ValueError`` MAX_TRIES times instead of failing fast.
* Anthropic / DeepSeek raised on an unknown-model pricing lookup where
  OpenAI / local fall back to a $0 cost.
* Sync ``LLMClient.query`` hot-looped its retries with no sleep while all five
  sibling retry loops slept ~1s.
"""

import asyncio
from types import SimpleNamespace

import pytest

from google.genai import types

import shinka.llm.llm as llm_mod
from shinka.llm.llm import LLMClient
from shinka.llm.constants import MAX_RETRIES
from shinka.llm.providers.anthropic import get_anthropic_costs
from shinka.llm.providers.deepseek import (
    get_deepseek_costs,
    query_deepseek,
    query_deepseek_async,
)
from shinka.llm.providers.gemini import (
    DEFAULT_THINKING_BUDGET,
    GeminiStructuredOutputError,
    _giveup_gemini,
    query_gemini,
    query_gemini_async,
)


# ---------------------------------------------------------------------------
# DeepSeek stubs
# ---------------------------------------------------------------------------


def _deepseek_response(
    *,
    prompt_tokens,
    completion_tokens,
    reasoning_tokens,
    content="ok",
    reasoning_content="thinking",
):
    details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=details,
    )
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeDeepSeekClient:
    def __init__(self, response, *, is_async=False):
        async def _acreate(**kwargs):
            return response

        def _create(**kwargs):
            return response

        create = _acreate if is_async else _create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


# ---------------------------------------------------------------------------
# DeepSeek async reasoning-token accounting parity
# ---------------------------------------------------------------------------


def test_deepseek_async_subtracts_thinking_tokens():
    # completion_tokens includes the reasoning tokens; output must exclude them
    # and thinking_tokens must be reported (both were dropped in the async twin).
    response = _deepseek_response(
        prompt_tokens=10, completion_tokens=30, reasoning_tokens=12
    )
    client = _FakeDeepSeekClient(response, is_async=True)
    result = asyncio.run(
        query_deepseek_async(client, "deepseek-drift-test", "msg", "sys", [], None)
    )
    assert result.input_tokens == 10
    assert result.thinking_tokens == 12
    assert result.output_tokens == 18  # 30 - 12, not the raw 30


def test_deepseek_sync_and_async_report_identical_token_accounting():
    response = _deepseek_response(
        prompt_tokens=10, completion_tokens=30, reasoning_tokens=12
    )
    sync_result = query_deepseek(
        _FakeDeepSeekClient(response), "deepseek-drift-test", "msg", "sys", [], None
    )
    async_result = asyncio.run(
        query_deepseek_async(
            _FakeDeepSeekClient(response, is_async=True),
            "deepseek-drift-test",
            "msg",
            "sys",
            [],
            None,
        )
    )
    assert sync_result.output_tokens == async_result.output_tokens == 18
    assert sync_result.thinking_tokens == async_result.thinking_tokens == 12
    assert sync_result.input_tokens == async_result.input_tokens == 10


# ---------------------------------------------------------------------------
# Unknown-model pricing falls back to $0 (anthropic + deepseek)
# ---------------------------------------------------------------------------


def test_anthropic_unknown_model_pricing_defaults_to_zero():
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=20)
    )
    costs = get_anthropic_costs(response, "anthropic/not-in-catalog")
    assert costs["cost"] == 0.0
    assert costs["input_cost"] == 0.0
    assert costs["output_cost"] == 0.0
    # Token counts are still reported, only the price is zeroed.
    assert costs["input_tokens"] == 10
    assert costs["output_tokens"] == 20


def test_deepseek_unknown_model_pricing_defaults_to_zero():
    response = _deepseek_response(
        prompt_tokens=5, completion_tokens=15, reasoning_tokens=5
    )
    costs = get_deepseek_costs(response, "deepseek/not-in-catalog")
    assert costs["cost"] == 0.0
    assert costs["input_cost"] == 0.0
    assert costs["output_cost"] == 0.0
    assert costs["output_tokens"] == 10  # 15 - 5
    assert costs["thinking_tokens"] == 5


def test_query_deepseek_does_not_raise_on_unknown_model():
    # A pricing-catalog miss must not abort an otherwise-complete generation.
    response = _deepseek_response(
        prompt_tokens=5, completion_tokens=15, reasoning_tokens=5
    )
    result = query_deepseek(
        _FakeDeepSeekClient(response), "deepseek/not-in-catalog", "msg", "sys", [], None
    )
    assert result.cost == 0.0
    assert result.content == "ok"


# ---------------------------------------------------------------------------
# Gemini stubs
# ---------------------------------------------------------------------------


def _gemini_part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def _gemini_response(*, parts=None, text=None, finish_reason=types.FinishReason.STOP):
    content = SimpleNamespace(parts=parts or [_gemini_part("hi")])
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], text=text)


class _FakeGeminiClient:
    def __init__(self, response):
        self.models = SimpleNamespace(generate_content=lambda **kwargs: response)


class _FakeGeminiAsyncClient:
    def __init__(self, response):
        async def _generate(**kwargs):
            return response

        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=_generate)
        )


# ---------------------------------------------------------------------------
# Gemini default thinking_budget is identical for sync and async
# ---------------------------------------------------------------------------


def test_gemini_default_thinking_budget_is_identical_for_sync_and_async(monkeypatch):
    from shinka.llm.providers import gemini as gemini_module

    budgets = []

    def _record(thinking_budget):
        budgets.append(thinking_budget)
        return None  # thinking_config=None is accepted by GenerateContentConfig

    monkeypatch.setattr(gemini_module, "build_gemini_thinking_config", _record)

    response = _gemini_response(parts=[_gemini_part("hi")])
    query_gemini(
        _FakeGeminiClient(response), "gemini-2.5-flash", "m", "s", [], None,
        max_tokens=64,
    )
    asyncio.run(
        query_gemini_async(
            _FakeGeminiAsyncClient(response), "gemini-2.5-flash", "m", "s", [], None,
            max_tokens=64,
        )
    )

    # Both paths, given no explicit thinking_budget, use the same constant.
    assert budgets == [DEFAULT_THINKING_BUDGET, DEFAULT_THINKING_BUDGET]
    assert DEFAULT_THINKING_BUDGET == 1024


# ---------------------------------------------------------------------------
# Gemini deterministic structured-output error is not retried
# ---------------------------------------------------------------------------


def test_gemini_giveup_predicate_targets_only_structured_output():
    # The predicate must not broaden what backoff gives up on: transient errors
    # (empty/truncated responses, API errors) still retry.
    assert _giveup_gemini(GeminiStructuredOutputError("x")) is True
    assert _giveup_gemini(ValueError("Gemini response contained no text output")) is False
    assert _giveup_gemini(RuntimeError("boom")) is False


def test_gemini_structured_output_error_is_not_retried(monkeypatch):
    import time

    sleeps = []
    # backoff would sleep between retries; give-up must skip all sleeps.
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    response = _gemini_response(parts=[_gemini_part("hi")])
    with pytest.raises(ValueError, match="structured output"):
        query_gemini(
            _FakeGeminiClient(response),
            "gemini-2.5-flash",
            "m",
            "s",
            [],
            object(),  # any non-None output_model triggers the guard
        )
    assert sleeps == []  # not retried => no backoff sleep


# ---------------------------------------------------------------------------
# Sync LLMClient.query sleeps between retries (parity with the 5 siblings)
# ---------------------------------------------------------------------------


def test_sync_llm_client_query_sleeps_between_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    def _always_fails(**kwargs):
        raise RuntimeError("transient failure")

    monkeypatch.setattr(llm_mod, "query", _always_fails)

    client = LLMClient(model_names="test-model", verbose=False)
    result = client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})

    assert result is None
    # Sleeps after every failed attempt except the last => MAX_RETRIES - 1.
    assert sleeps == [1] * (MAX_RETRIES - 1)
