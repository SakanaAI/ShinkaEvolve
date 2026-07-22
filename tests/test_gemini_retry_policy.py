"""Regression tests for Gemini sync/async defaults and retry policy."""

import asyncio
from types import SimpleNamespace

from google.genai import types
import pytest

import shinka.llm.client as client_module
import shinka.llm.llm as llm_module
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers.gemini import (
    DEFAULT_THINKING_BUDGET,
    GeminiStructuredOutputError,
    _giveup_gemini,
    query_gemini,
    query_gemini_async,
)


def _part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def _response():
    content = SimpleNamespace(parts=[_part("hi")])
    candidate = SimpleNamespace(content=content, finish_reason=types.FinishReason.STOP)
    return SimpleNamespace(candidates=[candidate], text=None)


class _SyncClient:
    def __init__(self, response):
        self.models = SimpleNamespace(generate_content=lambda **kwargs: response)


class _AsyncClient:
    def __init__(self, response):
        async def generate(**kwargs):
            return response

        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate))


def test_default_thinking_budget_matches_across_sync_and_async(monkeypatch):
    from shinka.llm.providers import gemini as gemini_module

    budgets = []

    def record_budget(thinking_budget):
        budgets.append(thinking_budget)
        return None

    monkeypatch.setattr(gemini_module, "build_gemini_thinking_config", record_budget)
    response = _response()

    query_gemini(
        _SyncClient(response), "gemini-2.5-flash", "msg", "sys", [], None
    )
    asyncio.run(
        query_gemini_async(
            _AsyncClient(response), "gemini-2.5-flash", "msg", "sys", [], None
        )
    )

    assert budgets == [DEFAULT_THINKING_BUDGET, DEFAULT_THINKING_BUDGET]
    assert DEFAULT_THINKING_BUDGET == 1024


def test_giveup_targets_only_unsupported_structured_output():
    assert _giveup_gemini(GeminiStructuredOutputError("unsupported")) is True
    assert _giveup_gemini(ValueError("empty response")) is False
    assert _giveup_gemini(RuntimeError("transient")) is False


def test_structured_output_error_is_not_retried(monkeypatch):
    import time

    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(ValueError, match="structured output"):
        query_gemini(
            _SyncClient(_response()),
            "gemini-2.5-flash",
            "msg",
            "sys",
            [],
            object(),
        )

    assert sleeps == []


def test_sync_llm_client_does_not_retry_structured_output_error(monkeypatch):
    calls = 0
    sleeps = []

    def unsupported(**kwargs):
        nonlocal calls
        calls += 1
        raise GeminiStructuredOutputError("unsupported")

    monkeypatch.setattr(llm_module, "query", unsupported)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = LLMClient(model_names="gemini-2.5-flash", verbose=False)

    with pytest.raises(GeminiStructuredOutputError):
        client.query(
            "msg", "sys", llm_kwargs={"model_name": "gemini-2.5-flash"}
        )

    assert calls == 1
    assert sleeps == []


def test_async_llm_client_does_not_retry_structured_output_error(monkeypatch):
    calls = 0
    sleeps = []

    async def unsupported(**kwargs):
        nonlocal calls
        calls += 1
        raise GeminiStructuredOutputError("unsupported")

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_module, "query_async", unsupported)
    monkeypatch.setattr(llm_module.asyncio, "sleep", record_sleep)
    client = AsyncLLMClient(model_names="gemini-2.5-flash", verbose=False)

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(
            client.query(
                "msg", "sys", llm_kwargs={"model_name": "gemini-2.5-flash"}
            )
        )

    assert calls == 1
    assert sleeps == []


def test_async_gemini_client_construction_uses_non_retryable_error(monkeypatch):
    monkeypatch.setattr(
        client_module, "build_google_genai_client", lambda **kwargs: object()
    )

    with pytest.raises(GeminiStructuredOutputError):
        client_module.get_async_client_llm(
            "gemini-2.5-flash", structured_output=True
        )


@pytest.mark.parametrize("helper_name", ["query_fn", "sample_kwargs_query_fn"])
def test_sync_batch_helpers_do_not_retry_structured_output_error(
    monkeypatch, helper_name
):
    calls = 0
    sleeps = []

    def unsupported(**kwargs):
        nonlocal calls
        calls += 1
        raise GeminiStructuredOutputError("unsupported")

    monkeypatch.setattr(llm_module, "query", unsupported)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    helper = getattr(llm_module, helper_name)
    helper_kwargs = {
        "idx": 0,
        "msg": "msg",
        "system_msg": "sys",
        "output_model": object(),
    }
    if helper_name == "query_fn":
        helper_kwargs["kwargs"] = {"model_name": "gemini-2.5-flash"}
    else:
        helper_kwargs["model_names"] = "gemini-2.5-flash"

    with pytest.raises(GeminiStructuredOutputError):
        helper(**helper_kwargs)

    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "helper_name",
    ["_query_async_with_retry", "_sample_kwargs_query_async_with_retry"],
)
def test_async_batch_helpers_do_not_retry_structured_output_error(
    monkeypatch, helper_name
):
    calls = 0
    sleeps = []

    async def unsupported(**kwargs):
        nonlocal calls
        calls += 1
        raise GeminiStructuredOutputError("unsupported")

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_module, "query_async", unsupported)
    monkeypatch.setattr(llm_module.asyncio, "sleep", record_sleep)
    client = AsyncLLMClient(
        model_names="gemini-2.5-flash",
        output_model=object(),
        verbose=False,
    )
    helper = getattr(client, helper_name)
    helper_kwargs = {"idx": 0, "msg": "msg", "system_msg": "sys"}
    if helper_name == "_query_async_with_retry":
        helper_kwargs["kwargs"] = {"model_name": "gemini-2.5-flash"}

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(helper(**helper_kwargs))

    assert calls == 1
    assert sleeps == []
