"""Regression tests for Gemini sync/async defaults and retry policy."""

import asyncio
from types import SimpleNamespace

from google.genai import types
import pytest

import shinka.llm.client as client_module
import shinka.llm.llm as llm_module
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers.errors import (
    NonRetryableLLMError,
    StructuredOutputNotSupportedError,
)
from shinka.llm.providers.gemini import (
    DEFAULT_THINKING_BUDGET,
    GeminiStructuredOutputError,
    _giveup_gemini,
    query_gemini,
    query_gemini_async,
)


def _response():
    part = SimpleNamespace(text="hi", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason=types.FinishReason.STOP)
    return SimpleNamespace(candidates=[candidate], text=None, usage_metadata=None)


class _SyncClient:
    def __init__(self):
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        return _response()


class _AsyncClient:
    def __init__(self):
        self.calls = 0
        self.aio = SimpleNamespace(models=self)

    async def generate_content(self, **kwargs):
        self.calls += 1
        return _response()


def test_default_thinking_budget_matches_across_sync_and_async(monkeypatch):
    from shinka.llm.providers import gemini as gemini_module

    budgets = []

    def record_budget(thinking_budget):
        budgets.append(thinking_budget)
        return None

    monkeypatch.setattr(gemini_module, "build_gemini_thinking_config", record_budget)

    query_gemini(_SyncClient(), "gemini-2.5-flash", "msg", "sys", [], None)
    asyncio.run(
        query_gemini_async(
            _AsyncClient(), "gemini-2.5-flash", "msg", "sys", [], None
        )
    )

    assert budgets == [DEFAULT_THINKING_BUDGET, DEFAULT_THINKING_BUDGET]
    assert DEFAULT_THINKING_BUDGET == 1024


def test_latest_gemini_thinking_level_config_is_unchanged(monkeypatch):
    from shinka.llm.providers import gemini as gemini_module

    captured = []

    def record_level(level):
        captured.append(level)
        return None

    monkeypatch.setattr(
        gemini_module, "build_gemini_thinking_level_config", record_level
    )

    query_gemini(
        _SyncClient(),
        "gemini-3.7-flash",
        "msg",
        "sys",
        [],
        None,
        thinking_level="HIGH",
    )

    assert captured == ["HIGH"]


def test_giveup_targets_non_retryable_errors():
    assert _giveup_gemini(GeminiStructuredOutputError("unsupported")) is True
    assert _giveup_gemini(NonRetryableLLMError("deterministic")) is True
    assert _giveup_gemini(ValueError("empty response")) is False
    assert _giveup_gemini(RuntimeError("transient")) is False


@pytest.mark.parametrize("is_async", [False, True])
def test_structured_output_error_is_not_retried(monkeypatch, is_async):
    client = _AsyncClient() if is_async else _SyncClient()
    sleeps = []

    async def record_async_sleep(seconds):
        sleeps.append(seconds)

    if is_async:
        monkeypatch.setattr(asyncio, "sleep", record_async_sleep)
    else:
        monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(StructuredOutputNotSupportedError, match="structured output"):
        if is_async:
            asyncio.run(
                query_gemini_async(
                    client,
                    "gemini-2.5-flash",
                    "msg",
                    "sys",
                    [],
                    object(),
                )
            )
        else:
            query_gemini(
                client,
                "gemini-2.5-flash",
                "msg",
                "sys",
                [],
                object(),
            )

    assert client.calls == 0
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
        client.query("msg", "sys", llm_kwargs={"model_name": "gemini-2.5-flash"})

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


@pytest.mark.parametrize("client_factory", [LLMClient, AsyncLLMClient])
def test_public_client_rejects_structured_output_before_construction(
    monkeypatch, client_factory
):
    builder_calls = 0
    sleeps = []

    def unexpected_build(**kwargs):
        nonlocal builder_calls
        builder_calls += 1
        raise AssertionError("Gemini client must not be built")

    async def record_async_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(client_module, "build_google_genai_client", unexpected_build)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(llm_module.asyncio, "sleep", record_async_sleep)
    client = client_factory(
        model_names="gemini-2.5-flash", output_model=object(), verbose=False
    )

    with pytest.raises(StructuredOutputNotSupportedError):
        result = client.query(
            "msg", "sys", llm_kwargs={"model_name": "gemini-2.5-flash"}
        )
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    assert builder_calls == 0
    assert sleeps == []


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


class _FailedAsyncResult:
    def get(self):
        raise GeminiStructuredOutputError("unsupported")


class _FailedPool:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def apply_async(self, *args, **kwargs):
        return _FailedAsyncResult()


@pytest.mark.parametrize("method_name", ["batch_query", "batch_kwargs_query"])
def test_sync_public_batch_propagates_structured_output_error(
    monkeypatch, method_name
):
    monkeypatch.setattr(llm_module.mp, "Pool", lambda **kwargs: _FailedPool())
    client = LLMClient(model_names="gemini-2.5-flash", verbose=False)
    method = getattr(client, method_name)
    kwargs = {"num_samples": 1, "msg": "msg", "system_msg": "sys"}
    if method_name == "batch_query":
        kwargs["llm_kwargs"] = [{"model_name": "gemini-2.5-flash"}]

    with pytest.raises(GeminiStructuredOutputError):
        method(**kwargs)


@pytest.mark.parametrize("method_name", ["batch_query", "batch_kwargs_query"])
def test_async_public_batch_propagates_structured_output_error(
    monkeypatch, method_name
):
    client = AsyncLLMClient(model_names="gemini-2.5-flash", verbose=False)
    helper_name = (
        "_query_async_with_retry"
        if method_name == "batch_query"
        else "_sample_kwargs_query_async_with_retry"
    )

    async def unsupported(*args, **kwargs):
        raise GeminiStructuredOutputError("unsupported")

    monkeypatch.setattr(client, helper_name, unsupported)
    method = getattr(client, method_name)
    kwargs = {"num_samples": 1, "msg": "msg", "system_msg": "sys"}
    if method_name == "batch_query":
        kwargs["llm_kwargs"] = [{"model_name": "gemini-2.5-flash"}]

    with pytest.raises(GeminiStructuredOutputError):
        asyncio.run(method(**kwargs))


@pytest.mark.parametrize("client_factory", [LLMClient, AsyncLLMClient])
@pytest.mark.parametrize("method_name", ["batch_query", "batch_kwargs_query"])
def test_public_batch_rejects_gemini_before_start(
    monkeypatch, client_factory, method_name
):
    work_started = 0

    def unexpected_pool(**kwargs):
        nonlocal work_started
        work_started += 1
        raise AssertionError("batch work must not start")

    async def unexpected_helper(*args, **kwargs):
        nonlocal work_started
        work_started += 1
        raise AssertionError("batch work must not start")

    monkeypatch.setattr(llm_module.mp, "Pool", unexpected_pool)
    client = client_factory(
        model_names="gemini-2.5-flash", output_model=object(), verbose=False
    )
    helper_name = (
        "_query_async_with_retry"
        if method_name == "batch_query"
        else "_sample_kwargs_query_async_with_retry"
    )
    if isinstance(client, AsyncLLMClient):
        monkeypatch.setattr(client, helper_name, unexpected_helper)

    method = getattr(client, method_name)
    kwargs = {"num_samples": 1, "msg": "msg", "system_msg": "sys"}
    if method_name == "batch_query":
        kwargs["llm_kwargs"] = [{"model_name": "gemini-2.5-flash"}]

    with pytest.raises(GeminiStructuredOutputError):
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    assert work_started == 0
