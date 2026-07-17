"""Client memoization tests for shinka.llm.client.

Constructing an SDK client builds a fresh httpx connection pool, so rebuilding
one per query defeats HTTP keep-alive. These tests pin the memoization contract:
identical parameters reuse one client, differing parameters do not, async
clients are scoped to their event loop, and no real network is touched (API keys
come from monkeypatched env).
"""

import asyncio

import pytest

import shinka.llm.client as llm_client
from shinka.llm.client import get_async_client_llm, get_client_llm


@pytest.fixture(autouse=True)
def _isolate_client_caches():
    """Start every test from an empty cache so identity assertions are exact."""
    llm_client._SYNC_CLIENT_CACHE.clear()
    llm_client._ASYNC_CLIENT_CACHE.clear()
    yield
    llm_client._SYNC_CLIENT_CACHE.clear()
    llm_client._ASYNC_CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    """Fake keys so clients construct offline without hitting any provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")


# --- sync -----------------------------------------------------------------


def test_sync_identical_params_reuse_same_client():
    first, _, _ = get_client_llm("gpt-5-mini")
    second, _, _ = get_client_llm("gpt-5-mini")

    assert first is second


def test_sync_different_provider_returns_distinct_clients():
    openai_client, _, openai_provider = get_client_llm("gpt-5-mini")
    anthropic_client, _, anthropic_provider = get_client_llm(
        "claude-3-5-haiku-20241022"
    )

    assert openai_provider == "openai"
    assert anthropic_provider == "anthropic"
    assert openai_client is not anthropic_client


def test_sync_different_model_returns_distinct_clients():
    mini, _, _ = get_client_llm("gpt-5-mini")
    other, _, _ = get_client_llm("gpt-5.4-mini")

    assert mini is not other


def test_sync_structured_output_not_shared_with_plain_callers():
    plain, _, _ = get_client_llm("gpt-5-mini", structured_output=False)
    structured, _, _ = get_client_llm("gpt-5-mini", structured_output=True)

    # An instructor-wrapped client must never be handed to a plain caller.
    assert plain is not structured
    # ...but structured callers still memoize among themselves.
    structured_again, _, _ = get_client_llm("gpt-5-mini", structured_output=True)
    assert structured is structured_again


def test_sync_distinct_endpoints_are_not_shared(monkeypatch):
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "test-local-key")

    first, _, _ = get_client_llm("local/m@http://localhost:9001/v1")
    second, _, _ = get_client_llm("local/m@http://localhost:9002/v1")

    assert first is not second
    assert str(first.base_url).startswith("http://localhost:9001")
    assert str(second.base_url).startswith("http://localhost:9002")


def test_sync_distinct_api_key_sources_are_not_shared(monkeypatch):
    monkeypatch.setenv("KEY_A", "secret-a")
    monkeypatch.setenv("KEY_B", "secret-b")
    url = "http://localhost:9100/v1"

    client_a, _, _ = get_client_llm(f"local/m@{url}?api_key_env=KEY_A")
    client_b, _, _ = get_client_llm(f"local/m@{url}?api_key_env=KEY_B")

    assert client_a is not client_b
    # Same key source is still reused.
    client_a_again, _, _ = get_client_llm(f"local/m@{url}?api_key_env=KEY_A")
    assert client_a is client_a_again


# --- async ----------------------------------------------------------------


def test_async_reuses_client_outside_event_loop():
    first, _, _ = get_async_client_llm("gpt-5-mini")
    second, _, _ = get_async_client_llm("gpt-5-mini")

    assert first is second


def test_async_reuses_client_within_same_event_loop():
    async def scenario():
        first, _, _ = get_async_client_llm("gpt-5-mini")
        second, _, _ = get_async_client_llm("gpt-5-mini")
        return first is second

    assert asyncio.run(scenario())


def test_async_does_not_share_client_across_event_loops():
    async def scenario():
        client, _, _ = get_async_client_llm("gpt-5-mini")
        return client

    first = asyncio.run(scenario())
    second = asyncio.run(scenario())

    assert first is not second


def test_async_different_provider_returns_distinct_clients():
    async def scenario():
        openai_client, _, _ = get_async_client_llm("gpt-5-mini")
        anthropic_client, _, _ = get_async_client_llm("claude-3-5-haiku-20241022")
        return openai_client, anthropic_client

    openai_client, anthropic_client = asyncio.run(scenario())

    assert openai_client is not anthropic_client


def test_async_structured_output_not_shared_with_plain_callers():
    async def scenario():
        plain, _, _ = get_async_client_llm("gpt-5-mini", structured_output=False)
        structured, _, _ = get_async_client_llm("gpt-5-mini", structured_output=True)
        return plain, structured

    plain, structured = asyncio.run(scenario())

    assert plain is not structured
