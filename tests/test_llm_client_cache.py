"""Client memoization tests for shinka.llm.client.

Constructing an SDK client builds a fresh httpx connection pool, so rebuilding
one per query defeats HTTP keep-alive. These tests pin the memoization contract:
identical parameters reuse one client, differing parameters do not, async
clients are scoped to their event loop, and no real network is touched (API keys
come from monkeypatched env).
"""

import asyncio
import gc
import weakref

import pytest

import shinka.llm.client as llm_client
from shinka.llm.client import get_async_client_llm, get_client_llm


@pytest.fixture(autouse=True)
def _isolate_client_caches():
    """Start every test from an empty cache so identity assertions are exact."""
    llm_client._SYNC_CLIENT_CACHE.clear()
    yield
    llm_client._SYNC_CLIENT_CACHE.clear()


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


@pytest.mark.parametrize(
    ("variable", "first", "second"),
    [
        ("OPENAI_BASE_URL", "https://one.example/v1", "https://two.example/v1"),
        ("OPENAI_ORG_ID", "org-one", "org-two"),
        ("OPENAI_PROJECT_ID", "project-one", "project-two"),
    ],
)
def test_sync_openai_runtime_configuration_is_part_of_cache_key(
    monkeypatch, variable, first, second
):
    monkeypatch.setenv(variable, first)
    first_client, _, _ = get_client_llm("gpt-5-mini")

    monkeypatch.setenv(variable, second)
    second_client, _, _ = get_client_llm("gpt-5-mini")

    assert first_client is not second_client


@pytest.mark.parametrize(
    "variable",
    ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"],
)
def test_sync_anthropic_runtime_configuration_is_part_of_cache_key(
    monkeypatch, variable
):
    monkeypatch.setenv(variable, "first")
    first, _, _ = get_client_llm("claude-3-5-haiku-20241022")

    monkeypatch.setenv(variable, "second")
    second, _, _ = get_client_llm("claude-3-5-haiku-20241022")

    assert first is not second


@pytest.mark.parametrize(
    ("model_name", "credential_variables"),
    [
        ("gpt-5-mini", ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY")),
        (
            "claude-3-5-haiku-20241022",
            ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        ),
        (
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_BEARER_TOKEN_BEDROCK",
            ),
        ),
    ],
)
def test_sync_implicit_credential_sources_are_not_cached(
    monkeypatch, model_name, credential_variables
):
    for variable in credential_variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    first, _, _ = get_client_llm(model_name)
    second, _, _ = get_client_llm(model_name)

    assert first is not second


@pytest.mark.parametrize(
    ("model_name", "variable"),
    [
        ("gpt-5-mini", "OPENAI_CUSTOM_HEADERS"),
        ("claude-3-5-haiku-20241022", "ANTHROPIC_CUSTOM_HEADERS"),
        ("anthropic.claude-3-5-haiku-20241022-v1:0", "AWS_REGION"),
        (
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "ANTHROPIC_BEDROCK_BASE_URL",
        ),
    ],
)
def test_sync_sdk_runtime_configuration_is_part_of_cache_key(
    monkeypatch, model_name, variable
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setenv(variable, "first")
    first, _, _ = get_client_llm(model_name)

    monkeypatch.setenv(variable, "second")
    second, _, _ = get_client_llm(model_name)

    assert first is not second


def test_sync_bedrock_bearer_token_is_part_of_cache_key(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "first-token")
    first, _, _ = get_client_llm("anthropic.claude-3-5-haiku-20241022-v1:0")

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "second-token")
    second, _, _ = get_client_llm("anthropic.claude-3-5-haiku-20241022-v1:0")

    assert first is not second


@pytest.mark.parametrize(
    ("builder_name", "constructor_name"),
    [
        ("_build_sync_client", "AnthropicBedrock"),
        ("_build_async_client", "AsyncAnthropicBedrock"),
    ],
)
def test_bedrock_builders_forward_aws_session_token(
    monkeypatch, builder_name, constructor_name
):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "temporary-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "temporary-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "temporary-session-token")
    monkeypatch.setattr(llm_client.anthropic, constructor_name, constructor)

    resolved = llm_client.resolve_model_backend(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    client = getattr(llm_client, builder_name)("bedrock", False, resolved)

    assert client is not None
    assert captured["aws_session_token"] == "temporary-session-token"


def test_sync_vertex_project_and_location_are_part_of_cache_key(monkeypatch):
    built = []

    def build_google_genai_client(**kwargs):
        client = object()
        built.append(client)
        return client

    monkeypatch.setattr(llm_client, "build_google_genai_client", build_google_genai_client)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-one")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    first, _, _ = get_client_llm("gemini-2.5-flash")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-two")
    second, _, _ = get_client_llm("gemini-2.5-flash")

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    third, _, _ = get_client_llm("gemini-2.5-flash")

    assert len(built) == 3
    assert len({id(first), id(second), id(third)}) == 3


# --- async ----------------------------------------------------------------


def test_async_does_not_cache_client_outside_event_loop():
    first, _, _ = get_async_client_llm("gpt-5-mini")
    second, _, _ = get_async_client_llm("gpt-5-mini")

    assert first is not second


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


def test_async_loop_cache_does_not_retain_closed_event_loop():
    async def scenario():
        get_async_client_llm("gpt-5-mini")

    loop = asyncio.new_event_loop()
    loop.run_until_complete(scenario())
    loop_reference = weakref.ref(loop)
    loop.close()
    del loop
    gc.collect()

    assert loop_reference() is None


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
