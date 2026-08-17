"""Provider client-cache safety and lifecycle regressions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import multiprocessing as mp
import threading
import time
from typing import Any
import weakref

import instructor
import pytest

import shinka.llm.client as llm_client
from shinka.core.async_runner import ShinkaEvolveRunner


class _SyncTransport:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _AsyncTransport:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _GoogleAsyncView:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _GoogleTransport:
    def __init__(self) -> None:
        self.aio = _GoogleAsyncView()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _stable_test_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    llm_client.close_sync_client_cache()
    yield
    llm_client.close_sync_client_cache()


def test_sync_reuses_constructor_for_different_models(monkeypatch: pytest.MonkeyPatch):
    built: list[object] = []

    def build(*_args: Any) -> object:
        client = object()
        built.append(client)
        return client

    monkeypatch.setattr(llm_client, "_build_sync_client", build)

    first, first_model, _ = llm_client.get_client_llm("gpt-5-mini")
    second, second_model, _ = llm_client.get_client_llm("gpt-5.4-mini")

    assert first is second
    assert first_model == "gpt-5-mini"
    assert second_model == "gpt-5.4-mini"
    assert built == [first]


def test_sync_separates_plain_and_structured_clients(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    plain, _, _ = llm_client.get_client_llm("gpt-5-mini")
    structured, _, _ = llm_client.get_client_llm(
        "gpt-5-mini", structured_output=True
    )
    structured_again, _, _ = llm_client.get_client_llm(
        "gpt-5.4-mini", structured_output=True
    )

    assert plain is not structured
    assert structured is structured_again


@pytest.mark.parametrize(
    ("variable", "first", "second"),
    [
        ("OPENAI_API_KEY", "key-one", "key-two"),
        ("OPENAI_BASE_URL", "https://one.example/v1", "https://two.example/v1"),
        ("OPENAI_ORG_ID", "org-one", "org-two"),
        ("OPENAI_PROJECT_ID", "project-one", "project-two"),
    ],
)
def test_sync_separates_openai_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    first: str,
    second: str,
):
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setenv(variable, first)
    first_client, _, _ = llm_client.get_client_llm("gpt-5-mini")

    monkeypatch.setenv(variable, second)
    second_client, _, _ = llm_client.get_client_llm("gpt-5-mini")

    assert first_client is not second_client


def test_cache_keys_do_not_retain_raw_secrets_or_endpoint_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "raw-local-secret"
    endpoint_password = "endpoint-password"
    query_token = "query-token"
    monkeypatch.setenv("LOCAL_SECRET", secret)
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    llm_client.get_client_llm(
        "local/model@https://user:"
        f"{endpoint_password}@example.test/v1?token={query_token}"
        "&api_key_env=LOCAL_SECRET"
    )

    cache_representation = repr(llm_client._SYNC_CLIENT_CACHE)
    assert secret not in cache_representation
    assert endpoint_password not in cache_representation
    assert query_token not in cache_representation


@pytest.mark.parametrize(
    ("model_name", "credential_variables"),
    [
        ("gpt-5-mini", ("OPENAI_API_KEY",)),
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
def test_implicit_credentials_bypass_sync_cache(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    credential_variables: tuple[str, ...],
):
    for variable in credential_variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    first, _, _ = llm_client.get_client_llm(model_name)
    second, _, _ = llm_client.get_client_llm(model_name)

    assert first is not second


def test_vertex_implicit_credentials_bypass_sync_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    first, _, _ = llm_client.get_client_llm("gemini-2.5-flash")
    second, _, _ = llm_client.get_client_llm("gemini-2.5-flash")

    assert first is not second


def test_bedrock_session_token_is_forwarded_and_separates_clients(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[dict[str, Any]] = []

    def constructor(**kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "temporary-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "temporary-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-one")
    monkeypatch.setattr(llm_client.anthropic, "AnthropicBedrock", constructor)
    first, _, _ = llm_client.get_client_llm(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-two")
    second, _, _ = llm_client.get_client_llm(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    assert first is not second
    assert [call["aws_session_token"] for call in captured] == [
        "session-one",
        "session-two",
    ]


def test_async_bedrock_builder_forwards_session_token(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def constructor(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "temporary-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "temporary-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "temporary-session")
    monkeypatch.setattr(llm_client.anthropic, "AsyncAnthropicBedrock", constructor)
    resolved = llm_client.resolve_model_backend(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    spec = llm_client._resolve_client_spec(
        "bedrock", False, resolved, llm_client._async_constructors()
    )

    llm_client._build_async_client(spec)

    assert captured["aws_session_token"] == "temporary-session"


def test_local_endpoint_and_key_source_separate_clients(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KEY_A", "secret-a")
    monkeypatch.setenv("KEY_B", "secret-b")
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    first, _, _ = llm_client.get_client_llm(
        "local/model@http://localhost:9001/v1?api_key_env=KEY_A"
    )
    second, _, _ = llm_client.get_client_llm(
        "local/model@http://localhost:9002/v1?api_key_env=KEY_A"
    )
    third, _, _ = llm_client.get_client_llm(
        "local/model@http://localhost:9001/v1?api_key_env=KEY_B"
    )

    assert len({id(first), id(second), id(third)}) == 3


def test_gemini_api_key_rotation_separates_clients(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-one")
    first, _, _ = llm_client.get_client_llm("gemini-2.5-flash")

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-two")
    second, _, _ = llm_client.get_client_llm("gemini-2.5-flash")

    assert first is not second


def test_sync_concurrent_callers_construct_once(monkeypatch: pytest.MonkeyPatch):
    construction_count = 0
    count_lock = threading.Lock()

    def build(*_args: Any) -> object:
        nonlocal construction_count
        with count_lock:
            construction_count += 1
        time.sleep(0.03)
        return object()

    monkeypatch.setattr(llm_client, "_build_sync_client", build)

    with ThreadPoolExecutor(max_workers=8) as executor:
        clients = list(
            executor.map(lambda _: llm_client.get_client_llm("gpt-5-mini")[0], range(8))
        )

    assert construction_count == 1
    assert all(client is clients[0] for client in clients)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
def test_forked_child_does_not_reuse_parent_client(monkeypatch: pytest.MonkeyPatch):
    build_count = 0

    def build(*_args: Any) -> int:
        nonlocal build_count
        build_count += 1
        return build_count

    monkeypatch.setattr(llm_client, "_build_sync_client", build)
    parent_client, _, _ = llm_client.get_client_llm("gpt-5-mini")
    context = mp.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)

    def child_target() -> None:
        child_client, _, _ = llm_client.get_client_llm("gpt-5-mini")
        child_connection.send(child_client == parent_client)
        child_connection.close()

    process = context.Process(target=child_target)
    process.start()
    child_connection.close()
    try:
        assert parent_connection.poll(10), "forked child did not report cache state"
        child_reused_parent = parent_connection.recv()
        process.join(timeout=10)
        assert not process.is_alive(), "forked child did not exit"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent_connection.close()

    assert process.exitcode == 0
    assert child_reused_parent is False


def test_close_sync_cache_unwraps_instructor_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
):
    transport = _SyncTransport()
    wrapper = instructor.Instructor(client=transport, create=lambda **_kwargs: None)
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: wrapper)
    llm_client.get_client_llm("gpt-5-mini", structured_output=True)

    llm_client.close_sync_client_cache()
    llm_client.close_sync_client_cache()

    assert transport.close_calls == 1


def test_async_client_is_reused_only_inside_its_event_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: object())

    outside_first, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
    outside_second, _, _ = llm_client.get_async_client_llm("gpt-5-mini")

    async def same_loop() -> bool:
        first, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
        second, _, _ = llm_client.get_async_client_llm("gpt-5.4-mini")
        return first is second

    async def one_client() -> object:
        client, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
        return client

    first_loop_client = asyncio.run(one_client())
    second_loop_client = asyncio.run(one_client())

    assert outside_first is not outside_second
    assert asyncio.run(same_loop())
    assert first_loop_client is not second_loop_client


def test_async_structured_clients_reuse_separately(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: object())

    async def scenario() -> tuple[object, object, object]:
        plain, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
        structured, _, _ = llm_client.get_async_client_llm(
            "gpt-5-mini", structured_output=True
        )
        structured_again, _, _ = llm_client.get_async_client_llm(
            "gpt-5.4-mini", structured_output=True
        )
        return plain, structured, structured_again

    plain, structured, structured_again = asyncio.run(scenario())

    assert plain is not structured
    assert structured is structured_again


def test_async_loop_cache_does_not_retain_closed_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: object())

    async def scenario() -> None:
        llm_client.get_async_client_llm("gpt-5-mini")

    loop = asyncio.new_event_loop()
    loop.run_until_complete(scenario())
    loop_reference = weakref.ref(loop)
    loop.close()
    del loop
    gc.collect()

    assert loop_reference() is None


def test_close_async_cache_awaits_transport_and_clears_current_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    transports: list[_AsyncTransport] = []

    def build(*_args: Any) -> _AsyncTransport:
        transport = _AsyncTransport()
        transports.append(transport)
        return transport

    monkeypatch.setattr(llm_client, "_build_async_client", build)

    async def scenario() -> tuple[object, object]:
        first, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
        await llm_client.close_async_client_cache()
        await llm_client.close_async_client_cache()
        second, _, _ = llm_client.get_async_client_llm("gpt-5-mini")
        return first, second

    first, second = asyncio.run(scenario())

    assert transports[0].close_calls == 1
    assert transports[1].close_calls == 0
    assert first is not second


def test_close_async_google_cache_closes_async_and_sync_transports(
    monkeypatch: pytest.MonkeyPatch,
):
    transport = _GoogleTransport()
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: transport)

    async def scenario() -> None:
        llm_client.get_async_client_llm("gemini-2.5-flash")
        await llm_client.close_async_client_cache()

    asyncio.run(scenario())

    assert transport.aio.close_calls == 1
    assert transport.close_calls == 1


@pytest.mark.parametrize("run_fails", [False, True])
def test_sync_runner_closes_its_owned_async_cache(
    monkeypatch: pytest.MonkeyPatch, run_fails: bool
):
    events: list[str] = []
    runner = object.__new__(ShinkaEvolveRunner)

    async def run_body() -> None:
        events.append("run")
        if run_fails:
            raise RuntimeError("run failed")

    async def close_cache() -> None:
        events.append("close")

    monkeypatch.setattr(runner, "_run_async", run_body)
    monkeypatch.setattr(llm_client, "close_async_client_cache", close_cache)

    if run_fails:
        with pytest.raises(RuntimeError, match="run failed"):
            runner.run()
    else:
        runner.run()

    assert events == ["run", "close"]
