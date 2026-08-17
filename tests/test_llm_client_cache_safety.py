"""Adversarial cache identity and shutdown regressions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

import shinka.llm.client as llm_client
from shinka.core.async_runner import ShinkaEvolveRunner


def test_constructor_uses_same_credential_snapshot_as_cache_key(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_keys: list[str] = []

    def constructor(**kwargs: Any) -> object:
        captured_keys.append(kwargs["api_key"])
        monkeypatch.setenv("OPENAI_API_KEY", "rotated-during-construction")
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "snapshotted-key")
    monkeypatch.setattr(llm_client.openai, "OpenAI", constructor)

    first, _, _ = llm_client.get_client_llm("gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "snapshotted-key")
    second, _, _ = llm_client.get_client_llm("gpt-5-mini")

    assert captured_keys == ["snapshotted-key"]
    assert first is second


def test_bedrock_constructor_uses_snapshotted_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_base_urls: list[str] = []

    def constructor(**kwargs: Any) -> object:
        captured_base_urls.append(kwargs["base_url"])
        monkeypatch.setenv(
            "ANTHROPIC_BEDROCK_BASE_URL", "https://rotated.example.test"
        )
        return object()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_REGION_NAME", "eu-central-1")
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    monkeypatch.setattr(llm_client.anthropic, "AnthropicBedrock", constructor)

    first, _, _ = llm_client.get_client_llm(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL")
    second, _, _ = llm_client.get_client_llm(
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    assert captured_base_urls == [
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
    ]
    assert first is second


@pytest.mark.parametrize("getter", ["get_client_llm", "get_async_client_llm"])
def test_gemini_structured_output_fails_before_client_construction(
    monkeypatch: pytest.MonkeyPatch, getter: str
):
    constructed = False

    def build(*_args: Any) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(
        llm_client,
        "_build_sync_client" if getter == "get_client_llm" else "_build_async_client",
        build,
    )

    with pytest.raises(
        llm_client.StructuredOutputNotSupportedError,
        match="Gemini does not support structured output",
    ):
        getattr(llm_client, getter)("gemini-2.5-flash", structured_output=True)

    assert constructed is False


@pytest.mark.parametrize(
    ("variable", "first_value", "second_value"),
    [
        ("HTTP_PROXY", "http://proxy-one", "http://proxy-two"),
        ("HTTPS_PROXY", "http://proxy-one", "http://proxy-two"),
        ("ALL_PROXY", "http://proxy-one", "http://proxy-two"),
        ("NO_PROXY", "one.example", "two.example"),
        ("SSL_CERT_FILE", "/tmp/cert-one", "/tmp/cert-two"),
        ("SSL_CERT_DIR", "/tmp/certs-one", "/tmp/certs-two"),
        ("SHINKA_GOOGLE_GENAI_IP_FAMILY", "system", "ipv4"),
        ("SHINKA_GOOGLE_GENAI_IP_PROBE_TIMEOUT", "0.1", "0.2"),
    ],
)
def test_transport_environment_rotation_separates_clients(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    first_value: str,
    second_value: str,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setenv(variable, first_value)
    first, _, _ = llm_client.get_client_llm("gemini-2.5-flash")

    monkeypatch.setenv(variable, second_value)
    second, _, _ = llm_client.get_client_llm("gemini-2.5-flash")

    assert first is not second


def test_runtime_rotation_prunes_superseded_sync_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())

    for index in range(20):
        monkeypatch.setenv("OPENAI_API_KEY", f"rotated-key-{index}")
        llm_client.get_client_llm("gpt-5-mini")

    openai_keys = [
        key for key in llm_client._SYNC_CLIENT_CACHE if key[:2] == ("openai", False)
    ]
    assert len(openai_keys) == 1


def test_runtime_rotation_prunes_superseded_async_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: object())

    async def scenario() -> int:
        for index in range(20):
            monkeypatch.setenv("OPENAI_API_KEY", f"rotated-key-{index}")
            llm_client.get_async_client_llm("gpt-5-mini")
        loop = asyncio.get_running_loop()
        cache = getattr(loop, llm_client._ASYNC_CACHE_ATTRIBUTE)
        return len([key for key in cache if key[:2] == ("openai", False)])

    assert asyncio.run(scenario()) == 1


def test_gemini_cache_hit_reapplies_process_network_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    applied_policies: list[str] = []

    def configure_network() -> str:
        applied_policies.append("applied")
        return "system"

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "system")
    monkeypatch.setattr(llm_client, "_build_sync_client", lambda *_args: object())
    monkeypatch.setattr(
        llm_client,
        "configure_google_genai_network",
        configure_network,
    )

    llm_client.get_client_llm("gemini-2.5-flash")
    llm_client.get_client_llm("gemini-2.5-flash")

    assert applied_policies == ["applied"]


def test_sync_close_failure_does_not_log_exception_secret(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    secret = "endpoint-query-secret"

    class FailingTransport:
        def close(self) -> None:
            raise RuntimeError(f"https://example.test?v={secret}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        llm_client, "_build_sync_client", lambda *_args: FailingTransport()
    )
    llm_client.get_client_llm("gpt-5-mini")

    with caplog.at_level(logging.WARNING):
        llm_client.close_sync_client_cache()

    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text


def test_google_sync_close_runs_when_async_close_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class AsyncView:
        async def aclose(self) -> None:
            raise RuntimeError("async close failed")

    class GoogleTransport:
        def __init__(self) -> None:
            self.aio = AsyncView()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    transport = GoogleTransport()
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(llm_client, "_build_async_client", lambda *_args: transport)

    async def scenario() -> None:
        llm_client.get_async_client_llm("gemini-2.5-flash")
        await llm_client.close_async_client_cache()

    asyncio.run(scenario())

    assert transport.close_calls == 1


@pytest.mark.parametrize("run_fails", [False, True])
def test_direct_async_runner_releases_cache_scope(
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

    monkeypatch.setattr(runner, "_run_async", run_body, raising=False)
    monkeypatch.setattr(llm_client, "close_async_client_cache", close_cache)

    if run_fails:
        with pytest.raises(RuntimeError, match="run failed"):
            asyncio.run(runner.run_async())
    else:
        asyncio.run(runner.run_async())

    assert events == ["run", "close"]


def test_overlapping_async_cache_scopes_close_after_last_user(
    monkeypatch: pytest.MonkeyPatch,
):
    close_calls = 0

    async def close_cache() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(llm_client, "close_async_client_cache", close_cache)

    async def scenario() -> tuple[int, int]:
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()

        async def user(entered: asyncio.Event, release: asyncio.Event) -> None:
            async with llm_client.async_client_cache_scope():
                entered.set()
                await release.wait()

        first = asyncio.create_task(user(first_entered, release_first))
        second = asyncio.create_task(user(second_entered, release_second))
        await first_entered.wait()
        await second_entered.wait()
        release_first.set()
        await first
        after_first = close_calls
        release_second.set()
        await second
        return after_first, close_calls

    after_first, after_second = asyncio.run(scenario())

    assert after_first == 0
    assert after_second == 1
