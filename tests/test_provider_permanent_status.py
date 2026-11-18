"""Provider calls must fail fast on HTTP statuses that cannot succeed."""

import asyncio
from typing import Any, Callable
from unittest.mock import patch

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.anthropic import query_anthropic, query_anthropic_async
from shinka.llm.providers.deepseek import query_deepseek, query_deepseek_async
from shinka.llm.providers.errors import (
    LLMEndpointError,
    NonRetryableLLMError,
    RequestRejectedError,
)
from shinka.llm.providers.gemini import query_gemini, query_gemini_async
from shinka.llm.providers.local_openai import (
    query_local_openai,
    query_local_openai_async,
)
from shinka.llm.providers.openai import query_openai, query_openai_async
from shinka.llm.providers.retry import is_permanent_status_error

PERMANENT_STATUSES = [400, 401, 403, 404, 413, 422]
ENDPOINT_STATUSES = {401, 403, 404}
TRANSIENT_STATUSES = [408, 409, 429, 500, 503]


def _http_response(status: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://provider.invalid/v1")
    return httpx.Response(status, request=request, headers=headers)


def _openai_error(status: int, headers: dict[str, str] | None = None):
    return openai.APIStatusError(
        "rejected", response=_http_response(status, headers), body=None
    )


def _anthropic_error(status: int, headers: dict[str, str] | None = None):
    return anthropic.APIStatusError(
        "rejected",
        response=_http_response(status, headers),
        body={"type": "error", "error": {"type": "provider_error_type"}},
    )


def _gemini_error(status: int, headers: dict[str, str] | None = None):
    payload = {"error": {"code": status, "message": "rejected", "status": "REJECTED"}}
    error_type = genai_errors.ClientError if status < 500 else genai_errors.ServerError
    return error_type(status, payload, _http_response(status, headers))


PROVIDERS: dict[str, tuple[Callable, Callable, Callable]] = {
    "openai": (query_openai, query_openai_async, _openai_error),
    "anthropic": (query_anthropic, query_anthropic_async, _anthropic_error),
    "deepseek": (query_deepseek, query_deepseek_async, _openai_error),
    "local_openai": (query_local_openai, query_local_openai_async, _openai_error),
    "gemini": (query_gemini, query_gemini_async, _gemini_error),
}


class _FailingClient:
    """Stands in for every SDK client: each call raises the next queued error."""

    def __init__(self, errors: list[Exception], asynchronous: bool):
        self.errors = list(errors)
        self.asynchronous = asynchronous
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        # responses / messages / chat.completions / models / aio.models
        return self

    def with_options(self, **kwargs: Any) -> "_FailingClient":
        return self

    def _fail(self, **kwargs: Any) -> Any:
        self.calls += 1
        error = self.errors[min(self.calls, len(self.errors)) - 1]
        if not self.asynchronous:
            raise error

        async def fail() -> None:
            raise error

        return fail()

    create = _fail
    generate_content = _fail


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """A regression would otherwise sleep through 20 backoff rounds."""

    async def no_async_sleep(seconds):
        return None

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(asyncio, "sleep", no_async_sleep)


def _run_query(provider: str, client: _FailingClient) -> None:
    query, query_async, _ = PROVIDERS[provider]
    history: list[dict] = []
    arguments = (client, "test-model", "msg", "sys", history, None)
    if client.asynchronous:
        asyncio.run(query_async(*arguments))
    else:
        query(*arguments)


@pytest.mark.parametrize("status", PERMANENT_STATUSES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_permanent_status_is_raised_after_one_attempt(provider, asynchronous, status):
    make_error = PROVIDERS[provider][2]
    client = _FailingClient([make_error(status)], asynchronous)

    with pytest.raises(NonRetryableLLMError, match=str(status)) as raised:
        _run_query(provider, client)

    assert client.calls == 1
    error = raised.value
    if status in ENDPOINT_STATUSES:
        assert isinstance(error, LLMEndpointError)
        assert error.status_code == status
    else:
        assert isinstance(error, RequestRejectedError)


@pytest.mark.parametrize("status", TRANSIENT_STATUSES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_transient_status_is_still_retried(provider, asynchronous, status):
    make_error = PROVIDERS[provider][2]
    client = _FailingClient([make_error(status), make_error(400)], asynchronous)

    with pytest.raises(NonRetryableLLMError, match="400"):
        _run_query(provider, client)

    assert client.calls == 2


@pytest.mark.parametrize(
    "provider,expected_code",
    [
        ("openai", "APIStatusError"),
        ("anthropic", "provider_error_type"),
        ("gemini", "REJECTED"),
    ],
)
def test_endpoint_error_carries_the_provider_error_code(provider, expected_code):
    client = _FailingClient([PROVIDERS[provider][2](401)], asynchronous=False)

    with pytest.raises(LLMEndpointError) as raised:
        _run_query(provider, client)

    assert raised.value.code == expected_code


def test_server_retry_directive_overrides_the_status_code():
    assert is_permanent_status_error(_openai_error(400)) is True
    assert is_permanent_status_error(_openai_error(429)) is False
    assert is_permanent_status_error(_openai_error(500)) is False
    assert (
        is_permanent_status_error(_openai_error(400, {"x-should-retry": "true"}))
        is False
    )
    assert (
        is_permanent_status_error(_openai_error(429, {"x-should-retry": "false"}))
        is True
    )
    assert is_permanent_status_error(RuntimeError("no status")) is False


@pytest.mark.parametrize("status", sorted(ENDPOINT_STATUSES))
def test_endpoint_failures_are_permanent_whatever_the_server_directive(status):
    error = _openai_error(status, {"x-should-retry": "true"})
    client = _FailingClient([error], asynchronous=False)

    with pytest.raises(LLMEndpointError, match=str(status)):
        _run_query("local_openai", client)

    assert client.calls == 1


@pytest.mark.parametrize(
    "provider,status,error_type",
    [
        ("openai", 400, NonRetryableLLMError),
        ("anthropic", 401, LLMEndpointError),
    ],
)
def test_permanent_status_crosses_the_client_retry_loop_once(
    provider, status, error_type
):
    """A revoked key used to cost 20 backoff rounds times 3 client retries."""
    client = _FailingClient([PROVIDERS[provider][2](status)], asynchronous=True)

    async def run():
        with patch(
            "shinka.llm.query.get_async_client_llm",
            return_value=(client, "test-model", provider),
        ):
            await AsyncLLMClient(model_names="test-model", verbose=False).query(
                "msg", "sys", llm_kwargs={"model_name": "test-model"}
            )

    with pytest.raises(error_type, match=str(status)):
        asyncio.run(run())

    assert client.calls == 1
