"""One refused request must not abort the other samples of a batch."""

import asyncio
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import pytest

from shinka.llm import AsyncLLMClient, LLMClient
from shinka.llm import llm as llm_module
from shinka.llm.providers.errors import (
    LLMEndpointError,
    NonRetryableLLMError,
    RequestRejectedError,
)

METHODS = ["batch_kwargs_query", "batch_kwargs_query_indexed"]


@dataclass
class Response:
    content: str
    cost: float = 0.0


@pytest.fixture(params=[LLMClient, AsyncLLMClient], ids=["sync", "async"])
def batch_client(request, monkeypatch):
    def create(outcomes):
        client = request.param(verbose=False)

        def sample(index, *args):
            outcome = outcomes[index]
            if isinstance(outcome, Exception):
                raise outcome
            return index, outcome

        async def sample_async(index, *args):
            return sample(index, *args)

        monkeypatch.setattr(llm_module.mp, "Pool", ThreadPool)
        monkeypatch.setattr(llm_module, "sample_kwargs_query_fn", sample)
        if isinstance(client, AsyncLLMClient):
            monkeypatch.setattr(
                client, "_sample_kwargs_query_async_with_retry", sample_async
            )
        return client

    return create


def _query_batch(client, method_name):
    result = getattr(client, method_name)(
        num_samples=3, msg=["first", "second", "third"], system_msg="system"
    )
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


def _contents(results) -> list[str]:
    replies = [item[1] if isinstance(item, tuple) else item for item in results]
    return sorted(reply.content for reply in replies)


@pytest.mark.parametrize("method_name", METHODS)
def test_rejected_request_drops_only_its_own_sample(batch_client, method_name):
    client = batch_client(
        [
            Response("first"),
            RequestRejectedError("OpenAI rejected the request with HTTP 400"),
            Response("third"),
        ]
    )

    assert _contents(_query_batch(client, method_name)) == ["first", "third"]


@pytest.mark.parametrize("method_name", METHODS)
def test_endpoint_failure_drops_only_its_own_sample(batch_client, method_name):
    """A pool with one unreachable model still returns the other models' replies."""
    client = batch_client(
        [Response("first"), LLMEndpointError(404, "model_not_found"), Response("third")]
    )

    assert _contents(_query_batch(client, method_name)) == ["first", "third"]


@pytest.mark.parametrize("method_name", METHODS)
def test_batch_without_any_reply_raises_the_endpoint_failure(batch_client, method_name):
    client = batch_client(
        [
            RequestRejectedError("OpenAI rejected the request with HTTP 400"),
            LLMEndpointError(401, "invalid_api_key"),
            None,
        ]
    )

    with pytest.raises(LLMEndpointError, match="401"):
        _query_batch(client, method_name)


@pytest.mark.parametrize("method_name", METHODS)
def test_fully_rejected_batch_raises_the_rejection(batch_client, method_name):
    """A batch in which nothing was answered must not look like an empty success."""
    client = batch_client(
        [RequestRejectedError("HTTP 400: first"), RequestRejectedError("HTTP 400"), None]
    )

    with pytest.raises(RequestRejectedError, match="first"):
        _query_batch(client, method_name)


@pytest.mark.parametrize("method_name", METHODS)
def test_other_permanent_error_still_aborts_the_batch(batch_client, method_name):
    client = batch_client(
        [Response("first"), NonRetryableLLMError("unsupported option"), Response("third")]
    )

    with pytest.raises(NonRetryableLLMError, match="unsupported option"):
        _query_batch(client, method_name)


def test_aborted_async_batch_bills_what_a_dropped_sample_had_spent():
    rejected = RequestRejectedError("HTTP 400")
    rejected.cost = 0.5
    aborting = NonRetryableLLMError("unsupported option")
    aborting.cost = 0.25

    async def outcome(index, value):
        if isinstance(value, Exception):
            raise value
        return index, value

    async def collect():
        outcomes = [rejected, aborting, Response("third", cost=1.0)]
        return await llm_module._collect_async_batch(
            [outcome(index, value) for index, value in enumerate(outcomes)]
        )

    with pytest.raises(NonRetryableLLMError, match="unsupported option") as raised:
        asyncio.run(collect())

    assert raised.value.cost == pytest.approx(1.75)


def test_rejected_request_is_still_non_retryable_for_single_queries():
    assert issubclass(RequestRejectedError, NonRetryableLLMError)
    assert not issubclass(RequestRejectedError, LLMEndpointError)
