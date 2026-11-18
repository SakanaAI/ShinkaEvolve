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
def test_endpoint_failure_still_aborts_the_batch(batch_client, method_name):
    client = batch_client(
        [Response("first"), LLMEndpointError(401, "invalid_api_key"), None]
    )

    with pytest.raises(LLMEndpointError, match="401"):
        _query_batch(client, method_name)


def test_rejected_request_is_still_non_retryable_for_single_queries():
    assert issubclass(RequestRejectedError, NonRetryableLLMError)
    assert not issubclass(RequestRejectedError, LLMEndpointError)
