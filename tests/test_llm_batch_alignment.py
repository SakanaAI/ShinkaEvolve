import asyncio
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import pytest

from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer
from shinka.database import Program
from shinka.llm import AsyncLLMClient, LLMClient
from shinka.llm import llm as llm_module
from shinka.llm.providers.errors import NonRetryableLLMError


@dataclass
class Response:
    content: str
    cost: float = 0.0


@pytest.fixture(params=[LLMClient, AsyncLLMClient], ids=["sync", "async"])
def batch_client(request, monkeypatch):
    def create(outcomes):
        client = request.param(verbose=True)

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


def query_batch(client, method_name):
    result = getattr(client, method_name)(
        num_samples=3, msg=["first", "second", "third"], system_msg="system"
    )
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("request failed"), Response("")]
)
def test_meta_summaries_keep_metadata_and_generation_order(batch_client, failure):
    client = batch_client(
        [failure, Response("Summary three", 0.3), Response("Summary two", 0.2)]
    )
    programs = [
        Program(id="first", code="pass", generation=1, metadata={"patch_name": "p1"}),
        Program(
            id="third",
            code="pass",
            generation=3,
            correct=False,
            metadata={"patch_name": "p3"},
        ),
        Program(
            id="second",
            code="pass",
            generation=2,
            correct=True,
            metadata={"patch_name": "p2"},
        ),
    ]
    if isinstance(client, AsyncLLMClient):
        summary, cost = asyncio.run(
            AsyncMetaSummarizer(
                MetaSummarizer(), client
            )._step1_individual_summaries_async(programs)
        )
    else:
        summary, cost = MetaSummarizer(client)._step1_individual_summaries(programs)

    assert summary == (
        "Summary two\n**Program Identifier:** "
        "Generation 2 - Patch Name p2 - Correct Program: True\n\n"
        "Summary three\n**Program Identifier:** "
        "Generation 3 - Patch Name p3 - Correct Program: False"
    )
    assert cost == pytest.approx(0.5)


@pytest.mark.parametrize("failure", [None, RuntimeError("request failed")])
def test_indexed_batch_keeps_request_indices_after_failure(batch_client, failure):
    second, third = Response("second", 0.2), Response("third", 0.3)
    client = batch_client([failure, second, third])

    assert query_batch(client, "batch_kwargs_query_indexed") == [
        (1, second),
        (2, third),
    ]


@pytest.mark.parametrize("failure", [None, RuntimeError("request failed")])
def test_compact_batch_returns_only_successful_responses(batch_client, failure):
    second, third = Response("second", 0.2), Response("third", 0.3)
    client = batch_client([failure, second, third])

    assert query_batch(client, "batch_kwargs_query") == [second, third]


@pytest.mark.parametrize(
    "method_name", ["batch_kwargs_query", "batch_kwargs_query_indexed"]
)
def test_batch_returns_no_results_when_all_requests_fail(batch_client, method_name):
    client = batch_client([None, RuntimeError("request failed"), None])

    assert query_batch(client, method_name) == []


@pytest.mark.parametrize(
    "method_name", ["batch_kwargs_query", "batch_kwargs_query_indexed"]
)
def test_batch_propagates_non_retryable_errors(batch_client, method_name):
    client = batch_client([NonRetryableLLMError("unsupported"), None, None])

    with pytest.raises(NonRetryableLLMError, match="unsupported"):
        query_batch(client, method_name)
