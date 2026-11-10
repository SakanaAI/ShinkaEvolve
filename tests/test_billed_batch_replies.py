"""Public batch APIs retain every billed empty reply without repeated exhaustion."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers import QueryResult
from shinka.llm.providers.errors import EmptyResponseError


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def apply_async(self, function, args):
        return SimpleNamespace(get=lambda: function(*args))


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "method", ["batch_query", "batch_kwargs_query", "batch_kwargs_query_indexed"]
)
@pytest.mark.parametrize("scenario", ["exhausted", "recovered", "all_empty"])
def test_batch_retains_billed_failures(monkeypatch, mode, method, scenario):
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        if scenario == "recovered" and len(calls) == 2:
            return QueryResult(
                "answer",
                "msg",
                "sys",
                [],
                "gpt-5-mini",
                {},
                10,
                10,
                cost=0.25,
            )
        raise EmptyResponseError(
            "no text",
            "max_output_tokens" if scenario == "exhausted" else None,
            {"cost": 0.5, "input_tokens": 1000, "thinking_tokens": 100},
        )

    async def query_async(**kwargs):
        return query(**kwargs)

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(llm_module, "query", query)
    monkeypatch.setattr(llm_module, "query_async", query_async)
    monkeypatch.setattr(llm_module.mp, "Pool", InlinePool)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)
    client_type = LLMClient if mode == "sync" else AsyncLLMClient
    client = client_type(model_names="gpt-5-mini", verbose=False)
    kwargs = {"num_samples": 1, "msg": "msg", "system_msg": "sys"}
    if method == "batch_query":
        kwargs["llm_kwargs"] = [{"model_name": "gpt-5-mini"}]
    responses = getattr(client, method)(**kwargs)
    if mode == "async":
        responses = asyncio.run(responses)
    assert len(responses) == 1
    result = responses[0]
    if method == "batch_kwargs_query_indexed":
        assert result[0] == 0
        result = result[1]
    expected_calls = {
        "exhausted": 1,
        "recovered": 2,
        "all_empty": llm_module.MAX_RETRIES,
    }[scenario]
    assert len(calls) == expected_calls
    assert result.num_total_queries == expected_calls
    assert result.cost == pytest.approx(
        0.75 if scenario == "recovered" else 0.5 * expected_calls
    )
    assert result.content == ("answer" if scenario == "recovered" else None)
    if scenario == "exhausted":
        assert result.incomplete_reason == "max_output_tokens"
