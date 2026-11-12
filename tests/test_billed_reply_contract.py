"""Billed empty replies retain failure context and complete usage totals."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers import QueryResult
from shinka.llm.providers.errors import EmptyResponseError
from shinka.llm.providers.openai import query_openai, query_openai_async
from pydantic import BaseModel


def empty_reply(reason="max_output_tokens"):
    return EmptyResponseError(
        "no text",
        reason,
        dict(
            cost=0.5,
            input_cost=0.1,
            output_cost=0.4,
            input_tokens=1000,
            output_tokens=0,
            thinking_tokens=65536,
        ),
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_output_limit_reason_survives_client_and_serialization(monkeypatch, mode):
    def fail(**kwargs):
        raise empty_reply()

    async def fail_async(**kwargs):
        fail(**kwargs)

    monkeypatch.setattr(llm_module, "query", fail)
    monkeypatch.setattr(llm_module, "query_async", fail_async)
    client_type = LLMClient if mode == "sync" else AsyncLLMClient
    client = client_type(model_names="gpt-5-mini", verbose=False)
    result = client.query("msg", "sys", llm_kwargs={"model_name": "gpt-5-mini"})
    if mode == "async":
        result = asyncio.run(result)
    assert result.incomplete_reason == "max_output_tokens"
    assert result.to_dict()["incomplete_reason"] == "max_output_tokens"
    assert result.cost == 0.5
    assert "Content:" in str(result)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_recovered_response_aggregates_usage_and_query_counts(monkeypatch, mode):
    replies = [
        empty_reply(None),
        QueryResult(
            "answer",
            "msg",
            "sys",
            [],
            "gpt-5-mini",
            {},
            10,
            10,
            cost=0.25,
            input_cost=0.1,
            output_cost=0.15,
        ),
    ]

    def query(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def query_async(**kwargs):
        return query(**kwargs)

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(llm_module, "query", query)
    monkeypatch.setattr(llm_module, "query_async", query_async)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)
    client_type = LLMClient if mode == "sync" else AsyncLLMClient
    result = client_type(model_names="gpt-5-mini", verbose=False).query(
        "msg", "sys", llm_kwargs={"model_name": "gpt-5-mini"}
    )
    if mode == "async":
        result = asyncio.run(result)
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == (
        1010,
        10,
        65536,
    )
    assert result.num_total_queries == 2
    assert result.cost == 0.75


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_structured_empty_reply_preserves_billing_and_termination(mode):
    response = SimpleNamespace(
        output_parsed=None,
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=65536,
            output_tokens_details=SimpleNamespace(reasoning_tokens=65536),
            cost_details={
                "upstream_inference_input_cost": 0.1,
                "upstream_inference_output_cost": 0.4,
            },
        ),
    )

    async def parse_async(**kwargs):
        return response

    parse = (lambda **kwargs: response) if mode == "sync" else parse_async
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    with pytest.raises(EmptyResponseError) as caught:
        if mode == "sync":
            query_openai(client, "gpt-5-mini", "msg", "sys", [], object())
        else:
            asyncio.run(
                query_openai_async(client, "gpt-5-mini", "msg", "sys", [], object())
            )
    assert caught.value.cost == 0.5
    assert caught.value.hit_output_limit


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_structured_success_preserves_parsed_model_and_cost(mode):
    class Answer(BaseModel):
        answer: str

    parsed = Answer(answer="OK")
    response = SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            cost_details={
                "upstream_inference_input_cost": 0.1,
                "upstream_inference_output_cost": 0.4,
            },
        ),
    )

    async def parse_async(**kwargs):
        return response

    parse = (lambda **kwargs: response) if mode == "sync" else parse_async
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    if mode == "sync":
        result = query_openai(client, "gpt-5-mini", "msg", "sys", [], Answer)
    else:
        result = asyncio.run(
            query_openai_async(client, "gpt-5-mini", "msg", "sys", [], Answer)
        )
    assert result.content is parsed
    assert result.cost == 0.5
    assert result.new_msg_history[-1] == {"role": "assistant", "content": "answer:OK\n"}
