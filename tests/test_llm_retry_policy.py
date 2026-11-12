"""Regression tests for top-level LLM retry pacing."""

import asyncio

import pytest

import shinka.llm.llm as llm_module
from shinka.llm.constants import MAX_RETRIES
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers import QueryResult
from shinka.llm.providers.errors import EmptyResponseError


def _empty_reply(reason, cost):
    return EmptyResponseError(
        "OpenAI response contained no text output",
        incomplete_reason=reason,
        cost_results={
            "input_tokens": 1000,
            "output_tokens": 0,
            "thinking_tokens": 65536,
            "input_cost": 0.25,
            "output_cost": cost - 0.25,
            "cost": cost,
        },
    )


def _answer(cost):
    return QueryResult(
        content="answer",
        msg="msg",
        system_msg="sys",
        new_msg_history=[],
        model_name="test-model",
        kwargs={},
        input_tokens=10,
        output_tokens=10,
        cost=cost,
        input_cost=cost / 2,
        output_cost=cost / 2,
    )


async def _no_sleep(seconds):
    return None


def test_sync_query_sleeps_between_transient_failures(monkeypatch):
    calls = 0
    sleeps = []

    def always_fails(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("transient failure")

    monkeypatch.setattr(llm_module, "query", always_fails)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = LLMClient(model_names="test-model", verbose=False)

    result = client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})

    assert result is None
    assert calls == MAX_RETRIES
    assert sleeps == [1] * (MAX_RETRIES - 1)


def test_async_query_sleeps_between_transient_failures(monkeypatch):
    calls = 0
    sleeps = []

    async def always_fails(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("transient failure")

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_module, "query_async", always_fails)
    monkeypatch.setattr(llm_module.asyncio, "sleep", record_sleep)
    client = AsyncLLMClient(model_names="test-model", verbose=False)

    result = asyncio.run(
        client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})
    )

    assert result is None
    assert calls == MAX_RETRIES
    assert sleeps == [1] * (MAX_RETRIES - 1)


def test_async_query_does_not_resend_after_output_token_limit(monkeypatch):
    calls = 0

    async def out_of_tokens(**kwargs):
        nonlocal calls
        calls += 1
        raise _empty_reply("max_output_tokens", 3.55)

    monkeypatch.setattr(llm_module, "query_async", out_of_tokens)
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_sleep)
    client = AsyncLLMClient(model_names="test-model", verbose=False)

    result = asyncio.run(
        client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})
    )

    assert calls == 1
    assert result is not None
    assert result.content is None
    assert result.cost == pytest.approx(3.55)
    assert result.thinking_tokens == 65536
    assert result.model_name == "test-model"


def test_async_query_charges_empty_replies_to_later_answer(monkeypatch):
    replies = [_empty_reply(None, 0.5), _answer(0.25)]

    async def empty_then_answer(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm_module, "query_async", empty_then_answer)
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_sleep)
    client = AsyncLLMClient(model_names="test-model", verbose=False)

    result = asyncio.run(
        client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})
    )

    assert result.content == "answer"
    assert result.cost == pytest.approx(0.75)
    assert result.input_cost + result.output_cost == pytest.approx(result.cost)


def test_async_query_returns_billed_cost_when_every_try_is_empty(monkeypatch):
    calls = 0

    async def always_empty(**kwargs):
        nonlocal calls
        calls += 1
        raise _empty_reply("content_filter", 0.5)

    monkeypatch.setattr(llm_module, "query_async", always_empty)
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_sleep)
    client = AsyncLLMClient(model_names="test-model", verbose=False)

    result = asyncio.run(
        client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})
    )

    assert calls == MAX_RETRIES
    assert result.content is None
    assert result.cost == pytest.approx(0.5 * MAX_RETRIES)
    assert result.num_total_queries == MAX_RETRIES


def test_sync_query_does_not_resend_after_output_token_limit(monkeypatch):
    calls = 0

    def out_of_tokens(**kwargs):
        nonlocal calls
        calls += 1
        raise _empty_reply("max_output_tokens", 3.55)

    monkeypatch.setattr(llm_module, "query", out_of_tokens)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    client = LLMClient(model_names="test-model", verbose=False)

    result = client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})

    assert calls == 1
    assert result.content is None
    assert result.cost == pytest.approx(3.55)
