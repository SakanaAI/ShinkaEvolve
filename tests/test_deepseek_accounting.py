"""Regression tests for DeepSeek sync/async token and cost accounting."""

import asyncio
from types import SimpleNamespace

from shinka.llm.providers.deepseek import (
    get_deepseek_costs,
    query_deepseek,
    query_deepseek_async,
)


def _response(prompt_tokens=10, completion_tokens=30, reasoning_tokens=12):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    message = SimpleNamespace(content="ok", reasoning_content="thinking")
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _Client:
    def __init__(self, response, *, asynchronous=False):
        async def create_async(**kwargs):
            return response

        def create_sync(**kwargs):
            return response

        create = create_async if asynchronous else create_sync
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_sync_and_async_exclude_reasoning_from_output_tokens():
    response = _response()
    sync_result = query_deepseek(
        _Client(response), "deepseek/not-in-catalog", "msg", "sys", [], None
    )
    async_result = asyncio.run(
        query_deepseek_async(
            _Client(response, asynchronous=True),
            "deepseek/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert sync_result.output_tokens == async_result.output_tokens == 18
    assert sync_result.thinking_tokens == async_result.thinking_tokens == 12
    assert sync_result.input_tokens == async_result.input_tokens == 10


def test_unknown_model_preserves_tokens_and_defaults_cost_to_zero():
    costs = get_deepseek_costs(_response(5, 15, 5), "deepseek/not-in-catalog")

    assert costs == {
        "input_tokens": 5,
        "output_tokens": 10,
        "thinking_tokens": 5,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cost": 0.0,
    }


def test_sync_and_async_treat_missing_reasoning_count_as_zero():
    response = _response(reasoning_tokens=None)

    sync_result = query_deepseek(
        _Client(response), "deepseek/not-in-catalog", "msg", "sys", [], None
    )
    async_result = asyncio.run(
        query_deepseek_async(
            _Client(response, asynchronous=True),
            "deepseek/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert sync_result.output_tokens == async_result.output_tokens == 30
    assert sync_result.thinking_tokens == async_result.thinking_tokens == 0
