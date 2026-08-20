"""Regression tests for Anthropic pricing parity and response block parsing."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.providers.anthropic as anthropic_provider
from shinka.llm.providers.anthropic import (
    get_anthropic_costs,
    query_anthropic,
    query_anthropic_async,
)


def _response(
    input_tokens: int = 10,
    output_tokens: int = 20,
    thinking_tokens: int | None = None,
):
    details = (
        None
        if thinking_tokens is None
        else SimpleNamespace(thinking_tokens=thinking_tokens)
    )
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=details,
        ),
    )


class _Client:
    def __init__(self, response, *, asynchronous: bool = False):
        async def create_async(**_kwargs):
            return response

        def create_sync(**_kwargs):
            return response

        create = create_async if asynchronous else create_sync
        self.messages = SimpleNamespace(create=create)


def test_unknown_model_preserves_tokens_and_defaults_cost_to_zero():
    costs = get_anthropic_costs(_response(), "anthropic/not-in-catalog")

    assert costs == {
        "input_tokens": 10,
        "output_tokens": 20,
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cost": 0.0,
    }


def test_thinking_tokens_are_separated_but_billed_as_output(monkeypatch):
    billed_usage = []

    def calculate_cost(model, input_tokens, output_tokens):
        billed_usage.append((model, input_tokens, output_tokens))
        return 1.0, 2.0

    monkeypatch.setattr(anthropic_provider, "model_exists", lambda _model: True)
    monkeypatch.setattr(anthropic_provider, "calculate_cost", calculate_cost)

    costs = get_anthropic_costs(_response(thinking_tokens=7), "claude-test")

    assert costs == {
        "input_tokens": 10,
        "output_tokens": 13,
        "thinking_tokens": 7,
        "input_cost": 1.0,
        "output_cost": 2.0,
        "cost": 3.0,
    }
    assert billed_usage == [("claude-test", 10, 20)]


def test_thinking_tokens_are_bounded_by_billed_output():
    costs = get_anthropic_costs(
        _response(output_tokens=20, thinking_tokens=25),
        "anthropic/not-in-catalog",
    )

    assert costs["output_tokens"] == 0
    assert costs["thinking_tokens"] == 20


def test_sync_and_async_share_thinking_accounting():
    response = _response(thinking_tokens=7)
    sync_result = query_anthropic(
        _Client(response), "anthropic/not-in-catalog", "msg", "sys", [], None
    )
    async_result = asyncio.run(
        query_anthropic_async(
            _Client(response, asynchronous=True),
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert sync_result.output_tokens == async_result.output_tokens == 13
    assert sync_result.thinking_tokens == async_result.thinking_tokens == 7


def _blocks_response(blocks):
    return SimpleNamespace(
        content=[SimpleNamespace(type=name, **attrs) for name, attrs in blocks],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            output_tokens_details=None,
        ),
    )


CONTENT_BLOCK_CASES = [
    pytest.param([("text", {"text": "a"})], "a", "", id="text-only"),
    pytest.param(
        [("thinking", {"thinking": "t"}), ("text", {"text": "a"})],
        "a",
        "t",
        id="thinking-then-text",
    ),
    pytest.param(
        [
            ("thinking", {"thinking": "t"}),
            ("redacted_thinking", {"data": "e"}),
            ("text", {"text": "a"}),
        ],
        "a",
        "t\ne",
        id="redacted-thinking-in-the-middle",
    ),
    pytest.param(
        [("redacted_thinking", {"data": "e"}), ("text", {"text": "a"})],
        "a",
        "e",
        id="redacted-thinking-first",
    ),
    pytest.param([("thinking", {"thinking": "t"})], "", "t", id="thinking-only"),
    pytest.param(
        [
            ("thinking", {"thinking": "t"}),
            ("text", {"text": "p1"}),
            ("text", {"text": "p2"}),
        ],
        "p1\np2",
        "t",
        id="several-text-blocks",
    ),
    pytest.param(
        [("text", {"text": "a"}), ("tool_use", {"id": "u1", "name": "f", "input": {}})],
        "a",
        "",
        id="unhandled-block-type-is-skipped",
    ),
]


@pytest.mark.parametrize(
    "blocks, expected_content, expected_thought", CONTENT_BLOCK_CASES
)
def test_content_blocks_are_split_by_type(blocks, expected_content, expected_thought):
    result = query_anthropic(
        _Client(_blocks_response(blocks)),
        "anthropic/not-in-catalog",
        "msg",
        "sys",
        [],
        None,
    )

    assert result.content == expected_content
    assert result.thought == expected_thought


@pytest.mark.parametrize(
    "blocks, expected_content, expected_thought", CONTENT_BLOCK_CASES
)
def test_async_content_blocks_are_split_by_type(
    blocks, expected_content, expected_thought
):
    result = asyncio.run(
        query_anthropic_async(
            _Client(_blocks_response(blocks), asynchronous=True),
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert result.content == expected_content
    assert result.thought == expected_thought
