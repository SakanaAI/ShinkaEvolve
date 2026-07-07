from types import SimpleNamespace

import pytest

from shinka.llm.providers.openai import (
    _retry_after_seconds,
    get_openai_costs,
    query_openai,
)


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cost_details=None,
):
    output_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_tokens_details=output_details,
        cost_details=cost_details,
    )


def test_get_openai_costs_defaults_to_zero_for_unknown_model_without_cost_details():
    response = SimpleNamespace(
        usage=_usage(
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=5,
            cost_details=None,
        )
    )

    costs = get_openai_costs(response, "openrouter/not-in-pricing")
    assert costs["input_tokens"] == 10
    assert costs["output_tokens"] == 15
    assert costs["thinking_tokens"] == 5
    assert costs["input_cost"] == 0.0
    assert costs["output_cost"] == 0.0
    assert costs["cost"] == 0.0


def test_get_openai_costs_uses_openrouter_cost_details_when_available():
    response = SimpleNamespace(
        usage=_usage(
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=0,
            cost_details={
                "upstream_inference_input_cost": 0.12,
                "upstream_inference_output_cost": 0.34,
            },
        )
    )

    costs = get_openai_costs(response, "openrouter/qwen/qwen3-coder")
    assert costs["input_cost"] == 0.12
    assert costs["output_cost"] == 0.34
    assert costs["cost"] == 0.46


def test_retry_after_seconds_reads_cloudflare_error_body():
    exc = SimpleNamespace(
        response=SimpleNamespace(headers={}),
        body={"retry_after": 60},
    )

    assert _retry_after_seconds(exc) == 60


def test_retry_after_seconds_prefers_response_header():
    exc = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after": "30"}),
        body={"retry_after": 60},
    )

    assert _retry_after_seconds(exc) == 30


class _FakeOpenAIClient:
    def __init__(self, response):
        self.responses = SimpleNamespace(create=lambda **kwargs: response)


def _content(text: str):
    return SimpleNamespace(type="output_text", text=text)


def _response(*, output=None, output_text=None, status="completed"):
    return SimpleNamespace(
        output=[] if output is None else output,
        output_text=output_text,
        status=status,
        usage=_usage(input_tokens=10, output_tokens=20),
    )


def test_query_openai_reads_text_from_first_output_item():
    response = _response(
        output=[SimpleNamespace(type="message", content=[_content("123")])]
    )

    result = query_openai(
        _FakeOpenAIClient(response),
        "gpt-5-mini",
        "problem",
        "system",
        [],
        None,
        max_output_tokens=8192,
    )

    assert result.content == "123"
    assert result.new_msg_history[-1] == {"role": "assistant", "content": "123"}


def test_query_openai_reads_text_after_reasoning_output_item():
    response = _response(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="reasoning summary")],
            ),
            SimpleNamespace(type="message", content=[_content("456")]),
        ]
    )

    result = query_openai(
        _FakeOpenAIClient(response),
        "gpt-5-mini",
        "problem",
        "system",
        [],
        None,
        max_output_tokens=8192,
    )

    assert result.content == "456"
    assert result.thought == "reasoning summary"


def test_query_openai_scans_later_output_items_for_text():
    response = _response(
        output=[
            {"type": "reasoning", "summary": [{"text": "first"}]},
            {"type": "tool_call", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": "789"}]},
        ]
    )

    result = query_openai(
        _FakeOpenAIClient(response),
        "gpt-5-mini",
        "problem",
        "system",
        [],
        None,
        max_output_tokens=8192,
    )

    assert result.content == "789"
    assert result.thought == "first"


def test_query_openai_uses_output_text_fallback():
    response = _response(output_text="321")

    result = query_openai(
        _FakeOpenAIClient(response),
        "gpt-5-mini",
        "problem",
        "system",
        [],
        None,
        max_output_tokens=8192,
    )

    assert result.content == "321"


def test_query_openai_raises_clear_error_when_response_has_no_text():
    response = _response(
        output=[SimpleNamespace(type="reasoning", summary=[])],
        status="incomplete",
    )
    response.incomplete_details = SimpleNamespace(reason="max_output_tokens")

    with pytest.raises(ValueError, match="OpenAI response contained no text output"):
        query_openai(
            _FakeOpenAIClient(response),
            "gpt-5-mini",
            "problem",
            "system",
            [],
            None,
            max_output_tokens=8192,
        )


def test_query_openai_does_not_treat_reasoning_text_as_output():
    response = _response(
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[
                    SimpleNamespace(type="reasoning_text", text="internal reasoning")
                ],
            )
        ],
        status="incomplete",
    )

    with pytest.raises(ValueError, match="OpenAI response contained no text output"):
        query_openai(
            _FakeOpenAIClient(response),
            "gpt-5-mini",
            "problem",
            "system",
            [],
            None,
            max_output_tokens=8192,
        )
