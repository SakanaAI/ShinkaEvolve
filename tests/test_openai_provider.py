from types import SimpleNamespace

from shinka.llm.providers.openai import get_openai_costs


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
