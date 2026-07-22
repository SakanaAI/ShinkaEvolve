"""Regression tests for Anthropic pricing fallback parity."""

from types import SimpleNamespace

from shinka.llm.providers.anthropic import get_anthropic_costs


def test_unknown_model_preserves_tokens_and_defaults_cost_to_zero():
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=20)
    )

    costs = get_anthropic_costs(response, "anthropic/not-in-catalog")

    assert costs == {
        "input_tokens": 10,
        "output_tokens": 20,
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cost": 0.0,
    }
