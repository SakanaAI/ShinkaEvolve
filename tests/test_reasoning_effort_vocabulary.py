"""One reasoning-effort vocabulary, validated up front, mapped by every provider."""

import ast
from pathlib import Path

import pytest

from shinka.llm.kwargs import (
    REASONING_EFFORTS,
    normalize_reasoning_effort,
    normalize_reasoning_efforts,
    sample_model_kwargs,
)
from shinka.llm.llm import AsyncLLMClient, LLMClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILIES = {
    "legacy_openai": "o4-mini",
    "legacy_openai_required": "gpt-5-mini",
    "current_openai": "gpt-6-luna",
    "gemini_budget": "gemini-2.5-flash",
    "gemini_level": "gemini-3.7-flash",
    "claude_budget": "claude-sonnet-4-6",
    "bedrock_budget": "us.anthropic.claude-sonnet-4-6-v1:0",
    "claude_adaptive": "claude-sonnet-5-5",
    "deepseek": "deepseek-v4-pro",
    "openrouter": "openrouter/qwen/qwen3-coder",
    "local": "local/qwen2.5-coder@http://localhost:11434/v1",
}
ACCEPTED_SPELLINGS = (*REASONING_EFFORTS, "", "auto")


def _sample(model: str, effort, max_tokens: int = 32768) -> dict:
    kwargs = sample_model_kwargs(
        model_names=[model],
        temperatures=[0.5],
        max_tokens=[max_tokens],
        reasoning_efforts=effort,
    )
    assert kwargs.pop("model_name") == model
    return kwargs


@pytest.mark.parametrize("effort", ACCEPTED_SPELLINGS)
@pytest.mark.parametrize("model", MODEL_FAMILIES.values(), ids=MODEL_FAMILIES)
def test_every_model_family_maps_every_effort(model, effort):
    kwargs = _sample(model, [effort])

    assert kwargs.get("reasoning", {}).get("effort", "unset") not in ("", "auto")


@pytest.mark.parametrize("effort", ["", "none", "auto", "disabled"])
@pytest.mark.parametrize(
    ("model", "thinking_off"),
    [
        ("gemini-2.5-flash", {"thinking_budget": 0}),
        ("claude-sonnet-4-6", {}),
        ("us.anthropic.claude-sonnet-4-6-v1:0", {}),
        ("deepseek-v4-pro", {"extra_body": {"thinking": {"type": "disabled"}}}),
    ],
)
def test_budget_and_deepseek_models_turn_thinking_off(model, thinking_off, effort):
    kwargs = _sample(model, [effort])
    kwargs.pop("temperature", None)

    assert kwargs == {"max_tokens": 32768, **thinking_off}


@pytest.mark.parametrize(
    ("effort", "budget"),
    [("min", 1024), ("minimal", 1024), ("high", 8192), ("xhigh", 12288)],
)
def test_budget_models_accept_openai_style_effort_names(effort, budget):
    assert _sample("gemini-2.5-flash", [effort])["thinking_budget"] == budget
    assert _sample("claude-sonnet-4-6", [effort])["thinking"] == {
        "type": "enabled",
        "budget_tokens": budget,
    }


@pytest.mark.parametrize("effort", ["", "auto", "disabled"])
def test_legacy_openai_sends_no_empty_or_auto_effort(effort):
    assert _sample("o4-mini", [effort])["reasoning"] == {"effort": None}
    assert _sample("gpt-5-mini", [effort])["reasoning"] == {
        "effort": "low",
        "summary": "auto",
    }


@pytest.mark.parametrize(
    ("effort", "sent"),
    [
        ("none", "none"),
        ("min", "low"),
        ("minimal", "minimal"),
        ("xhigh", "xhigh"),
        ("max", "high"),
    ],
)
def test_legacy_openai_effort_mapping_is_unchanged(effort, sent):
    assert _sample("gpt-5-mini", [effort])["reasoning"]["effort"] == sent


@pytest.mark.parametrize("model", MODEL_FAMILIES.values(), ids=MODEL_FAMILIES)
def test_unknown_effort_is_rejected_for_every_model_family(model):
    with pytest.raises(ValueError, match="Unsupported reasoning effort: 'turbo'"):
        _sample(model, ["turbo"])


def test_unknown_effort_is_rejected_even_when_not_sampled():
    for _ in range(20):
        with pytest.raises(ValueError, match="Unsupported reasoning effort"):
            _sample("gpt-5-mini", ["low", "turbo"])


@pytest.mark.parametrize("client_type", [LLMClient, AsyncLLMClient])
def test_clients_reject_unknown_effort_at_construction(client_type):
    with pytest.raises(ValueError, match="Unsupported reasoning effort: 'auot'"):
        client_type(model_names="gpt-5-mini", reasoning_efforts=["low", "auot"])

    client = client_type(model_names="gpt-5-mini", reasoning_efforts="medium")
    assert client.reasoning_efforts == "medium"


@pytest.mark.parametrize("client_type", [LLMClient, AsyncLLMClient])
def test_yaml_off_means_disabled(client_type):
    """`reasoning_efforts: off` reaches the client as the boolean False."""
    from omegaconf import OmegaConf

    configured = OmegaConf.create("reasoning_efforts: off").reasoning_efforts
    assert configured is False

    client_type(model_names="gpt-5-mini", reasoning_efforts=configured)
    assert normalize_reasoning_efforts(configured) == ["disabled"]
    assert normalize_reasoning_efforts([False, "low"]) == ["disabled", "low"]
    assert _sample("gemini-2.5-flash", configured)["thinking_budget"] == 0


@pytest.mark.parametrize("effort", [True, 3, 0.5])
@pytest.mark.parametrize("client_type", [LLMClient, AsyncLLMClient])
def test_other_non_string_efforts_get_the_vocabulary_error(client_type, effort):
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        client_type(model_names="gpt-5-mini", reasoning_efforts=effort)


def test_normalization_covers_aliases_case_and_single_values():
    assert normalize_reasoning_effort("") == "disabled"
    assert normalize_reasoning_effort(None) == "disabled"
    assert normalize_reasoning_effort("auto") == "disabled"
    assert normalize_reasoning_effort(" High ") == "high"
    assert normalize_reasoning_efforts("low") == ["low"]
    assert normalize_reasoning_efforts(["min", "minimal"]) == ["min", "minimal"]
    assert normalize_reasoning_efforts(REASONING_EFFORTS) == list(REASONING_EFFORTS)


@pytest.mark.parametrize("temperature", [0, 1, 0.5])
def test_scalar_temperature_may_be_an_integer(temperature):
    kwargs = sample_model_kwargs(
        model_names=["openrouter/qwen/qwen3-coder"],
        temperatures=temperature,
        max_tokens=1024,
        reasoning_efforts="disabled",
    )

    assert kwargs["temperature"] == temperature


def _literal_efforts(node: ast.expr) -> list[str]:
    value = ast.literal_eval(node)
    return [value] if isinstance(value, str) else list(value)


def _configured_example_efforts() -> list[tuple[str, str]]:
    """Collect `reasoning_efforts=...` keywords and dict entries from examples."""
    configured: list[tuple[str, str]] = []
    for script in sorted((ROOT / "examples").glob("*/run_evo.py")):
        value_nodes: list[ast.expr] = []
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "reasoning_efforts":
                value_nodes.append(node.value)
            elif isinstance(node, ast.Dict):
                value_nodes.extend(
                    value
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant)
                    and key.value == "reasoning_efforts"
                )
        configured.extend(
            (script.parent.name, effort)
            for value_node in value_nodes
            for effort in _literal_efforts(value_node)
        )
    return configured


def test_shipped_examples_only_name_supported_efforts():
    configured = _configured_example_efforts()

    assert ("game_2048", "disabled") in configured
    for example, effort in configured:
        assert effort in REASONING_EFFORTS, (example, effort)
