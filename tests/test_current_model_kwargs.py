"""Current provider contracts, including disabled and maximum reasoning."""

import pytest

from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.model_resolver import ResolvedModel

OPENAI_MODELS = (
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-6-luna",
)
CLAUDE_MODELS = (
    "claude-fable-5-1",
    "claude-fable-5-2",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)


def _sample(model: str, effort: str, max_tokens: int = 4096):
    return sample_model_kwargs(
        model_names=[model],
        temperatures=[0.25],
        max_tokens=[max_tokens],
        reasoning_efforts=[effort],
    )


@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize("effort", ["high", "xhigh", "max"])
def test_current_openai_preserves_supported_reasoning_levels(model, effort):
    assert _sample(model, effort) == {
        "model_name": model,
        "max_output_tokens": 4096,
        "reasoning": {"effort": effort, "summary": "auto"},
    }


@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize("effort", ["disabled", "none", ""])
def test_current_openai_disabled_reasoning_respects_model_capabilities(model, effort):
    required = model in ("gpt-6-astra", "gpt-6.1-sol")
    reasoning = {"effort": "low", "summary": "auto"} if required else {"effort": "none"}
    assert _sample(model, effort) == {
        "model_name": model,
        "max_output_tokens": 4096,
        "reasoning": reasoning,
    }


@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize("effort", ["min", "minimal"])
def test_current_openai_maps_minimal_effort_to_low(model, effort):
    assert _sample(model, effort)["reasoning"]["effort"] == "low"


def test_azure_current_openai_uses_underlying_model_capabilities():
    kwargs = _sample("azure-gpt-6.1-sol", "disabled")
    assert kwargs["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "temperature" not in kwargs


@pytest.mark.parametrize("prefix", ["anthropic.", "us.anthropic.", "global.anthropic."])
@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_runtime_bedrock_claude_uses_same_thinking_contract(monkeypatch, prefix, model):
    qualified_model = prefix + model
    monkeypatch.setattr(
        "shinka.llm.kwargs.resolve_model_backend",
        lambda _model: ResolvedModel(qualified_model, qualified_model, "bedrock"),
    )
    kwargs = _sample(qualified_model, "max")
    assert kwargs == {
        "model_name": qualified_model,
        "max_tokens": 4096,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "max"},
    }


@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_current_claude_uses_adaptive_thinking_and_output_effort(model, effort):
    assert _sample(model, effort) == {
        "model_name": model,
        "max_tokens": 4096,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("claude-fable-5-1", {"type": "adaptive", "display": "summarized"}),
        ("claude-fable-5-2", {"type": "adaptive", "display": "summarized"}),
        ("claude-opus-5-5", {"type": "adaptive", "display": "summarized"}),
        ("claude-sonnet-5-5", {"type": "between_tools"}),
        ("claude-haiku-5-5", {"type": "disabled"}),
    ],
)
@pytest.mark.parametrize("effort", ["disabled", "none", ""])
def test_current_claude_disabled_reasoning_uses_supported_mode(model, thinking, effort):
    assert _sample(model, effort) == {
        "model_name": model,
        "max_tokens": 4096,
        "thinking": thinking,
        "output_config": {"effort": "low"},
    }


@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize("effort", ["min", "minimal"])
def test_current_claude_maps_minimal_effort_to_low(model, effort):
    assert _sample(model, effort)["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_current_claude_caps_output_at_documented_limit(model):
    assert _sample(model, "high", max_tokens=200_000)["max_tokens"] == 128_000


@pytest.mark.parametrize("model", [*OPENAI_MODELS, *CLAUDE_MODELS])
def test_current_models_reject_unknown_effort_before_request(model):
    with pytest.raises(ValueError, match="reasoning effort"):
        _sample(model, "turbo")


@pytest.mark.parametrize(
    "model",
    [
        "gemini-4-argon",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ],
)
@pytest.mark.parametrize("effort", ["min", "minimal", "medium", "max", "disabled"])
def test_current_gemini_uses_levels_without_sampling_or_token_budgets(model, effort):
    expected = {"model_name": model, "max_tokens": 4096}
    if effort != "disabled":
        expected["thinking_level"] = (
            "MINIMAL"
            if model == "gemini-3.5-flash-lite" and effort in ("min", "minimal")
            else {"min": "LOW", "minimal": "LOW", "medium": "MEDIUM", "max": "HIGH"}[
                effort
            ]
        )
    assert _sample(model, effort) == expected


def test_legacy_openai_max_alias_is_preserved():
    assert _sample("gpt-5-mini", "max")["reasoning"]["effort"] == "high"


def test_legacy_claude_keeps_manual_thinking_budget():
    kwargs = _sample("claude-sonnet-4-6", "low")
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in kwargs
