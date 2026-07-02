from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.pricing import is_reasoning_model, requires_reasoning


def test_sample_model_kwargs_uses_max_tokens_for_local_openai():
    kwargs = sample_model_kwargs(
        model_names=["local/qwen2.5-coder@http://localhost:11434/v1"],
        temperatures=[0.25],
        max_tokens=[321],
        reasoning_efforts=["disabled"],
    )

    assert kwargs["model_name"] == "local/qwen2.5-coder@http://localhost:11434/v1"
    assert kwargs["temperature"] == 0.25
    assert kwargs["max_tokens"] == 321
    assert "max_output_tokens" not in kwargs


def test_sample_model_kwargs_uses_max_output_tokens_for_dynamic_openrouter():
    kwargs = sample_model_kwargs(
        model_names=["openrouter/qwen/qwen3-coder"],
        temperatures=[0.15],
        max_tokens=[222],
        reasoning_efforts=["disabled"],
    )

    assert kwargs["model_name"] == "openrouter/qwen/qwen3-coder"
    assert kwargs["temperature"] == 0.15
    assert kwargs["max_output_tokens"] == 222
    assert "max_tokens" not in kwargs


def test_gpt5_mini_pricing_metadata_enables_reasoning_kwargs():
    assert is_reasoning_model("gpt-5-mini")
    assert requires_reasoning("gpt-5-mini")

    kwargs = sample_model_kwargs(
        model_names=["gpt-5-mini"],
        temperatures=[0.0],
        max_tokens=[8192],
        reasoning_efforts=["minimal"],
    )

    assert kwargs["temperature"] == 1.0
    assert kwargs["max_output_tokens"] == 8192
    assert kwargs["reasoning"] == {"effort": "minimal", "summary": "auto"}
