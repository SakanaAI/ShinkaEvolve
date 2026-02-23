from shinka.llm.kwargs import sample_model_kwargs


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
