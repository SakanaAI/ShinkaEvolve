import pytest

from shinka.llm.providers.model_resolver import resolve_model_backend


def test_resolve_known_pricing_model():
    resolved = resolve_model_backend("gpt-5-mini")
    assert resolved.provider == "openai"
    assert resolved.api_model_name == "gpt-5-mini"
    assert resolved.base_url is None


def test_resolve_openrouter_dynamic_model():
    resolved = resolve_model_backend("openrouter/qwen/qwen3-coder")
    assert resolved.provider == "openrouter"
    assert resolved.api_model_name == "qwen/qwen3-coder"
    assert resolved.base_url is None


def test_resolve_local_model_with_inline_url():
    resolved = resolve_model_backend("local/qwen2.5-coder@http://localhost:11434/v1")
    assert resolved.provider == "local_openai"
    assert resolved.api_model_name == "qwen2.5-coder"
    assert resolved.base_url == "http://localhost:11434/v1"


def test_resolve_azure_prefixed_model():
    resolved = resolve_model_backend("azure-gpt-4.1")
    assert resolved.provider == "azure_openai"
    assert resolved.api_model_name == "gpt-4.1"


def test_invalid_local_model_format_raises():
    with pytest.raises(ValueError, match="not supported|Invalid local model URL"):
        resolve_model_backend("local/bad-format")
