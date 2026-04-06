import pytest

from shinka.embed.client import resolve_embedding_backend


def test_resolve_known_embedding_model():
    resolved = resolve_embedding_backend("text-embedding-3-small")

    assert resolved.provider == "openai"
    assert resolved.api_model_name == "text-embedding-3-small"
    assert resolved.base_url is None


def test_resolve_openrouter_embedding_model():
    resolved = resolve_embedding_backend("openrouter/qwen/qwen3-coder")

    assert resolved.provider == "openrouter"
    assert resolved.api_model_name == "qwen/qwen3-coder"
    assert resolved.base_url is None


def test_resolve_local_embedding_model_with_inline_url():
    resolved = resolve_embedding_backend(
        "local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"
    )

    assert resolved.provider == "local_openai"
    assert resolved.api_model_name == "BAAI/bge-small-en-v1.5"
    assert resolved.base_url == "http://localhost:8080/v1"
    assert resolved.api_key_env_name is None


def test_resolve_local_embedding_model_with_api_key_env_query_param():
    resolved = resolve_embedding_backend(
        "local/dummy-embed@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
    )

    assert resolved.provider == "local_openai"
    assert resolved.api_model_name == "dummy-embed"
    assert resolved.base_url == "https://api.example.test/v1"
    assert resolved.api_key_env_name == "CUSTOM_API_KEY"


def test_invalid_local_embedding_model_format_raises():
    with pytest.raises(ValueError, match="not supported|Invalid local model URL"):
        resolve_embedding_backend("local/bad-format")
