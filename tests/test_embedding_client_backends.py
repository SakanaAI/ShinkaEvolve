import asyncio
from types import SimpleNamespace

import openai
import pytest

import shinka.embed.client as embed_client_module
from shinka.embed.client import get_async_client_embed, get_client_embed
from shinka.embed.embedding import AsyncEmbeddingClient, EmbeddingClient


def test_get_client_embed_dynamic_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    client, model_name = get_client_embed("openrouter/qwen/qwen3-coder")

    assert model_name == "qwen/qwen3-coder"
    assert "openrouter.ai" in str(client.base_url)


def test_get_async_client_embed_dynamic_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    client, model_name = get_async_client_embed("openrouter/qwen/qwen3-coder")

    assert isinstance(client, openai.AsyncOpenAI)
    assert model_name == "qwen/qwen3-coder"
    assert "openrouter.ai" in str(client.base_url)


def test_get_client_embed_local_openai_inline_url(monkeypatch):
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)

    client, model_name = get_client_embed(
        "local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"
    )

    assert model_name == "BAAI/bge-small-en-v1.5"
    assert str(client.base_url).startswith("http://localhost:8080")


def test_get_async_client_embed_local_openai_inline_url(monkeypatch):
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "test-local-key")

    client, model_name = get_async_client_embed(
        "local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"
    )

    assert isinstance(client, openai.AsyncOpenAI)
    assert model_name == "BAAI/bge-small-en-v1.5"
    assert str(client.base_url).startswith("http://localhost:8080")


def test_get_client_embed_gemini_sets_timeout(monkeypatch):
    captured_kwargs = {}
    fake_client = object()

    def _fake_build_google_genai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        embed_client_module,
        "build_google_genai_client",
        _fake_build_google_genai_client,
    )

    client, model_name = get_client_embed("gemini-embedding-001")

    assert client is fake_client
    assert model_name == "gemini-embedding-001"
    assert captured_kwargs == {"timeout_ms": embed_client_module.TIMEOUT * 1000}


def test_get_async_client_embed_gemini_sets_timeout(monkeypatch):
    captured_kwargs = {}
    fake_client = object()

    def _fake_build_google_genai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        embed_client_module,
        "build_google_genai_client",
        _fake_build_google_genai_client,
    )

    client, model_name = get_async_client_embed("gemini-embedding-001")

    assert client is fake_client
    assert model_name == "gemini-embedding-001"
    assert captured_kwargs == {"timeout_ms": embed_client_module.TIMEOUT * 1000}


def test_sync_openrouter_embedding_unknown_price_defaults_to_zero(monkeypatch):
    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(total_tokens=7),
    )
    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: fake_response,
        )
    )

    monkeypatch.setattr(
        "shinka.embed.embedding.get_client_embed",
        lambda model_name: (fake_client, "qwen/qwen3-coder"),
    )

    client = EmbeddingClient(model_name="openrouter/qwen/qwen3-coder")

    embedding, cost = client.get_embedding("one two")

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == 0.0


def test_async_openrouter_embedding_unknown_price_defaults_to_zero(monkeypatch):
    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(total_tokens=7),
    )

    async def create(**kwargs):
        return fake_response

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=create,
        )
    )

    monkeypatch.setattr(
        "shinka.embed.embedding.get_async_client_embed",
        lambda model_name: (fake_client, "qwen/qwen3-coder"),
    )

    client = AsyncEmbeddingClient(model_name="openrouter/qwen/qwen3-coder")

    embedding, cost = asyncio.run(client.embed_async("one two"))

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == 0.0


def test_sync_local_embedding_unknown_price_defaults_to_zero(monkeypatch):
    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(total_tokens=7),
    )
    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: fake_response,
        )
    )

    monkeypatch.setattr(
        "shinka.embed.embedding.get_client_embed",
        lambda model_name: (fake_client, "BAAI/bge-small-en-v1.5"),
    )

    client = EmbeddingClient(
        model_name="local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"
    )

    embedding, cost = client.get_embedding("one two")

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == 0.0


def test_async_local_embedding_unknown_price_defaults_to_zero(monkeypatch):
    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(total_tokens=7),
    )

    async def create(**kwargs):
        return fake_response

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=create,
        )
    )

    monkeypatch.setattr(
        "shinka.embed.embedding.get_async_client_embed",
        lambda model_name: (fake_client, "BAAI/bge-small-en-v1.5"),
    )

    client = AsyncEmbeddingClient(
        model_name="local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"
    )

    embedding, cost = asyncio.run(client.embed_async("one two"))

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == 0.0


def test_get_client_embed_local_openai_uses_api_key_env_query_param(monkeypatch):
    captured_kwargs = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.base_url = kwargs["base_url"]

    monkeypatch.setenv("CUSTOM_API_KEY", "test-custom-key")
    monkeypatch.setattr("shinka.embed.client.openai.OpenAI", _FakeOpenAI)

    client, model_name = get_client_embed(
        "local/dummy-embed@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
    )

    assert model_name == "dummy-embed"
    assert str(client.base_url).startswith("https://api.example.test/v1")
    assert captured_kwargs["api_key"] == "test-custom-key"


def test_get_async_client_embed_local_openai_missing_api_key_env_raises(monkeypatch):
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CUSTOM_API_KEY"):
        get_async_client_embed(
            "local/dummy-embed@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
        )
