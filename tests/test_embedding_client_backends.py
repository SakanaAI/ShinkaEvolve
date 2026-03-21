import asyncio
from types import SimpleNamespace

import openai

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
