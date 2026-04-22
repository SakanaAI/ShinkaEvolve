import asyncio
from types import SimpleNamespace

import pytest

from shinka.embed import client as embed_client
from shinka.embed.embedding import AsyncEmbeddingClient, EmbeddingClient
from shinka.embed.providers.pricing import (
    get_model_price,
    get_provider,
    model_exists,
)


MODEL_NAME = "gemini-embedding-2-preview"


class _FakeGoogleModels:
    def __init__(self, total_tokens=11, count_tokens_exc=None):
        self.total_tokens = total_tokens
        self.count_tokens_exc = count_tokens_exc
        self.calls = []

    def count_tokens(self, *, model, contents, config=None):
        self.calls.append(("count_tokens", model, contents))
        if self.count_tokens_exc is not None:
            raise self.count_tokens_exc
        return SimpleNamespace(total_tokens=self.total_tokens)

    def embed_content(self, *, model, contents, config=None):
        self.calls.append(("embed_content", model, contents))
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )


class _FakeGoogleClient:
    def __init__(self, models):
        self.models = models


def test_new_gemini_embedding_model_is_registered():
    assert model_exists(MODEL_NAME)
    assert get_provider(MODEL_NAME) == "google"
    assert get_model_price(MODEL_NAME) == pytest.approx(0.20 / 1_000_000)


def test_get_client_embed_resolves_new_model_to_google(monkeypatch):
    captured = {}
    fake_client = object()

    def _fake_build_google_genai_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        embed_client,
        "build_google_genai_client",
        _fake_build_google_genai_client,
    )

    client, model_name = embed_client.get_client_embed(MODEL_NAME)

    assert client is fake_client
    assert model_name == MODEL_NAME
    assert captured == {"timeout_ms": embed_client.TIMEOUT * 1000}


def test_sync_google_embedding_uses_token_count_for_cost(monkeypatch):
    fake_models = _FakeGoogleModels(total_tokens=11)
    fake_client = _FakeGoogleClient(fake_models)

    monkeypatch.setattr(
        "shinka.embed.embedding.get_client_embed",
        lambda model_name: (fake_client, model_name),
    )

    client = EmbeddingClient(model_name=MODEL_NAME)

    embedding, cost = client.get_embedding("one two")

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == pytest.approx(11 * (0.20 / 1_000_000))
    assert fake_models.calls == [
        ("count_tokens", f"models/{MODEL_NAME}", "one two"),
        ("embed_content", f"models/{MODEL_NAME}", "one two"),
    ]


def test_async_google_embedding_uses_token_count_for_cost(monkeypatch):
    fake_models = _FakeGoogleModels(total_tokens=17)
    fake_client = _FakeGoogleClient(fake_models)

    monkeypatch.setattr(
        "shinka.embed.embedding.get_async_client_embed",
        lambda model_name: (fake_client, model_name),
    )

    client = AsyncEmbeddingClient(model_name=MODEL_NAME)

    embedding, cost = asyncio.run(client.embed_async("one two"))

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == pytest.approx(17 * (0.20 / 1_000_000))
    assert fake_models.calls == [
        ("count_tokens", f"models/{MODEL_NAME}", "one two"),
        ("embed_content", f"models/{MODEL_NAME}", "one two"),
    ]


def test_sync_google_embedding_falls_back_when_token_count_fails(monkeypatch):
    fake_models = _FakeGoogleModels(total_tokens=99, count_tokens_exc=RuntimeError("boom"))
    fake_client = _FakeGoogleClient(fake_models)

    monkeypatch.setattr(
        "shinka.embed.embedding.get_client_embed",
        lambda model_name: (fake_client, model_name),
    )

    client = EmbeddingClient(model_name=MODEL_NAME)

    embedding, cost = client.get_embedding("one two")

    assert embedding == [0.1, 0.2, 0.3]
    assert cost == pytest.approx(2 * (0.20 / 1_000_000))
    assert fake_models.calls == [
        ("count_tokens", f"models/{MODEL_NAME}", "one two"),
        ("embed_content", f"models/{MODEL_NAME}", "one two"),
    ]
