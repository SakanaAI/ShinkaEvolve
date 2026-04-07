# Embeddings API

Code similarity, archive analysis, and provider integration across OpenAI,
Azure, Gemini, OpenRouter, and local OpenAI-compatible backends.

---

## `EmbeddingClient`

Synchronous embedding client with token counting and cost estimation.

::: shinka.embed.embedding.EmbeddingClient
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - count_tokens
        - get_embedding
        - get_column_embedding

---

## `AsyncEmbeddingClient`

Async embedding client used by the async runtime.

::: shinka.embed.embedding.AsyncEmbeddingClient
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - count_tokens
        - get_embedding
        - get_column_embedding

---

## Backend Resolution Helpers

Provider-specific client construction:

::: shinka.embed.client.resolve_embedding_backend
    handler: python
    options:
      show_source: false

---

::: shinka.embed.client.get_client_embed
    handler: python
    options:
      show_source: false

---

::: shinka.embed.client.get_async_client_embed
    handler: python
    options:
      show_source: false
