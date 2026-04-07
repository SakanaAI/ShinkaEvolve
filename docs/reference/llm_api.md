# LLM API

Provider selection, request fan-out, and structured-output querying.

---

## `LLMClient`

Batch-oriented synchronous client for sampling candidate responses.

::: shinka.llm.llm.LLMClient
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - batch_query
        - batch_kwargs_query
        - get_kwargs

---

## `AsyncLLMClient`

Async counterpart for the same provider abstraction.

::: shinka.llm.llm.AsyncLLMClient
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - batch_query
        - batch_kwargs_query
        - get_kwargs

---

## Direct Query Helpers

Lower-level provider dispatch:

::: shinka.llm.query.query
    handler: python
    options:
      show_source: false

---

::: shinka.llm.query.query_async
    handler: python
    options:
      show_source: false

---

## Model Prioritization

Bandit-style model prioritization strategies via `shinka.llm.prioritization`.
Dynamically shifts sampling probability across models based on observed utility
and cost.
