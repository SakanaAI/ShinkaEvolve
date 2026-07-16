# LLM API

Provider selection, request fan-out, and structured-output querying.

---

## Azure OpenAI v1

Azure LLM model names use `azure-<deployment-name>`. Configure
`AZURE_OPENAI_API_KEY` and set `AZURE_API_ENDPOINT` to the Azure resource root,
such as `https://your-resource.openai.azure.com`. Shinka derives the
`/openai/v1/` base URL and uses the standard OpenAI Responses API client, so a
dated `AZURE_API_VERSION` is not required for LLM calls.

Azure embeddings continue to use the versioned Azure client and require
`AZURE_API_VERSION` in addition to the key and resource endpoint.

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
