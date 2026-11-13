# Local and OpenRouter Models

Shinka supports dynamic LLM backend routing in `LLMClient` and `AsyncLLMClient`.
It also supports dynamic embedding backend routing in `EmbeddingClient` and
`AsyncEmbeddingClient`.
You can use:

- models listed in the provider pricing CSVs (existing behavior)
- dynamic OpenRouter model IDs
- local OpenAI-compatible servers via inline endpoint URIs

---

## Supported Model Name Formats

### 1) Known models (from the runtime pricing catalog)

Shinka conditionally refreshes `https://models.dev/api.json` when a new run
starts. A validated last-known-good cache and the packaged pricing snapshot
keep offline runs working; resumes reuse their recorded run snapshot. Use
`SHINKA_PRICING_MODE=offline` to skip the network check or
`SHINKA_PRICING_MODE=required` to require a valid live response for new runs.

```yaml
evo_config:
  llm_models:
    - gpt-6.1-sol
    - claude-sonnet-5-5
```

The packaged offline catalog includes these current text models:

| Provider | Model IDs | Reference |
|----------|-----------|-----------|
| OpenAI | `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna` | [OpenAI models](https://developers.openai.com/api/docs/models) |
| Anthropic | `claude-fable-5-2`, `claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` | [Claude models](https://platform.claude.com/docs/en/models/overview) |
| Google | `gemini-4-hydrogen`, `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.5-flash-lite` | [Gemini models](https://ai.google.dev/gemini-api/docs/models) |

Live discovery can add models before the packaged snapshot is updated. Discovery
provides pricing and routing metadata; model-specific request settings still
need provider support in Shinka.

For a new run, opt into a current OpenAI and Google mutation pool explicitly:

```yaml
evo_config:
  llm_models:
    - gpt-6-luna
    - gemini-3.8-flash
    - gemini-4-argon
    - gpt-6.1-sol
```

Existing defaults are retained pending a representative benchmark of the new
pool. To use Claude, set `ANTHROPIC_API_KEY` and explicitly include a Claude model
in your pool. When resuming with a frozen `pricing_snapshot.json`, preserve the
original model pool: the recorded catalog may not contain newly added IDs.

Shinka translates `llm_kwargs.reasoning_efforts` into each model's supported
reasoning or thinking controls. GPT-6 Astra and GPT-6.1 Sol require reasoning;
Shinka uses low effort when reasoning is disabled for those models. Claude
Fable 5.1, Fable 5.2, and Opus 5.5 use adaptive
thinking that is always on; disabling reasoning in Shinka cannot turn it off
for those models. Thinking tokens contribute to billed output, so use a suitable
token limit and run budget when comparing models.

Gemini 4 Argon's bundled price is introductory through November 30, 2026;
refresh pricing for runs after that date. Its synchronous output limit is
65,536 tokens; larger outputs require Google's background API, which Shinka
does not currently use. See the [Argon guide](https://ai.google.dev/gemini-api/docs/whats-new-gemini-4-argon).

Older IDs removed from `models.dev` retain explicitly labeled historical prices
in the overlay so their bundled snapshots remain reproducible. A retained ID
does not guarantee that the provider still serves it.

### 2) Dynamic OpenRouter models

Prefix with `openrouter/`:

```yaml
evo_config:
  llm_models:
    - openrouter/qwen/qwen3-coder
    - openrouter/deepseek/deepseek-r1
```

Set env var:

```bash
OPENROUTER_API_KEY=...
```

### 3) Local OpenAI-compatible models

Use `local/<model>@<http(s)://endpoint>`:

```yaml
evo_config:
  llm_models:
    - local/qwen2.5-coder@http://localhost:11434/v1
```

Set optional env var:

```bash
LOCAL_OPENAI_API_KEY=local
```

If not set, Shinka uses `"local"` as a default token.

For a per-model custom key env var, append `api_key_env` to the endpoint URL:

```yaml
evo_config:
  llm_models:
    - local/dummy-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY
```

```bash
CUSTOM_API_KEY=...
```

Shinka strips `api_key_env` from the runtime base URL before creating the client.

---

## Local Embeddings

The same inline local format also works for `embedding_model`.

```yaml
evo_config:
  embedding_model: local/text-embeddings-inference@http://localhost:8080/v1
```

You can also use the same `api_key_env` query parameter for embeddings:

```yaml
evo_config:
  embedding_model: local/dummy-embed@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY
```

Common local embedding backends:

- Hugging Face TEI:
  `local/text-embeddings-inference@http://localhost:8080/v1`
- vLLM or another OpenAI-compatible embedding server:
  `local/BAAI/bge-small-en-v1.5@http://localhost:8000/v1`
- Ollama OpenAI-compatible endpoint:
  `local/embeddinggemma@http://localhost:11434/v1`

---

## Notes

- Dynamic OpenRouter/local model IDs are allowed even if not listed in the catalog.
- If a model has no pricing entry and the provider does not return cost metadata, Shinka records cost as `0.0`.
- Local OpenAI-compatible backend path currently uses chat-completions style calls.
- Local embedding backends use the OpenAI-compatible `/v1/embeddings` path.
- `api_key_env` must reference a single environment variable name, for example `CUSTOM_API_KEY`.
- Structured output is not supported yet for `local/...@...` models.

---

## Applies to Which Clients

These formats work across all LLM consumers that use `LLMClient` / `AsyncLLMClient`, including:

- mutation LLMs (`llm_models`)
- meta LLMs (`meta_llm_models`)
- novelty judge LLMs (`novelty_llm_models`)
- prompt evolution LLMs (`prompt_llm_models`)

For embeddings, the same format applies to:

- code similarity embeddings (`embedding_model`)
