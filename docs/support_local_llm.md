# Local and OpenRouter LLM Backends

Shinka supports dynamic LLM backend routing in `LLMClient` and `AsyncLLMClient`.
You can use:

- models listed in `shinka/llm/providers/pricing.csv` (existing behavior)
- dynamic OpenRouter model IDs
- local OpenAI-compatible servers via inline endpoint URIs

## Supported Model Name Formats

### 1) Known models (from `pricing.csv`)

```yaml
evo_config:
  llm_models:
    - gpt-5-mini
    - claude-sonnet-4-6
```

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

## Notes

- Dynamic OpenRouter/local model IDs are allowed even if not listed in `pricing.csv`.
- If a model has no pricing entry and the provider does not return cost metadata, Shinka records cost as `0.0`.
- Local OpenAI-compatible backend path currently uses chat-completions style calls.
- Structured output is not supported yet for `local/...@...` models.

## Applies to Which Clients

These formats work across all LLM consumers that use `LLMClient` / `AsyncLLMClient`, including:

- mutation LLMs (`llm_models`)
- meta LLMs (`meta_llm_models`)
- novelty judge LLMs (`novelty_llm_models`)
- prompt evolution LLMs (`prompt_llm_models`)
