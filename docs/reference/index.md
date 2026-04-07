# API Reference

Intentionally curated. Documents the runtime surface users are most likely to
compose directly.

---

## Coverage

| Module | Content |
|--------|---------|
| [Core Runtime](core_api.md) | `EvolutionConfig`, `ShinkaEvolveRunner`, `run_shinka_eval` |
| [Database](database_api.md) | `DatabaseConfig`, `Program`, `ProgramDatabase`, prompt DB |
| [Launch](launch_api.md) | `LocalJobConfig`, SLURM configs, `JobScheduler` |
| [LLM](llm_api.md) | `LLMClient`, `AsyncLLMClient`, query helpers, model prioritization |
| [Embeddings](embed_api.md) | `EmbeddingClient`, `AsyncEmbeddingClient`, backend resolution |

---

## Source of Truth

Reference pages render from Python objects and docstrings via `mkdocstrings`.
Signatures stay close to the code.

For config layering and Hydra presets, use [Configuration](../configuration.md)
rather than treating the API pages as the only source of truth.
