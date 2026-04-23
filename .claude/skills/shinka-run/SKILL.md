---
name: shinka-run
description: Run existing ShinkaEvolve tasks with the `shinka_run` CLI from a task directory (`evaluate.py` + `initial.<ext>`). Use when an agent needs to launch async evolution runs quickly with required `--results_dir`, generation count, and strict namespaced keyword overrides.
---

# Shinka Run CLI Skill
Run a batch of program mutations using ShinkaEvolve's CLI interface.

## When to Use
Use this skill when:
- `evaluate.py` and `initial.<ext>` already exist
- The user wants to run code evolution using the ShinkaEvolve/Shinka library
- You want configurable program evolution runs using explicit CLI args

Do not use this skill when:
- You need to scaffold a new task from scratch (use `shinka-setup`)

## What is ShinkaEvolve?
A framework developed by SakanaAI that combines LLMs with evolutionary algorithms to propose program mutations, that are then evaluated and archived. The goal is to optimize for performance and discover novel scientific insights. 

Repo and documentation: https://github.com/SakanaAI/ShinkaEvolve
Paper: https://arxiv.org/abs/2212.04180

## Workflow

1. Inspect task directory
```bash
ls -la <task_dir>
```
Confirm `evaluate.py` and `initial.<ext>` exist.

2. Inspect CLI reference quickly
```bash
shinka_run --help
```

3. Check model availability before proposing a run
```bash
shinka_models
shinka_models --verbose
```

Validate the exact run config against `shinka_models`:
- Mutation models: every entry in `evo.llm_models` must appear in the `llm` list.
- Meta recommendation models: if `evo.meta_rec_interval` is set and `evo.meta_llm_models` is set, every meta model must appear in the `llm` list.
- Prompt evolution models: if `evo.evolve_prompts=true`, use `evo.prompt_llm_models` when provided, otherwise `evo.llm_models`; every selected model must appear in the `llm` list.
- Embedding model: if `evo.embedding_model` is set, it must appear in the `embedding` list.
- Local OpenAI-compatible models are allowed for LLMs and embeddings via `local/<model>@http(s)://host[:port]/v1`, and these local models are not expected to appear in `shinka_models`.

Important runtime rules:
- Do not assume meta recommendations fall back to `evo.llm_models`. In the current runner, meta recommendations are only enabled when `evo.meta_llm_models` is explicitly set.
- Prompt evolution does fall back to `evo.llm_models` when `evo.prompt_llm_models` is unset.
- Treat `local/<model>@http(s)://host[:port]/v1` values as an explicit exception to the `shinka_models` membership check. Instead, confirm the local endpoint URL and serving status separately before running.
- If any required model is missing from `shinka_models`, stop and ask the user to either change the config or set the missing credentials first.

4. Confirm first-batch configuration with the user
- Minimum: budget scope, generation count, critical overrides.
- Explicitly confirm the mutation LLMs, meta recommendation LLMs, prompt evolution LLMs, and embedding model after checking them against `shinka_models`.
- If unclear, ask before running.
- Do not override any non-confirmed arguments.

5. Launch main run with explicit knobs
```bash
shinka_run \
  --task-dir <task_dir> \
  --results_dir <results_dir> \
  --num_generations 40 \
  --set db.num_islands=3 \
  --set job.time=00:10:00 \
  --set evo.task_sys_msg='<task-specific system message guiding search>'\
  --set evo.llm_models='["gpt-5-mini","gpt-5-nano"]' \
  --set evo.meta_llm_models='["gpt-5-mini"]' \
  --set evo.prompt_llm_models='["gpt-5-mini"]' \
  --set evo.embedding_model='text-embedding-3-small' \
  # Concurrency settings for parallel sampling and evaluation
  --max-evaluation-jobs 2 \
  --max-proposal-jobs 2 \
  --max-db-workers 2
```

6. Verify outputs before handoff
```bash
ls -la <results_dir>
```
Expect artifacts like run log, generation folders, and SQLite DBs.

7. Between-batch handoff (unless explicitly autonomous)
- Summarize outcomes from the finished batch.
- Ask user for the next batch config before running again.
- Explicitly ask: "What new directions should we push next batch? Please include algorithm ideas, constraints, and failure modes to avoid."
- Turn user feedback into a revised system prompt and pass it via `--set evo.task_sys_msg=...` in the next `shinka_run` call.
- If the prompt is long/multiline, put it in a config file and use `--config-fname` instead of shell-escaping.
- Unless the user explicitly wants a fresh run/fork, keep the same `--results_dir` for follow-up batches.

Example next-batch command with feedback-driven prompt:
```bash
shinka_run \
  --task-dir <task_dir> \
  --results_dir <results_dir> \
  --num_generations 20 \
  --set evo.task_sys_msg='<new system prompt derived from user feedback>' \
  --set db.num_islands=3
```

## Batch Control Policy (Required)

Treat one `shinka_run` invocation as one batch of program evaluations/generations.

- Default mode: human-in-the-loop between batches.
- After each batch and before the first, always ask the user what configuration to run next (budget, `--num_generations`, model/settings overrides, concurrency, islands, output path).
- Do not start the next batch until the user confirms the next config.
- Keep `--results_dir` fixed across continuation batches so Shinka can reload prior results.
- Exception: if the user explicitly asks for fully autonomous execution, you may continue across batches without re-asking between runs.
