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

3. Confirm first-batch configuration with the user
- Minimum: budget scope, generation count, critical overrides.
- If unclear, ask before running.
- Do not override any non-confirmed arguments.

4. Launch main run with explicit knobs
```bash
shinka_run \
  --task-dir <task_dir> \
  --results_dir <results_dir> \
  --num_generations 40 \
  --set db.num_islands=3 \
  --set job.time=00:10:00 \
  --set evo.task_sys_msg='<task-specific system message guiding search>'\
  --set evo.llm_models='["gpt-5-mini","gpt-5-nano"]'\
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
- Ask the user for feedback and next directions. Turn those into a new system prompt for the next batch of evolution generations.
- Unless the user explicitly wants a fresh run/fork, keep the same `--results_dir` for follow-up batches.

## Batch Control Policy (Required)

Treat one `shinka_run` invocation as one batch of program evaluations/generations.

- Default mode: human-in-the-loop between batches.
- After each batch and before the first, always ask the user what configuration to run next (budget, `--num_generations`, model/settings overrides, concurrency, islands, output path).
- Do not start the next batch until the user confirms the next config.
- Keep `--results_dir` fixed across continuation batches so Shinka can reload prior results.
- Exception: if the user explicitly asks for fully autonomous execution, you may continue across batches without re-asking between runs.