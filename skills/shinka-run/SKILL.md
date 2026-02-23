---
name: shinka-run
description: Run existing ShinkaEvolve tasks with the `shinka_run` CLI from a task directory (`evaluate.py` + `initial.<ext>`). Use when an agent needs to launch async evolution runs quickly with required `--results_dir`, generation count, and strict namespaced keyword overrides.
---

# Shinka Run CLI Skill

## Purpose

Use this skill when a task is already scaffolded and the agent should launch evolution from CLI, not by writing a custom `run_evo.py`.

`shinka_run` is agent-focused:
- No Hydra required
- Async runner under the hood
- Required `--task-dir`, `--results_dir`, `--num_generations`
- Strict override syntax via `--set`

## When to Use

Use this skill when:
- `evaluate.py` and `initial.<ext>` already exist
- You need quick evolution runs from terminal
- You want reproducible runs using explicit CLI args
- You want agents to control config knobs safely with fail-fast validation

Do not use this skill when:
- You need to scaffold a new task from scratch (use `shinka-setup`)

## Task Directory Contract

`--task-dir` must contain:
- `evaluate.py`
- `initial.<ext>` (e.g., `initial.py`, `initial.jl`)

If either is missing, `shinka_run` exits non-zero.

## CLI Interface

### Required Arguments
- `--task-dir PATH`: directory with evaluator + initial candidate
- `--results_dir PATH`: output directory
- `--num_generations N`: generation count (`N > 0`)

### Optional Arguments
- `--set NS.FIELD=VALUE`: repeatable override entries
- `--max-evaluation-jobs INT`: async evaluation concurrency
- `--max-proposal-jobs INT`: async proposal concurrency
- `--max-db-workers INT`: async DB workers
- `--verbose`: verbose logging
- `--debug`: extra async diagnostics

### Override Namespaces
- `evo.<field>` -> `EvolutionConfig`
- `db.<field>` -> `DatabaseConfig`
- `job.<field>` -> `LocalJobConfig`

Unknown namespace or field fails fast.

### Value Parsing Rules
- Scalars: `int`, `float`, `bool`, `str`
- Lists/dicts: valid JSON required
- Bool accepted values:
  - `true`, `false`, `1`, `0`, `yes`, `no` (case-insensitive)

### Precedence Rules
- `--results_dir` always sets `evo.results_dir`
- `--num_generations` always sets `evo.num_generations`
- Even if overridden via `--set`, required flags win
- For resume/continuation across batches, always reuse the same `--results_dir`.
- Changing `--results_dir` creates a separate run history and will not reload previous results/state.

## Batch Control Policy (Required)

Treat one `shinka_run` invocation as one batch of program evaluations/generations.

- Default mode: human-in-the-loop between batches.
- After each batch and before the first, always ask the user what configuration to run next (budget, `--num_generations`, model/settings overrides, concurrency, islands, output path).
- Do not start the next batch until the user confirms the next config.
- Keep `--results_dir` fixed across continuation batches so Shinka can reload prior results.
- Exception: if the user explicitly asks for fully autonomous execution, you may continue across batches without re-asking between runs.

## Standard Agent Workflow

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
- Minimum: budget scope, generation count, results directory, critical overrides.
- If unclear, ask before running.

4. Run a short smoke evolution first
```bash
shinka_run \
  --task-dir <task_dir> \
  --results_dir <results_dir_smoke> \
  --num_generations 2 \
  --verbose
```

5. Launch main run with explicit knobs
```bash
shinka_run \
  --task-dir <task_dir> \
  --results_dir <results_dir_main> \
  --num_generations 40 \
  --set evo.max_parallel_jobs=6 \
  --set db.num_islands=3 \
  --set job.time=00:10:00 \
  --set evo.llm_models='["gpt-5-mini","gpt-5-nano"]'
```

6. Verify outputs before handoff
```bash
ls -la <results_dir_main>
```
Expect artifacts like run log, generation folders, and SQLite DBs.

7. Between-batch handoff (unless explicitly autonomous)
- Summarize outcomes from the finished batch.
- Ask user for the next batch config before running again.
- Unless the user explicitly wants a fresh run/fork, keep the same `--results_dir` for follow-up batches.

## Practical Command Patterns

### Minimal
```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_min \
  --num_generations 10
```

### JSON Override
```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_json \
  --num_generations 20 \
  --set evo.llm_models='["gpt-5-mini"]' \
  --set job.extra_cmd_args='{"seed":42}'
```

### Concurrency Override
```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_async \
  --num_generations 30 \
  --max-evaluation-jobs 8 \
  --max-proposal-jobs 8 \
  --max-db-workers 4
```

## Validation Checklist for Agents

Before running:
- task dir exists
- `evaluate.py` exists
- `initial.<ext>` exists
- results dir path chosen
- generation budget set
- all `--set` keys namespaced (`evo/db/job`)

After running:
- command exited 0
- `results_dir` created and populated
- evolution log exists
- at least generation 0 artifacts exist

## Common Errors and Fixes

### `Invalid override ... Expected NS.FIELD=VALUE`
Fix:
- include namespace + equals sign
- example: `--set evo.max_parallel_jobs=4`

### `Unknown field 'evo.xxx'`
Fix:
- use a real dataclass field name from:
  - `EvolutionConfig`
  - `DatabaseConfig`
  - `LocalJobConfig`

### `Invalid JSON for ...`
Fix:
- lists/dicts must be JSON:
  - list: `'["a","b"]'`
  - dict: `'{"k":"v"}'`

### Missing task files
Fix:
- ensure `<task_dir>/evaluate.py`
- ensure `<task_dir>/initial.<ext>`

## Notes

- Use small smoke runs first (`--num_generations 1-3`) before expensive runs.
- Keep full CLI invocation in logs for reproducibility.
