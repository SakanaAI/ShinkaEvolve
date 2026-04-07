# CLI Usage

ShinkaEvolve exposes two complementary command-line interfaces:

| CLI | Purpose |
|-----|---------|
| `shinka_launch` | Hydra-driven launcher for preset-based experiments |
| `shinka_run` | Task-directory launcher for explicit, agent-friendly workflows |

---

## When To Use Which

**`shinka_launch`** — Hydra composition from packaged presets, concise override
syntax (`task=...`, `database=...`), workflow centered on named config groups.

**`shinka_run`** — One task directory as the entire interface, explicit
authoritative flags, repeatable namespaced overrides without Hydra group files.

---

## `shinka_launch`

Preserves the original shorthand override flow using the packaged config tree
under `shinka/configs/`.

```bash
shinka_launch
```

```bash
shinka_launch \
  task=circle_packing \
  database=island_large \
  evolution=small_budget \
  cluster=local \
  evo_config.num_generations=20
```

Best for: default baselines, iterating on Hydra presets, composed
cluster/database/evolution/task settings.

---

## `shinka_run`

Direct task launcher for async evolution. Expects a task directory containing
`evaluate.py` and `initial.<ext>`.

### Minimal run

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_agent_run \
  --num_generations 20
```

### Namespaced overrides

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_agent_custom \
  --num_generations 50 \
  --max-evaluation-jobs 6 \
  --set db.num_islands=2 \
  --set job.time=00:10:00 \
  --set job.activate_script=.venv/bin/activate \
  --set evo.llm_models='["gpt-5-mini","gemini-3-flash-preview"]'
```

### With YAML config

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --config-fname shinka_small.yaml \
  --results_dir results/circle_agent_from_yaml \
  --num_generations 50 \
  --set db.num_islands=2
```

---

## Override Grammar

`shinka_run` uses repeatable `--set <namespace>.<field>=<value>` overrides.

### Namespaces

| Namespace | Target |
|-----------|--------|
| `evo` | `EvolutionConfig` |
| `db` | `DatabaseConfig` |
| `job` | `JobConfig` |

### Rules

- Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`
- List and dict values must be valid JSON
- Unknown namespaces or fields fail fast

### Precedence

| Priority | Source |
|----------|--------|
| 1 (lowest) | `--config-fname` YAML |
| 2 | `--set` overrides |
| 3 | `--results_dir` always sets `evo.results_dir` |
| 4 (highest) | `--num_generations` always sets `evo.num_generations` |

---

## Concurrency Controls

The async CLI exposes runner-level concurrency separately from config objects:

| Flag | Controls |
|------|----------|
| `--max-evaluation-jobs` | Concurrent evaluation jobs |
| `--max-proposal-jobs` | Concurrent proposal generation jobs |
| `--max-db-workers` | Async database worker threads |

---

## Local Environments

| Mode | Config |
|------|--------|
| Current interpreter | Default |
| Sourced env | `activate_script` for `.venv/bin/activate` etc. |
| Conda env | `conda_env` for explicit Conda environment |

Useful when the evaluator depends on a task-local virtual environment or
non-default Python interpreter.

---

## Related Pages

| Page | Content |
|------|---------|
| [Getting Started](getting_started.md) | Installation and first-run walkthroughs |
| [Configuration](configuration.md) | Field-level config reference |
| [Examples](examples.md) | Runnable task directories |
| [API Reference](reference/launch_api.md) | Job config types and scheduler APIs |
