---
name: shinka-setup
description: Create ShinkaEvolve task scaffolds from a target directory and task description, producing `evaluate.py` and `initial.<ext>` (multi-language). Use when asked to set up new ShinkaEvolve tasks, evaluation harnesses, or baseline programs for ShinkaEvolve.
---

# ShinkaEvolve Task Setup

## What is ShinkaEvolve?

ShinkaEvolve is a framework combining LLMs with evolutionary algorithms to drive scientific discovery. Key concepts:

- **Population-based evolution**: Maintains a population of programs that evolve over generations
- **LLM mutation operators**: An ensemble of LLMs act as intelligent mutation operators suggesting code improvements
- **Island model**: Multiple evolution "islands" maintain diversity; periodic migration enables knowledge transfer
- **Parallel evaluation**: Supports local execution or SLURM cluster scheduling
- **Archive of solutions**: Tracks successful solutions for inspiration in future generations

### Core Files
| File | Purpose |
|------|---------|
| `initial.<ext>` | Starting solution in the chosen language with an evolve region that LLMs mutate |
| `evaluate.py` | Scores candidates and emits metrics/correctness outputs that guide selection |
| `run_evo.py` | Launches the evolution loop |
| `shinka.yaml` | Config: generations, islands, LLM models, patch types, etc. |

### Evolution Flow
1. Select parent(s) from archive/population
2. LLM proposes patch (diff, full rewrite, or crossover)
3. Evaluate candidate → `combined_score`
4. If valid, insert into island archive (higher score = better)
5. Periodically migrate top solutions between islands
6. Repeat for N generations

Repo and documentation: https://github.com/SakanaAI/ShinkaEvolve
Paper: https://arxiv.org/abs/2212.04180

## Quick Install (if Shinka is not set up yet)
Clone and install once before creating/running tasks:
```bash
git clone https://github.com/SakanaAI/ShinkaEvolve.git
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cd ..
```

## User Inputs (ask for if not provided)
- Target directory (task root)
- Task description + success criteria
- Target language for `initial.<ext>` (if omitted, default to Python)
- Evaluation metric(s) and score direction
- Number of eval runs / seeds
- Required assets or data files
- Dependencies or constraints (runtime, memory)

## Language Support (`initial.<ext>`)
Shinka supports multiple candidate-program languages. Choose one, then keep extension/config/evaluator aligned.

| `evo_config.language` | `initial.<ext>` |
|---|---|
| `python` | `initial.py` |
| `julia` | `initial.jl` |
| `cpp` | `initial.cpp` |
| `cuda` | `initial.cu` |
| `rust` | `initial.rs` |
| `swift` | `initial.swift` |
| `json` / `json5` | `initial.json` |

Rules:
- `evaluate.py` stays the evaluator entrypoint.
- Python candidates: prefer `run_shinka_eval` + `experiment_fn_name`.
- Non-Python candidates: evaluate via `subprocess` and write `metrics.json` + `correct.json`.
- Always set both `evo_config.language` and matching `evo_config.init_program_path`.

## Workflow
1. Inspect working directory. Detect chosen language + extension. Avoid overwriting existing `evaluate.py` or `initial.<ext>` without consent.
2. Write `initial.<ext>` with a clear evolve region (`EVOLVE-BLOCK` markers or language-equivalent comments) and stable I/O contract.
3. Write `evaluate.py`:
   - Python `initial.py`: call `run_shinka_eval` with `experiment_fn_name`, `get_experiment_kwargs`, `aggregate_metrics_fn`, `num_runs`, and optional `validate_fn`.
   - Non-Python `initial.<ext>`: run candidate program directly (usually via `subprocess`) and write `metrics.json` + `correct.json`.
4. Ensure candidate output schema matches evaluator expectations (tuple/dict for Python module eval, or file/CLI contract for non-Python).
5. Validate draft `evaluate.py` before handoff:
   - Run a smoke test:
     - `python evaluate.py --program-path initial.<ext> --results-dir /tmp/shinka_eval_smoke`
     - If evaluator uses snake_case args, run `--program_path` / `--results_dir` instead.
   - Confirm evaluator runs without exceptions.
   - Confirm a metrics `dict` is produced (either from `aggregate_fn` or `metrics.json`) with at least:
     - `combined_score` (numeric),
     - `public` (`dict`),
     - `private` (`dict`),
     - `extra_data` (`dict`),
     - `text_feedback` (string, can be empty).
   - Confirm `correct.json` exists with `correct` (bool) and `error` (string) fields.
6. Optional: 
    - If the user wants to run evolution manually, add `run_evo.py` plus a `shinka.yaml` config with matching language + `init_program_path`.
    - Ask the user if they want to use the `shinka-run` skill to perform optimization with the agent.

## Template: initial.<ext> (Python example)
```py
import random

# EVOLVE-BLOCK-START
def advanced_algo():
    # Implement the evolving algorithm here.
    return 0.0, ""
# EVOLVE-BLOCK-END

def solve_problem(params):
    return advanced_algo()

def run_experiment(random_seed: int | None = None, **kwargs):
    """Main entrypoint called by evaluator."""
    if random_seed is not None:
        random.seed(random_seed)

    score, text = solve_problem(kwargs)
    return float(score), text
```

For non-Python `initial.<ext>`, keep the same idea: small evolve region + deterministic program interface consumed by `evaluate.py`.

## Template: evaluate.py (Python `run_shinka_eval` path)
```py
import argparse
import numpy as np

from shinka.core import run_shinka_eval  # required for results storage


def get_kwargs(run_idx: int) -> dict:
    return {"random_seed": int(np.random.randint(0, 1_000_000_000))}


def aggregate_fn(results: list) -> dict:
    scores = [r[0] for r in results]
    texts = [r[1] for r in results if len(r) > 1]
    combined_score = float(np.mean(scores))
    text = texts[0] if texts else ""
    return {
        "combined_score": combined_score,
        "public": {},
        "private": {},
        "extra_data": {},
        "text_feedback": text,
    }


def validate_fn(result):
    # Return (True, None) or (False, "reason")
    return True, None


def main(program_path: str, results_dir: str):
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=3,
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,  # Optional
    )
    if not correct:
        raise RuntimeError(err or "Evaluation failed")


if __name__ == "__main__":
    # argparse program path & dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-path", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    main(program_path=args.program_path, results_dir=args.results_dir)
```

## Template: evaluate.py (non-Python `initial.<ext>` path)
```py
import argparse
import json
import os
from pathlib import Path


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    # 1) Execute candidate program_path (subprocess / runtime-specific call)
    # 2) Compute task metrics + correctness
    metrics = {
        "combined_score": 0.0,
        "public": {},
        "private": {},
        "extra_data": {},
        "text_feedback": "",
    }
    correct = False
    error = ""

    (Path(results_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (Path(results_dir) / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-path", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    main(program_path=args.program_path, results_dir=args.results_dir)
```

## (Optional) Template: run_evo.py (async)
```py
#!/usr/bin/env python3
import argparse
import asyncio
import yaml

from shinka.core import AsyncEvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """<task-specific system message guiding search>"""


async def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="05:00:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = AsyncEvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config["max_evaluation_jobs"],
        max_proposal_jobs=config["max_proposal_jobs"],
        max_db_workers=config["max_db_workers"],
        debug=False,
        verbose=True,
    )
    await runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config_path))
```

## (Optional) Template: shinka.yaml
```yaml
max_evaluation_jobs: 5
max_proposal_jobs: 5
max_db_workers: 4

db_config:
  db_path: evolution_db.sqlite
  num_islands: 2
  archive_size: 40
  elite_selection_ratio: 0.3
  num_archive_inspirations: 4
  num_top_k_inspirations: 2
  migration_interval: 10
  migration_rate: 0.1
  island_elitism: true
  enforce_island_separation: true
  parent_selection_strategy: weighted
  parent_selection_lambda: 10

evo_config:
  patch_types: [diff, full, cross]
  patch_type_probs: [0.6, 0.3, 0.1]
  num_generations: 100
  max_api_costs: 0.1
  max_patch_resamples: 3
  max_patch_attempts: 3
  max_novelty_attempts: 3
  job_type: local
  language: python  # Set to julia/cpp/cuda/rust/swift/json/json5 as needed
  llm_models: ["gemini-3-flash-preview", "gpt-5-mini", "gpt-5-nano"]
  llm_kwargs:
    temperatures: [0, 0.5, 1.0]
    reasoning_efforts: [min, low]
    max_tokens: 32768
  meta_rec_interval: 40
  meta_llm_models: ["gpt-5-mini"]
  meta_llm_kwargs:
    temperatures: [0]
    max_tokens: 16384
  embedding_model: text-embedding-3-small
  code_embed_sim_threshold: 0.99
  novelty_llm_models: ["gpt-5-nano"]
  novelty_llm_kwargs:
    temperatures: [0]
  init_program_path: initial.py  # Match chosen language extension (e.g., initial.jl)
  llm_dynamic_selection: ucb1
  llm_dynamic_selection_kwargs:
    exploration_coef: 1
  results_dir: results/results_task
```

## Notes
- Keep evolve markers tight; only code inside the region should evolve.
- Keep evaluator schema stable (`combined_score`, `public`, `private`, `extra_data`, `text_feedback`).
- Python module path: ensure `experiment_fn_name` matches function name in `initial.py`.
- Non-Python path: ensure evaluator/runtime contract matches `initial.<ext>` CLI/I/O.
- Higher `combined_score` values indicate better performance.
