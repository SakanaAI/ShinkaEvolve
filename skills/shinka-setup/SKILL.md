---
name: shinka-setup
description: Create ShinkaEvolve task scaffolds from a target directory and task description, producing `evaluate.py` and `initial.<ext>` (multi-language). Use when asked to set up new ShinkaEvolve tasks, evaluation harnesses, or baseline programs for ShinkaEvolve.
---

# Shinka Task Setup Skill
Create a setup scaffold consisting of an evaluation script and initial solution for an optimization problem given a user's task description. Both ingredients will be used within ShinkaEvolve, a framework combining LLMs with evolutionary algorithms to drive code optimization. 

# When to Use
Invoke this skill when the user:
- Wants to optimize code with LLM-driven code evolution (Shinka/ShinkaEvolve)
- No `evaluate.py` and `initial.<ext>` exist in the working directory

## User Inputs
- Task description + success criteria
- Target language for `initial.<ext>` (if omitted, default to Python)
- What parts of the script to optimize
- Evaluation metric(s) and score direction
- Number of eval runs / seeds
- Required assets or data files
- Dependencies or constraints (runtime, memory)

## Workflow
1. Check if all user inputs are provided and ask the user follow-up questions if not inferrable.
2. Inspect working directory. Detect chosen language + extension. Avoid overwriting existing `evaluate.py` or `initial.<ext>` without consent.
3. Write `initial.<ext>` with a clear evolve region (`EVOLVE-BLOCK` markers or language-equivalent comments) and stable I/O contract.
4. Write `evaluate.py`:
   - Python `initial.py`: call `run_shinka_eval` with `experiment_fn_name`, `get_experiment_kwargs`, `aggregate_metrics_fn`, `num_runs`, and optional `validate_fn`.
   - Non-Python `initial.<ext>`: run candidate program directly (usually via `subprocess`) and write `metrics.json` + `correct.json`.
5. Ensure candidate output schema matches evaluator expectations (tuple/dict for Python module eval, or file/CLI contract for non-Python).
6. Validate draft `evaluate.py` before handoff:
   - Run a smoke test:
     - `python evaluate.py --program_path initial.<ext> --results_dir /tmp/shinka_eval_smoke`
   - Confirm evaluator runs without exceptions.
   - Confirm a metrics `dict` is produced (either from `aggregate_fn` or `metrics.json`) with at least:
     - `combined_score` (numeric),
     - `public` (`dict`),
     - `private` (`dict`),
     - `extra_data` (`dict`),
     - `text_feedback` (string, can be empty).
   - Confirm `correct.json` exists with `correct` (bool) and `error` (string) fields.
7. Ask the user if they want to run the evolution themself or whether to use the `shinka-run` skill: 
    - If the user wants to run evolution manually, add `run_evo.py` plus a `shinka.yaml` config with matching language + `init_program_path`.
    - Ask the user if they want to use the `shinka-run` skill to perform optimization with the agent.

## What is ShinkaEvolve?
A framework developed by SakanaAI that combines LLMs with evolutionary algorithms to propose program mutations, that are then evaluated and archived. The goal is to optimize for performance and discover novel scientific insights. 

Repo and documentation: https://github.com/SakanaAI/ShinkaEvolve
Paper: https://arxiv.org/abs/2212.04180

### Evolution Flow
1. Select parent(s) from archive/population
2. LLM proposes patch (diff, full rewrite, or crossover)
3. Evaluate candidate → `combined_score`
4. If valid, insert into island archive (higher score = better)
5. Periodically migrate top solutions between islands
6. Repeat for N generations

### Core Files To Generate
| File | Purpose |
|------|---------|
| `initial.<ext>` | Starting solution in the chosen language with an evolve region that LLMs mutate |
| `evaluate.py` | Scores candidates and emits metrics/correctness outputs that guide selection |
| `run_evo.py` | (Optional) Launches the evolution loop |
| `shinka.yaml` | (Optional) Config: generations, islands, LLM models, patch types, etc. |

## Quick Install (if Shinka is not set up yet)
Clone and install once before creating/running tasks:

```bash
# Check if shinka is available in workspace environment
python -c "import shinka"

# If not; install via github cloning
git clone https://github.com/SakanaAI/ShinkaEvolve.git
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cd ..
```

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

## Template: `initial.<ext>` (Python example)
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

## Template: `evaluate.py` (Python `run_shinka_eval` path)
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
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(program_path=args.program_path, results_dir=args.results_dir)
```

## Template: `evaluate.py` (non-Python `initial.<ext>` path)
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
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(program_path=args.program_path, results_dir=args.results_dir)
```

## (Optional) Template: `run_evo.py` (async)

See `skills/shinka-setup/scripts/run_evo.py` for an example to edit.

## (Optional) Template: `shinka.yaml`

See `skills/shinka-setup/scripts/shinka.yaml` for an example to edit.

## Notes
- Keep evolve markers tight; only code inside the region should evolve.
- Keep evaluator schema stable (`combined_score`, `public`, `private`, `extra_data`, `text_feedback`).
- Python module path: ensure `experiment_fn_name` matches function name in `initial.py`.
- Non-Python path: ensure evaluator/runtime contract matches `initial.<ext>` CLI/I/O.
- Higher `combined_score` values indicate better performance.
