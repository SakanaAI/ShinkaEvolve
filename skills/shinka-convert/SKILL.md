---
name: shinka-convert
description: Convert an existing codebase in the current working directory into a ShinkaEvolve task directory by snapshotting the relevant code, adding evolve blocks, and generating `evaluate.py` plus Shinka runner/config files. Use when the user wants to optimize existing code with Shinka instead of creating a brand-new task from a natural-language description.
---

# Shinka Convert Skill
Use this skill to turn an existing project into a Shinka-ready task.

This is the alternative starting point to `shinka-setup`:
- `shinka-setup`: new task from natural-language task description
- `shinka-convert`: existing codebase to Shinka task conversion

After conversion, the user should still be able to use `shinka-run`.

## When to Use
Invoke this skill when the user:
- Wants to optimize an existing script or repo with Shinka/ShinkaEvolve
- Mentions adapting current code to Shinka output signatures, `metrics.json`, `correct.json`, or `EVOLVE-BLOCK` markers
- Wants a sidecar Shinka task generated from the current working directory

Do not use this skill when:
- The user wants a brand-new task scaffold from only a natural-language description
- `evaluate.py` and `initial.<ext>` already exist and the user only wants to launch evolution; use `shinka-run`

## User Inputs
Start from freeform instructions, then ask follow-ups only if high-impact details are missing.

Collect:
- What behavior or file/function to optimize
- Score direction and main metric
- Constraints: correctness, runtime, memory, determinism, style, allowed edits
- Whether original source must remain untouched
- Any required data/assets/dependencies

## Default Output
Generate a sidecar task directory at `./shinka_task/` unless the user requests another path.

The task directory should contain:
- `evaluate.py`
- `run_evo.py`
- `shinka.yaml`
- `initial.<ext>`
- A copied snapshot of the minimal runnable source subtree needed for evaluation

Do not edit the original source tree unless the user explicitly requests in-place conversion.

## Workflow
1. Inspect the current working directory.
   - Identify language, entrypoints, package/module layout, dependencies, and current outputs.
   - Prefer concrete evidence from the code over guesses.
2. Infer the evolvable region from the user's instructions.
   - If ambiguous, ask targeted follow-ups.
   - Keep the mutable region as small as practical.
3. Choose the minimal runnable snapshot scope.
   - Copy only the source subtree needed to execute the task in isolation.
   - Avoid repo-wide snapshots unless imports/runtime make that necessary.
4. Create the sidecar task directory.
   - Default: `./shinka_task/`
   - Avoid overwriting an existing task dir without consent.
5. Rewrite the snapshot into a stable Shinka contract.
   - Preserve original behavior outside the evolvable region.
   - Keep CLI behavior intact where practical.
   - Ensure the evolvable candidate entry file is named `initial.<ext>` so `shinka-run` can detect it.
   - Add tight `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers.
6. Generate the evaluator path.
   - Python: prefer exposing `run_experiment(...)` and use `run_shinka_eval`.
   - Non-Python: use `subprocess` and write `metrics.json` plus `correct.json`.
7. Generate `run_evo.py` and `shinka.yaml`.
   - Ensure `init_program_path` and `language` match the candidate file.
   - Keep the output directly compatible with `shinka-run`.
8. Smoke test before handoff.
   - Run `python evaluate.py --program_path <initial file> --results_dir /tmp/shinka_convert_smoke`
   - Confirm evaluator runs without exceptions.
   - Confirm required metrics/correctness outputs are written.
9. Ask the user for the next step.
   - Either run evolution manually
   - Or use the `shinka-run` skill

## Conversion Strategy by Language
### Python
- Preferred path: expose `run_experiment(...)` in the snapshot and evaluate via `run_shinka_eval`
- If the existing code is CLI-only, add a thin wrapper in the snapshot rather than forcing a subprocess evaluator unless imports are too brittle
- Keep imports relative to the copied task snapshot stable

### Non-Python
- Keep the candidate program executable in its own runtime
- Use Python `evaluate.py` as the Shinka entrypoint
- Write `metrics.json` and `correct.json` in `results_dir`

## Required Evaluator Contract
Metrics must include:
- `combined_score`
- `public`
- `private`
- `extra_data`
- `text_feedback`

Correctness must include:
- `correct`
- `error`

Higher `combined_score` values indicate better performance unless the user explicitly defines an inverted metric that you transform during aggregation.

## Python Conversion Template
Prefer shaping the copied program like this:

```py
from __future__ import annotations

# EVOLVE-BLOCK-START
def optimize_me(...):
    ...
# EVOLVE-BLOCK-END

def run_experiment(random_seed: int | None = None, **kwargs):
    ...
    return score, text_feedback
```

And the evaluator:

```py
from shinka.core import run_shinka_eval

def main(program_path: str, results_dir: str):
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=3,
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )
    if not correct:
        raise RuntimeError(err or "Evaluation failed")
```

## Non-Python Conversion Template
Use `evaluate.py` to run the candidate and write outputs:

```py
import json
import os
from pathlib import Path

def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    metrics = {
        "combined_score": 0.0,
        "public": {},
        "private": {},
        "extra_data": {},
        "text_feedback": "",
    }
    correct = {"correct": False, "error": ""}

    (Path(results_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(results_dir) / "correct.json").write_text(json.dumps(correct, indent=2))
```

## Bundled Assets
- Use `scripts/run_evo.py` as the starting runner template
- Use `scripts/shinka.yaml` as the starting config template

## Notes
- Keep evolve regions tight; do not make the whole project mutable by default
- Preserve correctness checks outside the evolve region where possible
- Prefer deterministic evaluation and stable seeds
- If the converted task is ready, offer to continue with `shinka-run`
