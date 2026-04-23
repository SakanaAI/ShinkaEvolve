"""
Evaluator for the lean_math task.

Each candidate program must expose a `run_experiment(run_idx)` function that
returns a dict with keys: score (int), total (int), solved (list), failed (dict).

combined_score = number of Lean theorems that compiled successfully (max 15).
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval


def _get_kwargs(run_idx: int) -> Dict[str, Any]:
    return {"run_idx": run_idx}


def _validate(run_output: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, dict):
        return False, f"Expected dict from run_experiment, got {type(run_output).__name__}"
    for key in ("score", "total", "solved", "failed"):
        if key not in run_output:
            return False, f"Missing required key '{key}' in run_experiment output"
    if not isinstance(run_output["score"], (int, float)):
        return False, f"'score' must be numeric, got {type(run_output['score']).__name__}"
    return True, None


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = results[0]
    score = float(result["score"])
    total = int(result.get("total", 15))
    solved: list = result.get("solved", [])
    failed: dict = result.get("failed", {})
    pct = f"{100.0 * score / total:.1f}%" if total else "0.0%"

    return {
        "combined_score": score,
        "public": {
            "num_solved": len(solved),
            "num_total": total,
            "pct_solved": pct,
            "solved": solved,
        },
        "private": {
            "failed_details": {k: v for k, v in list(failed.items())[:5]},
        },
    }


def main(program_path: str, results_dir: str) -> None:
    print(f"[lean_math] evaluating: {program_path}")
    os.makedirs(results_dir, exist_ok=True)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,
        run_workers=1,
        get_experiment_kwargs=_get_kwargs,
        validate_fn=_validate,
        aggregate_metrics_fn=_aggregate,
    )

    if correct:
        solved = metrics.get("public", {}).get("num_solved", "?")
        total = metrics.get("public", {}).get("num_total", 15)
        pct = metrics.get("public", {}).get("pct_solved", "?")
        print(f"[lean_math] score: {solved}/{total} ({pct})")
    else:
        print(f"[lean_math] evaluation failed: {error_msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean math proof evaluator")
    parser.add_argument("--program_path", default="initial.py")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
