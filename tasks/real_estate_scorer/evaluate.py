"""
Evaluator for the real estate scorer task.

Fitness: Spearman rank correlation between evolved scores and ground-truth
price_per_m2 rankings on the held-out test set (10 listings).
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from scipy.stats import spearmanr

from shinka.core import run_shinka_eval


def validate_scoring(
    run_output: dict,
) -> Tuple[bool, Optional[str]]:
    """Validate that the scoring function produced valid output."""
    if not isinstance(run_output, dict):
        return False, f"Expected dict output, got {type(run_output)}"

    required_keys = {"scores", "listings", "ground_truth_ranks"}
    missing = required_keys - set(run_output.keys())
    if missing:
        return False, f"Missing keys in output: {missing}"

    scores = run_output["scores"]
    listings = run_output["listings"]
    gt_ranks = run_output["ground_truth_ranks"]

    if len(scores) != len(listings):
        return False, (
            f"Score count ({len(scores)}) != listing count ({len(listings)})"
        )

    if len(scores) != len(gt_ranks):
        return False, (
            f"Score count ({len(scores)}) != ground truth count ({len(gt_ranks)})"
        )

    if not all(isinstance(s, (int, float)) and np.isfinite(s) for s in scores):
        return False, "Scores contain non-finite or non-numeric values"

    return True, "Scoring validation passed."


def aggregate_scoring_metrics(
    results: List[dict], results_dir: str
) -> Dict[str, Any]:
    """Compute Spearman correlation between scores and ground-truth rankings."""
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    output = results[0]
    scores = output["scores"]
    gt_ranks = output["ground_truth_ranks"]

    # Spearman correlation: higher score should correspond to higher rank (rank 1 = best)
    # Since rank 1 = best and we want high score = best, we negate ranks for correlation
    correlation, p_value = spearmanr(scores, [-r for r in gt_ranks])

    # Handle NaN correlation (e.g., all scores identical)
    if np.isnan(correlation):
        correlation = 0.0

    public_metrics = {
        "spearman_correlation": round(float(correlation), 4),
        "p_value": round(float(p_value), 6),
        "num_test_listings": len(scores),
    }

    return {
        "combined_score": float(correlation),
        "public": public_metrics,
        "private": {},
    }


def get_scoring_kwargs(run_index: int) -> Dict[str, Any]:
    """No extra kwargs needed per run."""
    return {}


def main(program_path: str, results_dir: str):
    """Run the real estate scorer evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator(r: List[dict]) -> Dict[str, Any]:
        return aggregate_scoring_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_scoring",
        num_runs=1,
        get_experiment_kwargs=get_scoring_kwargs,
        validate_fn=validate_scoring,
        aggregate_metrics_fn=_aggregator,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real estate scorer evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_scoring')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
