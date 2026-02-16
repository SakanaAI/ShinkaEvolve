"""
Evaluator for the 2048 game
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval

from env import render_str, Action


def validate_2048(
    run_output: Tuple[np.ndarray, np.ndarray, float],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates results based on the output of 'run_2048'.
    """
    boards, actions, max_val_reached, reached_2048, reached_max_steps, is_timed_out = (
        run_output
    )
    if not isinstance(boards, np.ndarray):
        boards = np.array(boards)
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)

    for action in actions:
        if action not in list(Action):
            # this should never happen
            msg = (
                f"Unknown action {action} found in actions. "
                "Expected one of "
                f"{[a for a in Action]}"
            )
            return False, msg

    if max_val_reached < 2:
        # this should never happen
        msg = f"Max value reached is lower than the initial value: {max_val_reached}"
        return False, msg

    if len(boards) != len(actions) + 1:
        # this should never happen
        msg = (
            "The numbers of boards and actions are incorrect. "
            f"Expected {len(actions) + 1} boards, got {len(boards)}"
        )
        return False, msg
    if is_timed_out:
        msg = (
            "The function get_best_move() timed out during execution. "
            "Ensure it completes within the time limit."
        )
        return False, msg

    if reached_2048:
        msg = "The game ends by reaching 2048"
    else:
        msg = (
            f"The game ends without reaching 2048, max value reached: {max_val_reached}"
        )

    if reached_max_steps:
        msg += f" by reaching the maximum number of actions ({len(actions)})."
    else:
        msg += f" after {len(actions)} actions."

    return True, msg


def aggregate_2048_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for the 2048 game. Assumes num_runs=1.
    Saves extra.npz with detailed board and action information for the run.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    boards, actions, max_val_reached, reached_2048, reached_max_steps, is_timed_out = (
        results[0]
    )

    num_actions = len(actions)

    # reaching 512 gets 1 score
    # each action costs 0.002 score
    # e.g., reaching 512 or more with less than 500 actions would get a postive reward
    combined_score = max_val_reached / 512 - num_actions * 0.002
    public_metrics = {
        "rendered_final_board": render_str(boards[-1]),
        "num_actions": len(actions),
        "max_val_reached": max_val_reached,
        "reached_2048": reached_2048,
        "reached_max_steps": reached_max_steps,
        "is_timed_out": is_timed_out,
    }
    private_metrics = {}
    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(extra_file, boards=boards, actions=actions)
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the 2048 game evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> Dict[str, Any]:
        return aggregate_2048_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_2048",
        num_runs=num_experiment_runs,
        validate_fn=validate_2048,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2048 evaluator using shinka.eval")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_2048')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
