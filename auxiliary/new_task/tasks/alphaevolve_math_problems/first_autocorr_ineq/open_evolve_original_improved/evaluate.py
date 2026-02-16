import argparse
import numpy as np
from typing import Any, Dict, List, Tuple

# The core runner from the Shinka framework
from shinka.core import run_shinka_eval

BENCHMARK = 1.5052939684401607


def verify_autocorrelation_solution(run_output: Any, **kwargs) -> Tuple[bool, str]:
    """Verify the autocorrelation solution for UPPER BOUND optimization"""

    # Check if run_output is a list of floats
    if not isinstance(run_output, list) or not all(isinstance(x, (float, int)) for x in run_output):
        return False, "Expected run_output to be a list of floats representing function values."

    # Check non-negativity
    if np.any(run_output < 0.0):
        return False, "Function must be non-negative."

    # Check that the integral is not too small to avoid numerical issues
    if get_integral_sq(run_output) < 1e-8:
        return False, ("Function integral is too small.",)

    return True, ""


def get_integral_sq(sequence: List[float]) -> float:
    """Helper function to compute the integral squared from a sequence."""
    
    n_points = len(sequence)
    dx = 0.5 / n_points
    f_nonneg = np.maximum(sequence, 0.0)

    # Compute the FULL autoconvolution
    autoconv = np.convolve(f_nonneg, f_nonneg, mode="full") * dx

    # The rest of the calculation can be simplified as we now take the max over the whole result
    integral_sq = (np.sum(f_nonneg) * dx) ** 2

    return integral_sq


def get_c1(sequence: List[float]) -> float:
    """Helper function to compute C1 from a sequence."""
    integral_sq = get_integral_sq(sequence)
    computed_c1 = float(np.max(autoconv / integral_sq))
    return computed_c1, integral_sq


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for runs (none needed)."""
    return {}


def aggregate_metrics_fn(results: List[Any]) -> Dict[str, Any]:
    """
    Calculates the final score based on results from all VALID runs.
    
    Args:
        results: A list of `run_output`s from successful validations.

    Returns:
        Dict containing metrics. MUST include "combined_score".
    """

    if not results:
        return {"combined_score": 0.0, "error": "No valid results"}

    sequence = results[0]  # We only have one run, so take the first result
    c1_achieved, integral_sq = get_c1(sequence)
    n_points = len(sequence)

    return {
        "combined_score": BENCHMARK / float(c1_achieved),
        "c1": float(c1_achieved),
        "inv_c1": 1.0 / c1 if c1 != 0 else 0.0,
        "sequence_length": n_points,
        "sum_a": integral_sq
    }


def main(program_path: str, results_dir: str):
    """
    Main entry point for evaluation.
    """
    print(f"Evaluating program: {program_path}")
    
    FUNCTION_NAME_TO_TEST = "run"                       # Must match the function name in initial.py
    NUMBER_OF_RUNS = 1                                  # How many times to run the code
    VALIDATE_FN = verify_autocorrelation_solution       # Validation function to check correctness of each run
    AGGREGATE_METRICS_FN = aggregate_metrics_fn         # Function to compute final metrics from all runs
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        #results_dir=results_dir,
        results_dir=".",
        experiment_fn_name=FUNCTION_NAME_TO_TEST,
        num_runs=NUMBER_OF_RUNS,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=VALIDATE_FN,
        aggregate_metrics_fn=AGGREGATE_METRICS_FN,
    )

    print("Metrics:", metrics)
    
    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")


if __name__ == "__main__":
    # Standard boilerplate for Hydra/Shinka execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    main(args.program_path, args.results_dir)