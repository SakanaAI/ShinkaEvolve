import argparse
import numpy as np
import os
from typing import Any, Dict, List, Tuple

from shinka.core.wrap_eval import run_shinka_eval

BENCHMARK = 1.5052939684401607

def validate_sequence(run_output: Any, atol=1e-6) -> Tuple[bool, str]:
    """
    Validates the sequence returned by the evolved program.
    Mimics the checks in the original evaluate_sequence function.
    """
    sequence = run_output

    if not isinstance(sequence, list):
        return False, f"Sequence type expected to be list, received {type(sequence)}"

    if not sequence:
        return False, "Sequence cannot be None or empty."

    # Type and value checks
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float, np.number)):
             return False, "Sequence entries must be integers or floats."
        if np.isnan(x) or np.isinf(x):
             return False, "Sequence cannot contain nans or infs."
        #if x == 0.0:
        #     return False, "Sequence cannot contain zero entries."

    # Simulate the transformation to check the sum constraint
    try:
        seq_array = np.array([float(x) for x in sequence])
        seq_array = np.maximum(0, seq_array)
        seq_array = np.minimum(10000000.0, seq_array)
        
        sum_a = np.sum(seq_array)
        if sum_a < 0.00001:
            return False, f"Sum of sequence entries too close to zero: {sum_a}."
            
    except Exception as e:
        return False, f"Error during validation processing: {str(e)}"

    return True, "Sequence is valid."

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def aggregate_metrics(results: List[Any], results_dir: str) -> Dict[str, Any]:
    """
    Calculates the score based on the validated sequence.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    # We assume num_runs=1, so we take the first result
    sequence = results[0]

    # Apply the logic from the original evaluate_sequence
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(10000000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)
    
    # Calculate c1 as per original script
    c1 = float(2 * n * max_b / (sum_a**2))
    
    # Shinka maximizes 'combined_score'.
    score = BENCHMARK / c1 if c1 != 0 else 0.0

    metrics = {
        "combined_score": score,
        "c1": c1,
        "inv_c1": 1.0 / c1 if c1 != 0 else 0.0,
        "sequence_length": n,
        "sum_a": float(sum_a)
    }
    
    return metrics

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run",
        num_runs=1,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_sequence,
        aggregate_metrics_fn=lambda r: aggregate_metrics(r, results_dir),
    )

    if correct:
        print("Evaluation completed successfully.")
        print("Metrics:", metrics)
    else:
        print(f"Evaluation failed: {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)