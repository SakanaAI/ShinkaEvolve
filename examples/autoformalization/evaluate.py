import os
import logging
import argparse
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

from shinka.llm.client import get_client_llm
from shinka.utils import validate_lean
from shinka.core import run_shinka_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_proof(run_output: Tuple[str, Optional[str]]) -> Tuple[bool, Optional[str]]:
    """
    Validates the proof generation results based on the output of ``generate_proof``.

    Args:
        run_output(Tuple[str, Optional[str]): the run output, containing
            - file_path (str): the path to a Lean 4 file containing an incomplete proof (may include ``sorry`` s)
            - proof_text (str): the output of ``generate_proof``.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    file_path, proof_text = run_output
    return validate_lean(proof_text, allow_sorry=False, timeout=60, verbose=False)


def aggregate_hypothesis_generation_metrics(results: Tuple[str, str], results_dir: str) -> Dict[str, Any]:
    """
    Aggregates metrics for the generation of hypotheses. Assumes num_runs=1. Saves extra.npz with detailed generation
     information.

    Args:
        results (Tuple[str, str]): the validated output of ``generate_proof``.
        results_dir (str): the path to the directory where to save the results.

    Returns:
        dict: a dictionary of the results.

    """
    print("Aggregation results:", results)
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    path, proof = results

    public_metrics = {
        "proof_length": len(proof),
    }

    private_metrics = {}
    metrics = {
        "combined_score": len(proof),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(extra_file, proof_length=len(results))
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)
    return metrics


def get_proof_generation_kwargs(run_index: int) -> Dict[str, Any]:
    """
    Provides keyword arguments for generating proofs.

    Args:
        run_index (int): the index of the run, added for compatibility with ShinkaEvolve and not used.

    Returns:
        dict: a dictionary of hypothesis generation parameters.
    """
    del run_index  # Unused
    return {
        "sampling_params": {
            "temperature": 0.2,
            "max_tokens": 1024,
            "n": 1,
            "top_p": 0.95,
        },
        "timeout": 60,
    }


def main(program_path: str, results_dir: str, prover_model: str) -> None:
    """
    Run the hypothesis evaluation using shinka.eval

    Args:
        program_path (str): Path to program to evaluate.
        results_dir (str): Dir to save results (metrics.json, correct.json, extra.npz)
        prover_model (str): LLM agent used to construct LEAN proofs based on the initial header and formalization.

    Returns:
        None
    """

    print(f"Evaluating program: {program_path} with {prover_model}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    client, prover_model = get_client_llm(prover_model, False,)

    # Helper functions
    def _aggregator_with_context(
        r: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """A curried function to pass results_dir to the aggregator, extracts the tuple from the list containing 1 element"""
        return aggregate_hypothesis_generation_metrics(r[0], results_dir)

    def _kwargs_with_context(run_index: int) -> dict:
        """A curried function to pass the proof client to the proof solver"""
        return {"model": prover_model, "proof_client": client} | get_proof_generation_kwargs(run_index=run_index)

    num_experiment_runs = 1

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_autoformalization",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=_kwargs_with_context,
        validate_fn=validate_proof,
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
    parser = argparse.ArgumentParser(description="Hypothesis evaluator using shinka.eval")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.lean",
        help="Path to program to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )

    parser.add_argument(
        "--prover_model",
        type=str,
        default="gpt-5-nano",  # or an actual prover like "deepseek-ai/DeepSeek-Prover-V2-7B" (requires local LLM support)
        help="LLM agent used to construct LEAN proofs based on the initial header and formalization.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir, parsed_args.prover_model)
