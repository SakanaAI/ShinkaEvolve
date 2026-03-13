from __future__ import annotations

import os
import logging
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from lean_interact import (
    LeanREPLConfig,
    AutoLeanServer,
    Command,
    TempRequireProject,
    FileCommand,
)
from lean_interact.interface import BaseREPLResponse, LeanError

try:
    from .utils_lean import validate_lean, generate_proof
except ImportError:
    from utils_lean import validate_lean, generate_proof
from shinka.llm.client import get_client_llm
from shinka.core import run_shinka_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LEAN_TIMEOUT = int(os.environ.get("AUTOFORMALIZATION_LEAN_TIMEOUT", "300"))
DEFAULT_PROVER_TIMEOUT = int(os.environ.get("AUTOFORMALIZATION_PROVER_TIMEOUT", "180"))


def _extract_theorem_block(lean_code: str) -> Optional[str]:
    match = re.search(
        r"theorem\s+abelian_group\b.*?:=",
        lean_code,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return match.group(0)


def validate_task_semantics(
    lean_code: Optional[str],
    artifact_name: str = "generated proof",
) -> Tuple[bool, Optional[str]]:
    """Check that the generated theorem still matches the intended task."""
    if not lean_code:
        return False, f"The {artifact_name} is empty."

    theorem_block = _extract_theorem_block(lean_code)
    if theorem_block is None:
        return False, f"The {artifact_name} does not preserve theorem `abelian_group`."

    if re.search(r":\s*True\s*:=", theorem_block):
        return False, f"The {artifact_name} became vacuous (`True`)."

    required_patterns = [
        (r"\[.*Group.*\]", "group context"),
        (r"\ba\b", "generator `a`"),
        (r"\bb\b", "generator `b`"),
    ]
    for pattern, label in required_patterns:
        if re.search(pattern, theorem_block) is None:
            return False, f"The {artifact_name} is missing the intended {label}."

    subgroup_patterns = [
        r"Subgroup",
        r"closure",
        r"\bH\b",
        r"∈\s*H",
    ]
    if not any(re.search(pattern, theorem_block) for pattern in subgroup_patterns):
        return False, f"The {artifact_name} is missing the intended generated-subgroup structure."

    commutativity_patterns = [
        r"a\s*\*\s*b\s*=\s*b\s*\*\s*a",
        r"Commute",
        r"Abelian",
        r"abelian_group",
        r"abelian",
    ]
    if not any(re.search(pattern, theorem_block) for pattern in commutativity_patterns):
        return (
            False,
            f"The {artifact_name} no longer states the intended commutativity/abelian goal.",
        )

    return True, None


def check_lean(
    path_or_str: str,
    timeout: int = DEFAULT_LEAN_TIMEOUT,
) -> BaseREPLResponse | LeanError:
    """
    Plug the generated proof through the Lean 4 compiler.

    Args:
        path_or_str (str): The path to the proof or the proof itself.

    Returns:
        BaseREPLResponse: the output of the Lean compiler.
    """
    project = TempRequireProject(lean_version="v4.24.0", require="mathlib")
    config = LeanREPLConfig(project=project)
    server = AutoLeanServer(config)  # start Lean REPL
    command = (
        FileCommand(path=path_or_str)
        if path_or_str.endswith(".lean")
        else Command(cmd=path_or_str)
    )
    server_output = server.run(command, timeout=timeout)
    logger.info(server_output.messages)
    return server_output


def validate_proof(
    run_output: Tuple[str, Optional[str]],
    timeout: int = DEFAULT_LEAN_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
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
    candidate_text = Path(file_path).read_text(encoding="utf-8")

    semantic_ok, semantic_error = validate_task_semantics(
        candidate_text,
        artifact_name="evolved Lean file",
    )
    if not semantic_ok:
        return False, semantic_error

    semantic_ok, semantic_error = validate_task_semantics(
        proof_text,
        artifact_name="generated proof",
    )
    if not semantic_ok:
        return False, semantic_error

    return validate_lean(
        proof_text,
        allow_sorry=False,
        timeout=timeout,
        verbose=False,
    )


def aggregate_hypothesis_generation_metrics(
    results: Tuple[str, str],
    results_dir: str,
    lean_timeout: int = DEFAULT_LEAN_TIMEOUT,
) -> Dict[str, Any]:
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

    path, lean_cmd = results
    formalization_text = Path(path).read_text(encoding="utf-8")
    candidate_ok, candidate_error = validate_task_semantics(
        formalization_text,
        artifact_name="evolved Lean file",
    )
    proof_ok, proof_error = validate_task_semantics(
        lean_cmd,
        artifact_name="generated proof",
    )

    if not candidate_ok or not proof_ok:
        semantic_error = candidate_error or proof_error
        return {
            "combined_score": 0.0,
            "public": {
                "proof_length": len(lean_cmd) if lean_cmd else 0,
                "formalization_length": len(formalization_text),
            },
            "private": {
                "candidate_semantic_error": candidate_error,
                "proof_semantic_error": proof_error,
            },
            "text_feedback": semantic_error
            or "Generated theorem did not match the intended task.",
        }

    server_output = check_lean(lean_cmd, timeout=lean_timeout)
    if not server_output.lean_code_is_valid(allow_sorry=False):
        messages = server_output
        text_feedback = (
            f"The generated proof:\n{lean_cmd} was invalid. Each error or sorry leads to a -1 penalty."
            f"Please consider the following compiler feedback and update the formalization accordingly:\n"
            f"{messages}"
        )
        combined_score = 0.0
    else:
        text_feedback = ""
        combined_score = float(max(1, 1000 - len(formalization_text)))

    public_metrics = {
        "proof_length": len(lean_cmd),
        "formalization_length": len(formalization_text),
    }

    private_metrics = {}
    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
        "text_feedback": text_feedback,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            proof_length=len(lean_cmd),
            formalization_length=len(formalization_text),
        )
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)
    return metrics


def get_proof_generation_kwargs(run_index: int) -> Dict[str, Any]:
    """
    Provides keyword arguments for generating proofs. Insert your sampling parameters here. The timeout is provided
    in seconds.

    Args:
        run_index (int): the index of the run, added for compatibility with ShinkaEvolve and not used.

    Returns:
        dict: a dictionary of hypothesis generation parameters.
    """
    del run_index  # Unused
    return {
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 16384,
        },
        "timeout": DEFAULT_PROVER_TIMEOUT,
    }


def main(
    program_path: str,
    results_dir: str,
    prover_model: str = "gpt-5.4",
    lean_timeout: int = DEFAULT_LEAN_TIMEOUT,
    prover_timeout: int = DEFAULT_PROVER_TIMEOUT,
) -> None:
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

    client, prover_model, _ = get_client_llm(prover_model)

    # Helper functions
    def _aggregator_with_context(
        r: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """A curried function to pass results_dir to the aggregator, extracts the tuple from the list containing 1 element"""
        return aggregate_hypothesis_generation_metrics(
            r[0],
            results_dir,
            lean_timeout=lean_timeout,
        )

    def _kwargs_with_context(run_index: int) -> dict:
        """A curried function to pass the proof client to the proof solver"""
        return {
            "model": prover_model,
            "proof_client": client,
        } | (
            get_proof_generation_kwargs(run_index=run_index)
            | {"timeout": prover_timeout}
        )

    num_experiment_runs = 1

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name=generate_proof,
        num_runs=num_experiment_runs,
        get_experiment_kwargs=_kwargs_with_context,
        validate_fn=lambda run_output: validate_proof(
            run_output,
            timeout=lean_timeout,
        ),
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
    parser = argparse.ArgumentParser(
        description="Hypothesis evaluator using shinka.eval"
    )
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
    parser.add_argument(
        "--lean_timeout",
        type=int,
        default=DEFAULT_LEAN_TIMEOUT,
        help="Timeout in seconds for Lean validation and compiler checks.",
    )
    parser.add_argument(
        "--prover_timeout",
        type=int,
        default=DEFAULT_PROVER_TIMEOUT,
        help="Timeout in seconds for the prover model API call.",
    )

    parsed_args = parser.parse_args()
    main(
        parsed_args.program_path,
        parsed_args.results_dir,
        parsed_args.prover_model,
        parsed_args.lean_timeout,
        parsed_args.prover_timeout,
    )
