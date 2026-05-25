"""
ShinkaEvolve evaluator for VerilogEval problems.

Compiles a candidate TopModule against a VerilogEval testbench and reference
solution using Icarus Verilog (iverilog v12), runs the simulation, and parses
the mismatch count to produce a continuous correctness score.

Called by ShinkaEvolve as:
    python evaluate.py --program_path <candidate.sv> --results_dir <dir>

Environment variables:
    VERILOG_EVAL_DIR   Path to the verilog-eval dataset directory
                       (default: ../../../verilog-eval/dataset_spec-to-rtl)
    VERILOG_PROBLEM    Problem ID like "Prob082_lfsr32" (default: Prob082_lfsr32)
    VERILOG_TIMEOUT    Simulation timeout in seconds (default: 60)
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple


def _find_dataset_dir() -> Path:
    env_dir = os.environ.get("VERILOG_EVAL_DIR")
    if env_dir:
        return Path(env_dir)
    candidates = [
        Path(__file__).resolve().parent / ".." / ".." / ".." / "verilog-eval" / "dataset_spec-to-rtl",
        Path.home() / "verilog-eval" / "dataset_spec-to-rtl",
        Path.home() / "Desktop" / "verilog-eval" / "dataset_spec-to-rtl",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Cannot find verilog-eval dataset. Set VERILOG_EVAL_DIR or clone "
        "https://github.com/NVlabs/verilog-eval next to ShinkaEvolve."
    )


def _get_problem_files(dataset_dir: Path, problem_id: str) -> Tuple[Path, Path]:
    test_file = dataset_dir / f"{problem_id}_test.sv"
    ref_file = dataset_dir / f"{problem_id}_ref.sv"
    if not test_file.exists():
        raise FileNotFoundError(f"Testbench not found: {test_file}")
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference not found: {ref_file}")
    return test_file, ref_file


def _parse_simulation_output(output: str) -> Tuple[int, int, str]:
    """Parse iverilog simulation output for mismatch count.

    Returns (mismatches, total_samples, error_text).
    """
    mismatch_pattern = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")
    for line in output.splitlines():
        m = mismatch_pattern.search(line)
        if m:
            return int(m.group(1)), int(m.group(2)), ""

    if "TIMEOUT" in output:
        return -1, 0, "Simulation timed out (infinite loop or latch)"

    return -1, 0, "Could not parse simulation output"


def _classify_compile_error(stderr: str) -> str:
    """Classify iverilog compile error into human-readable category."""
    if "syntax error" in stderr:
        return "syntax_error"
    if "Unable to bind wire/reg/memory `clk'" in stderr:
        return "missing_clk_port"
    if "Unable to bind wire/reg" in stderr:
        return "port_binding_error"
    if "This assignment requires an explicit cast" in stderr:
        return "explicit_cast_required"
    if "is declared here as wire" in stderr:
        return "wire_reg_confusion"
    if "Unknown module type" in stderr:
        return "missing_module"
    if "error" in stderr.lower():
        return "compile_error"
    return "unknown_compile_error"


def evaluate_verilog(
    program_path: str,
    results_dir: str,
    problem_id: str,
    dataset_dir: Path,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Run one VerilogEval problem evaluation.

    Returns metrics dict with combined_score in [0, 100].
    """
    test_file, ref_file = _get_problem_files(dataset_dir, problem_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # --- 1. Compile ---
        sim_vvp = tmpdir_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-Wall",
            "-Winfloop",
            "-Wno-timescale",
            "-g2012",
            "-s", "tb",
            "-o", str(sim_vvp),
            program_path,
            str(test_file),
            str(ref_file),
        ]

        compile_start = time.perf_counter()
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        compile_time = time.perf_counter() - compile_start

        if compile_result.returncode != 0:
            error_class = _classify_compile_error(compile_result.stderr)
            stderr_truncated = compile_result.stderr[:2000]
            return {
                "combined_score": 0.0,
                "public": {
                    "stage": "compile",
                    "error_class": error_class,
                    "compile_time": compile_time,
                    "mismatches": -1,
                    "total_samples": 0,
                    "correctness_ratio": 0.0,
                },
                "private": {"stderr": stderr_truncated},
                "text_feedback": f"Compilation failed ({error_class}): {stderr_truncated[:500]}",
            }

        compile_warnings = compile_result.stderr.strip()

        # --- 2. Simulate (via vvp, works on Windows and Linux) ---
        sim_start = time.perf_counter()
        try:
            sim_result = subprocess.run(
                ["vvp", str(sim_vvp)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {
                "combined_score": 0.0,
                "public": {
                    "stage": "simulate",
                    "error_class": "timeout",
                    "compile_time": compile_time,
                    "sim_time": timeout_seconds,
                    "mismatches": -1,
                    "total_samples": 0,
                    "correctness_ratio": 0.0,
                },
                "private": {},
                "text_feedback": f"Simulation timed out after {timeout_seconds}s. "
                "Likely an infinite loop from a combinational cycle or missing clock edge.",
            }
        sim_time = time.perf_counter() - sim_start

        sim_output = sim_result.stdout + "\n" + sim_result.stderr

        # --- 3. Parse results ---
        mismatches, total_samples, parse_error = _parse_simulation_output(sim_output)

        if mismatches < 0:
            error_msg = parse_error or "Simulation produced no mismatch report"
            return {
                "combined_score": 0.0,
                "public": {
                    "stage": "simulate",
                    "error_class": "no_output" if not parse_error else "timeout",
                    "compile_time": compile_time,
                    "sim_time": sim_time,
                    "mismatches": -1,
                    "total_samples": 0,
                    "correctness_ratio": 0.0,
                },
                "private": {"sim_output": sim_output[:2000]},
                "text_feedback": error_msg,
            }

        correctness_ratio = 1.0 - (mismatches / total_samples) if total_samples > 0 else 0.0
        combined_score = correctness_ratio * 100.0

        # --- 4. Collect per-output hints ---
        hint_pattern = re.compile(r"Hint: Output '(\w+)' has (\d+) mismatches")
        output_mismatches = {}
        for line in sim_output.splitlines():
            m = hint_pattern.search(line)
            if m:
                output_mismatches[m.group(1)] = int(m.group(2))

        feedback_parts = []
        if mismatches == 0:
            feedback_parts.append("All outputs match the reference.")
        else:
            feedback_parts.append(
                f"{mismatches} mismatches out of {total_samples} samples "
                f"(correctness: {correctness_ratio:.1%})."
            )
            for name, count in output_mismatches.items():
                if count > 0:
                    feedback_parts.append(f"  Output '{name}': {count} mismatches")

        if compile_warnings:
            feedback_parts.append(f"Compile warnings: {compile_warnings[:300]}")

        return {
            "combined_score": combined_score,
            "public": {
                "stage": "pass" if mismatches == 0 else "mismatch",
                "compile_time": compile_time,
                "sim_time": sim_time,
                "mismatches": mismatches,
                "total_samples": total_samples,
                "correctness_ratio": correctness_ratio,
                "output_mismatches": output_mismatches,
            },
            "private": {
                "compile_warnings": compile_warnings[:500],
            },
            "text_feedback": "\n".join(feedback_parts),
        }


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    problem_id = os.environ.get("VERILOG_PROBLEM", "Prob082_lfsr32")
    timeout = int(os.environ.get("VERILOG_TIMEOUT", "60"))

    try:
        dataset_dir = _find_dataset_dir()
    except FileNotFoundError as e:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        correct = False
        error = str(e)
        _save_results(results_dir, metrics, correct, error)
        return

    try:
        metrics = evaluate_verilog(
            program_path=program_path,
            results_dir=results_dir,
            problem_id=problem_id,
            dataset_dir=dataset_dir,
            timeout_seconds=timeout,
        )
    except FileNotFoundError as e:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        correct = False
        error = str(e)
        _save_results(results_dir, metrics, correct, error)
        return
    except Exception as e:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}}
        correct = False
        error = str(e)
        _save_results(results_dir, metrics, correct, error)
        return

    correct = metrics.get("combined_score", 0.0) == 100.0
    error = "" if correct else metrics.get("text_feedback", "")
    _save_results(results_dir, metrics, correct, error)

    print(f"Problem: {problem_id}")
    print(f"Score: {metrics['combined_score']:.1f}/100")
    if metrics.get("public", {}).get("mismatches", -1) >= 0:
        pub = metrics["public"]
        print(f"Mismatches: {pub['mismatches']}/{pub['total_samples']}")
        print(f"Compile: {pub.get('compile_time', 0):.2f}s  Simulate: {pub.get('sim_time', 0):.2f}s")


def _save_results(results_dir: str, metrics: dict, correct: bool, error: str):
    correct_path = Path(results_dir) / "correct.json"
    metrics_path = Path(results_dir) / "metrics.json"
    correct_path.write_text(
        json.dumps({"correct": correct, "error": error}, indent=4),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Verilog candidate against VerilogEval")
    parser.add_argument("--program_path", type=str, default="initial.sv")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
