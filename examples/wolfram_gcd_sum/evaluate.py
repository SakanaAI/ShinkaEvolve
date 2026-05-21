from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

from shinka.utils.wolfram import (
    build_wolframscript_argv,
    is_wolframscript_available,
    wolframscript_bin,
)

EXPECTED_RESULT = "336784"  # ground truth for N=300
N = 300
NUM_REPS = 3
PER_RUN_TIMEOUT_S = 60
# initial.wl sits next to this evaluator, so it is found regardless of where
# the candidate program lives during an evolution run.
SEED_PROGRAM = Path(__file__).resolve().parent / "initial.wl"


def _run_once(program_path):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        proc = subprocess.run(
            build_wolframscript_argv(["-file", program_path, out_path]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PER_RUN_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return {
                "error": f"rc={proc.returncode} stderr={proc.stderr.strip()[-300:]}"
            }
        output_file = Path(out_path)
        if not output_file.exists():
            return {"error": "no output file"}
        output_text = output_file.read_text(encoding="utf-8")
        if not output_text.strip():
            return {"error": "empty output file"}
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as e:
            return {"error": f"invalid JSON output: {e.msg}"}
        if not isinstance(payload, dict):
            return {"error": "output JSON must be an object"}
        for key in ("result", "time_ms"):
            if key not in payload:
                return {"error": f"output JSON missing {key!r}"}
        try:
            payload["time_ms"] = float(payload["time_ms"])
        except (TypeError, ValueError):
            return {"error": "output JSON field 'time_ms' must be numeric"}
        return payload
    finally:
        Path(out_path).unlink(missing_ok=True)


def _calibrate_baseline():
    """Time the deoptimized seed; candidates are scored as speedup over it.

    ``initial.wl`` is timed through the same harness as every candidate
    (``RepeatedTiming`` inside ``TimeConstrained``), so the baseline and the
    candidates are measured identically. Re-timed on each evaluator
    invocation, which keeps it correct during an evolution run where every
    generation has its own results directory.
    """
    baseline = _run_once(str(SEED_PROGRAM))
    if "error" in baseline:
        raise RuntimeError(f"baseline calibration failed: {baseline['error']}")
    return float(baseline["time_ms"])


def main(program_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    metrics = {"combined_score": -1.0, "public": {}, "private": {}}
    correct = False
    error = ""

    if not is_wolframscript_available():
        error = (
            f"`{wolframscript_bin()}` not found. Install Wolfram Engine or "
            f"Mathematica, or set WOLFRAMSCRIPT_BIN to the binary's absolute path."
        )
    else:
        runs = []
        try:
            baseline_time_ms = _calibrate_baseline()
            for _ in range(NUM_REPS):
                r = _run_once(program_path)
                if "error" in r:
                    error = r["error"]
                    break
                if r.get("result") == "TIMEOUT":
                    error = "candidate timed out (wolfram-side TimeConstrained)"
                    break
                runs.append(r)
        except subprocess.TimeoutExpired:
            error = f"subprocess timeout (>{PER_RUN_TIMEOUT_S}s)"
        except RuntimeError as e:
            error = str(e)

        if runs and not error:
            results_set = {r["result"] for r in runs}
            if len(results_set) > 1:
                error = f"non-deterministic results across runs: {results_set}"
            elif EXPECTED_RESULT not in results_set:
                actual = next(iter(results_set))
                error = f"wrong answer: got {actual!r}, expected {EXPECTED_RESULT!r}"
            else:
                correct = True

        if correct:
            times = [r["time_ms"] for r in runs]
            median_ms = statistics.median(times)
            speedup = baseline_time_ms / max(median_ms, 0.001)
            metrics = {
                "combined_score": speedup,
                "public": {
                    "median_time_ms": round(median_ms, 3),
                    "min_time_ms": round(min(times), 3),
                    "max_time_ms": round(max(times), 3),
                    "speedup_vs_baseline": round(speedup, 2),
                    "baseline_time_ms": round(baseline_time_ms, 3),
                    "n": N,
                },
                "private": {"all_run_times_ms": [round(t, 3) for t in times]},
                "text_feedback": (
                    f"All {NUM_REPS} runs returned the correct answer ({EXPECTED_RESULT}).\n"
                    f"Median runtime: {median_ms:.2f} ms.\n"
                    f"Baseline (deoptimized): {baseline_time_ms:.0f} ms.\n"
                    f"Speedup: {speedup:.2f}x. Higher is better."
                ),
            }
        elif not error:
            error = "no runs completed"

    if not correct:
        metrics = {
            "combined_score": -1.0,
            "public": {"error": error[:300]},
            "private": {},
            "text_feedback": (
                f"Candidate failed: {error}\n"
                f"Expected the function `solve[]` to return the integer {EXPECTED_RESULT} "
                f"(the sum of GCD(i,j) over i,j in 1..{N}). Make sure the result is "
                f"identical, then optimize for speed."
            ),
        }

    correct_path = Path(results_dir) / "correct.json"
    metrics_path = Path(results_dir) / "metrics.json"
    correct_path.write_text(
        json.dumps({"correct": correct, "error": error}, indent=4), encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print(f"Evaluated program: {program_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Correct: {correct}")
    if error:
        print(f"Error: {error}")
    print(f"Combined score: {metrics.get('combined_score', -1.0):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wolfram GCD-sum program")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.wl",
        help="Path to candidate Wolfram program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
