import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


QUERIES_PUBLIC = [100, 500, 1_000, 2_000, 5_000]
QUERIES_PRIVATE = [8_000, 12_000, 16_000, 20_000, 25_000]
TIMEOUT_SECONDS = 20


def _prime_count_prefix(max_n: int) -> list[int]:
    if max_n < 1:
        return [0]

    is_prime = [True] * (max_n + 1)
    is_prime[0] = False
    if max_n >= 1:
        is_prime[1] = False

    p = 2
    while p * p <= max_n:
        if is_prime[p]:
            multiple = p * p
            while multiple <= max_n:
                is_prime[multiple] = False
                multiple += p
        p += 1

    prefix = [0] * (max_n + 1)
    running = 0
    for n in range(max_n + 1):
        if is_prime[n]:
            running += 1
        prefix[n] = running
    return prefix


def _expected_counts(queries: list[int]) -> list[int]:
    prefix = _prime_count_prefix(max(queries))
    return [prefix[q] for q in queries]


def _parse_output(output_path: Path) -> list[int]:
    text = output_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    answers: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            answers.append(int(stripped))
    return answers


def _run_program(program_path: str, queries: list[int]) -> tuple[list[int], float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "queries.txt"
        output_path = tmpdir_path / "answers.txt"
        input_path.write_text("\n".join(str(q) for q in queries), encoding="utf-8")

        cmd = [
            "julia",
            "--startup-file=no",
            program_path,
            str(input_path),
            str(output_path),
        ]
        start = time.perf_counter()
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - start

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            error_parts = ["Julia program failed"]
            if stderr:
                error_parts.append(f"stderr: {stderr}")
            if stdout:
                error_parts.append(f"stdout: {stdout}")
            raise RuntimeError(" | ".join(error_parts))

        if not output_path.exists():
            raise RuntimeError("Julia program did not produce output file")

        answers = _parse_output(output_path)
        return answers, elapsed


def _build_metrics(
    queries: list[int],
    expected: list[int],
    predicted: list[int],
    runtime_seconds: float,
) -> tuple[dict[str, Any], bool, str]:
    if len(predicted) != len(expected):
        msg = (
            f"Output length mismatch. Expected {len(expected)} answers, "
            f"got {len(predicted)}."
        )
        metrics = {
            "combined_score": 0.0,
            "public": {
                "runtime_seconds": runtime_seconds,
                "num_queries": len(queries),
                "num_answers": len(predicted),
                "accuracy": 0.0,
            },
            "private": {},
        }
        return metrics, False, msg

    mismatches: list[dict[str, int]] = []
    for idx, (query, exp, pred) in enumerate(zip(queries, expected, predicted)):
        if exp != pred:
            mismatches.append(
                {
                    "index": idx,
                    "query": query,
                    "expected": exp,
                    "predicted": pred,
                }
            )

    num_correct = len(queries) - len(mismatches)
    accuracy = num_correct / len(queries)
    all_correct = len(mismatches) == 0

    # Encourage correctness first, then runtime optimization.
    combined_score = max(0.0, accuracy * 100.0 - runtime_seconds)

    metrics = {
        "combined_score": combined_score,
        "public": {
            "runtime_seconds": runtime_seconds,
            "num_queries": len(queries),
            "num_correct": num_correct,
            "accuracy": accuracy,
        },
        "private": {
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
        },
    }

    if all_correct:
        return metrics, True, ""
    msg = f"{len(mismatches)} mismatches found. First mismatch: {mismatches[0]}"
    return metrics, False, msg


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    all_queries = QUERIES_PUBLIC + QUERIES_PRIVATE
    expected = _expected_counts(all_queries)

    try:
        predicted, runtime_seconds = _run_program(program_path, all_queries)
        metrics, correct, error = _build_metrics(
            queries=all_queries,
            expected=expected,
            predicted=predicted,
            runtime_seconds=runtime_seconds,
        )
    except FileNotFoundError:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": 0.0, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = "Julia executable not found. Install Julia and make `julia` available in PATH."
    except subprocess.TimeoutExpired:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": TIMEOUT_SECONDS, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = f"Julia program timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": 0.0, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = str(exc)

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
    print(f"Combined score: {metrics.get('combined_score', 0.0):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Julia prime-counting program")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.jl",
        help="Path to candidate Julia program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
