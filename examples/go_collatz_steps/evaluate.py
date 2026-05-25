import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


QUERIES_PUBLIC = [1, 7, 27, 97, 871, 6_171]
QUERIES_PRIVATE = [77_031, 106_239, 216_367, 410_011, 837_799]
TIMEOUT_SECONDS = 20


def _collatz_steps(n: int) -> int:
    steps = 0
    value = n
    while value != 1:
        if value % 2 == 0:
            value //= 2
        else:
            value = 3 * value + 1
        steps += 1
    return steps


def _expected_steps(queries: list[int]) -> list[int]:
    return [_collatz_steps(query) for query in queries]


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

        cmd = ["go", "run", program_path, str(input_path), str(output_path)]
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
            error_parts = ["Go program failed"]
            if stderr:
                error_parts.append(f"stderr: {stderr}")
            if stdout:
                error_parts.append(f"stdout: {stdout}")
            raise RuntimeError(" | ".join(error_parts))

        if not output_path.exists():
            raise RuntimeError("Go program did not produce output file")

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


def _failure_metrics(runtime_seconds: float = 0.0) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "public": {"runtime_seconds": runtime_seconds, "accuracy": 0.0},
        "private": {},
    }


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    all_queries = QUERIES_PUBLIC + QUERIES_PRIVATE
    expected = _expected_steps(all_queries)

    try:
        predicted, runtime_seconds = _run_program(program_path, all_queries)
        metrics, correct, error = _build_metrics(
            queries=all_queries,
            expected=expected,
            predicted=predicted,
            runtime_seconds=runtime_seconds,
        )
    except FileNotFoundError:
        metrics = _failure_metrics()
        correct = False
        error = "Go executable not found. Install Go and make `go` available in PATH."
    except subprocess.TimeoutExpired:
        metrics = _failure_metrics(TIMEOUT_SECONDS)
        correct = False
        error = f"Go program timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        metrics = _failure_metrics()
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
    parser = argparse.ArgumentParser(
        description="Evaluate Go Collatz stopping-time program"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.go",
        help="Path to candidate Go program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
