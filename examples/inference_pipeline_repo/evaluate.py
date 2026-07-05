from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path


INPUTS = [-3.0, -1.5, 0.0, 2.0, 5.0]
EXPECTED = [2.0 * x + 1.0 for x in INPUTS]


def _load_pipeline(repo_path: Path):
    pipeline_path = repo_path / "src" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("candidate_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def evaluate(repo_path: Path) -> tuple[dict, bool, str]:
    module = _load_pipeline(repo_path)
    latencies = []
    prediction = []
    error = ""

    try:
        for _ in range(50):
            start = time.perf_counter()
            prediction = list(module.predict(INPUTS))
            latencies.append(time.perf_counter() - start)
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "public": {},
            "private": {},
            "text_feedback": str(exc),
        }, False, str(exc)

    max_abs_error = max(abs(a - b) for a, b in zip(prediction, EXPECTED))
    correct = max_abs_error <= 1e-9
    p50 = statistics.median(latencies)
    p90 = _percentile(latencies, 0.90)
    p99 = _percentile(latencies, 0.99)
    throughput = len(INPUTS) / max(p50, 1e-12)
    latency_score = 1.0 / (1.0 + p50 * 100000.0)
    combined_score = (1.0 if correct else 0.0) + latency_score

    metrics = {
        "combined_score": combined_score,
        "public": {
            "max_abs_error": max_abs_error,
            "latency_p50_seconds": p50,
            "latency_p90_seconds": p90,
            "latency_p99_seconds": p99,
        },
        "private": {
            "throughput_items_per_second": throughput,
            "peak_memory_bytes": 0,
            "compile_seconds": 0.0,
        },
        "text_feedback": "" if correct else f"max_abs_error={max_abs_error}",
    }
    return metrics, correct, error


def main(repo_path: str, results_dir: str) -> None:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, correct, error = evaluate(Path(repo_path).resolve())
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.repo_path, args.results_dir)
