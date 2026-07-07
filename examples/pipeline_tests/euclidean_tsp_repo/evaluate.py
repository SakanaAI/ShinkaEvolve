from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


INSTANCES = [
    [
        (0.05, 0.10), (0.18, 0.22), (0.24, 0.80), (0.32, 0.47),
        (0.41, 0.15), (0.52, 0.68), (0.63, 0.31), (0.73, 0.88),
        (0.84, 0.52), (0.94, 0.12), (0.11, 0.62), (0.57, 0.04),
    ],
    [
        (0.08, 0.92), (0.16, 0.40), (0.21, 0.12), (0.29, 0.73),
        (0.38, 0.55), (0.49, 0.86), (0.58, 0.19), (0.67, 0.63),
        (0.76, 0.34), (0.82, 0.78), (0.91, 0.08), (0.96, 0.49),
        (0.04, 0.24), (0.44, 0.02),
    ],
    [
        (0.03, 0.03), (0.12, 0.35), (0.19, 0.66), (0.27, 0.91),
        (0.35, 0.18), (0.43, 0.49), (0.51, 0.80), (0.59, 0.11),
        (0.68, 0.42), (0.76, 0.73), (0.84, 0.24), (0.92, 0.55),
        (0.09, 0.78), (0.31, 0.37), (0.62, 0.96), (0.97, 0.86),
    ],
]


def _load_candidate(repo_path: Path):
    module_path = repo_path / "src" / "solver.py"
    spec = importlib.util.spec_from_file_location("candidate_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tour_length(points: list[tuple[float, float]], tour: list[int]) -> float:
    return sum(
        _distance(points[tour[idx]], points[tour[(idx + 1) % len(tour)]])
        for idx in range(len(tour))
    )


def _validate_tour(points: list[tuple[float, float]], raw_tour: Any) -> tuple[bool, str, list[int]]:
    try:
        tour = [int(value) for value in raw_tour]
    except Exception as exc:
        return False, f"tour cannot be converted to integers: {exc}", []

    expected = list(range(len(points)))
    if sorted(tour) != expected:
        return False, "tour must be a permutation of all point indices", tour
    return True, "", tour


def evaluate(repo_path: Path) -> tuple[dict[str, Any], bool, str]:
    module = _load_candidate(repo_path)
    lengths = []
    feedback = []
    all_correct = True

    for idx, points_tuple in enumerate(INSTANCES):
        points = list(points_tuple)
        try:
            raw_tour = module.solve_tsp(points)
        except Exception as exc:
            return {
                "combined_score": 0.0,
                "public": {},
                "private": {},
                "extra_data": {},
                "text_feedback": f"solve_tsp failed on instance {idx}: {exc}",
            }, False, str(exc)

        valid, message, tour = _validate_tour(points, raw_tour)
        if not valid:
            all_correct = False
            feedback.append(f"instance {idx}: {message}")
            lengths.append(float("inf"))
            continue
        lengths.append(_tour_length(points, tour))

    if not all_correct:
        return {
            "combined_score": 0.0,
            "public": {"valid": False},
            "private": {"lengths": lengths},
            "extra_data": {},
            "text_feedback": "; ".join(feedback),
        }, False, "; ".join(feedback)

    mean_length = sum(lengths) / len(lengths)
    combined_score = 100.0 / (1.0 + mean_length)
    return {
        "combined_score": float(combined_score),
        "public": {
            "valid": True,
            "mean_tour_length": float(mean_length),
            "best_instance_length": float(min(lengths)),
            "worst_instance_length": float(max(lengths)),
        },
        "private": {"lengths": [float(length) for length in lengths]},
        "extra_data": {},
        "text_feedback": "",
    }, True, ""


def main(repo_path: str, results_dir: str) -> None:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, correct, error = evaluate(Path(repo_path).resolve())
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
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
