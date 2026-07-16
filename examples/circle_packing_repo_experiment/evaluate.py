from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


N_CIRCLES = 26


def _load_candidate(repo_path: Path):
    module_path = repo_path / "src" / "packing.py"
    spec = importlib.util.spec_from_file_location("candidate_packing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_centers_string(centers: np.ndarray) -> str:
    """Formats circle centers into a multi-line string for display."""
    return "\n".join(
        [
            f"  centers[{i}] = ({x_coord:.4f}, {y_coord:.4f})"
            for i, (x_coord, y_coord) in enumerate(centers)
        ]
    )


def adapted_validate_packing(
    run_output: Any,
    atol: float = 1e-9,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[np.ndarray], float]:
    """Validate circle packing results based on the output of run_packing."""
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    try:
        if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
            raise ValueError("run_packing must return exactly three values")
        centers, radii, reported_sum = run_output
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        reported_sum = float(reported_sum)
    except Exception as exc:
        msg = f"Could not coerce run_packing output: {exc}"
        return False, msg, None, None, 0.0

    if centers.shape != (N_CIRCLES, 2):
        msg = f"Centers shape incorrect. Expected ({N_CIRCLES}, 2), got {centers.shape}"
        return False, msg, centers, radii, 0.0
    if radii.shape != (N_CIRCLES,):
        msg = f"Radii shape incorrect. Expected ({N_CIRCLES},), got {radii.shape}"
        return False, msg, centers, radii, 0.0

    if (
        not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or not np.isfinite(reported_sum)
    ):
        msg = "Non-finite values found in centers, radii, or reported_sum."
        return False, msg, centers, radii, 0.0

    if np.any(radii < -atol):
        negative_indices = np.where(radii < 0)[0]
        msg = f"Negative radii found for circles at indices: {negative_indices}"
        return False, msg, centers, radii, 0.0

    actual_sum = float(np.sum(radii))
    if not np.isclose(actual_sum, reported_sum, atol=1e-7, rtol=0.0):
        msg = f"Sum of radii ({actual_sum:.6f}) does not match reported ({reported_sum:.6f})"
        return False, msg, centers, radii, actual_sum

    for i in range(N_CIRCLES):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            return False, msg, centers, radii, actual_sum

    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg, centers, radii, actual_sum

    return True, msg, centers, radii, actual_sum


def evaluate(repo_path: Path, results_dir: Path) -> tuple[dict[str, Any], bool, str]:
    try:
        module = _load_candidate(repo_path)
        run_output = module.run_packing()
        correct, error_msg, centers, radii, actual_sum = adapted_validate_packing(
            run_output
        )
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "public": {},
            "private": {},
            "text_feedback": f"run_packing failed: {exc}",
        }, False, str(exc)

    try:
        reported_sum = float(run_output[2])
    except Exception:
        reported_sum = actual_sum
    score = actual_sum if correct else 0.0

    public: dict[str, Any] = {
        "num_circles": N_CIRCLES,
        "valid": bool(correct),
        "centers_str": format_centers_string(centers) if centers is not None else "",
    }
    private: dict[str, Any] = {"reported_sum_of_radii": reported_sum}

    if radii is not None:
        private["min_radius"] = float(np.min(radii))
        private["max_radius"] = float(np.max(radii))

    if centers is not None and radii is not None:
        try:
            np.savez(
                results_dir / "extra.npz",
                centers=centers,
                radii=radii,
                reported_sum=reported_sum,
            )
        except Exception as exc:
            private["extra_npz_save_error"] = str(exc)

    metrics = {
        "combined_score": float(score),
        "public": public,
        "private": private,
        "text_feedback": error_msg or "",
    }
    return metrics, correct, error_msg or ""


def main(repo_path: str, results_dir: str) -> None:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics, correct, error = evaluate(Path(repo_path).resolve(), output_dir)
    except Exception as exc:
        correct = False
        error = f"Evaluator failed safely: {exc}"
        metrics = {
            "combined_score": 0.0,
            "public": {},
            "private": {},
            "text_feedback": error,
        }
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
