from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

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


def _format_centers(centers: np.ndarray) -> str:
    lines = []
    for idx, (x_coord, y_coord) in enumerate(centers):
        lines.append(f"centers[{idx}] = ({x_coord:.5f}, {y_coord:.5f})")
    return "\n".join(lines)


def _validate(centers: Any, radii: Any, reported_sum: Any, atol: float = 1e-9):
    try:
        centers_arr = np.asarray(centers, dtype=float)
        radii_arr = np.asarray(radii, dtype=float)
        reported = float(reported_sum)
    except Exception as exc:
        return False, f"could not coerce output to numeric arrays: {exc}", None, None, 0.0

    if centers_arr.shape != (N_CIRCLES, 2):
        return False, f"expected centers shape {(N_CIRCLES, 2)}, got {centers_arr.shape}", None, None, 0.0
    if radii_arr.shape != (N_CIRCLES,):
        return False, f"expected radii shape {(N_CIRCLES,)}, got {radii_arr.shape}", None, None, 0.0
    if not np.all(np.isfinite(centers_arr)) or not np.all(np.isfinite(radii_arr)) or not np.isfinite(reported):
        return False, "non-finite center, radius, or reported sum", centers_arr, radii_arr, 0.0
    if np.any(radii_arr < -atol):
        return False, "negative radius found", centers_arr, radii_arr, 0.0

    actual_sum = float(np.sum(radii_arr))
    if abs(actual_sum - reported) > 1e-7:
        return False, f"reported sum {reported:.10f} != actual sum {actual_sum:.10f}", centers_arr, radii_arr, actual_sum

    for idx, ((x_coord, y_coord), radius) in enumerate(zip(centers_arr, radii_arr)):
        if x_coord - radius < -atol or x_coord + radius > 1.0 + atol:
            return False, f"circle {idx} crosses vertical boundary", centers_arr, radii_arr, actual_sum
        if y_coord - radius < -atol or y_coord + radius > 1.0 + atol:
            return False, f"circle {idx} crosses horizontal boundary", centers_arr, radii_arr, actual_sum

    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            distance = float(np.linalg.norm(centers_arr[i] - centers_arr[j]))
            if distance + atol < radii_arr[i] + radii_arr[j]:
                return False, f"circles {i} and {j} overlap", centers_arr, radii_arr, actual_sum

    return True, "", centers_arr, radii_arr, actual_sum


def evaluate(repo_path: Path) -> tuple[dict[str, Any], bool, str]:
    module = _load_candidate(repo_path)
    try:
        centers, radii, reported_sum = module.run_packing()
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "public": {},
            "private": {},
            "extra_data": {},
            "text_feedback": f"run_packing failed: {exc}",
        }, False, str(exc)

    correct, message, centers_arr, radii_arr, actual_sum = _validate(
        centers, radii, reported_sum
    )
    score = float(actual_sum) if correct else 0.0
    public = {
        "num_circles": N_CIRCLES,
        "valid": bool(correct),
        "sum_of_radii": score,
    }
    if centers_arr is not None:
        public["centers"] = _format_centers(centers_arr)

    private = {}
    if radii_arr is not None:
        private["min_radius"] = float(np.min(radii_arr))
        private["max_radius"] = float(np.max(radii_arr))

    return {
        "combined_score": score,
        "public": public,
        "private": private,
        "extra_data": {},
        "text_feedback": message,
    }, correct, message


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
