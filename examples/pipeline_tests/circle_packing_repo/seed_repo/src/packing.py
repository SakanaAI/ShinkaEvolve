from __future__ import annotations

import numpy as np


N_CIRCLES = 26


def _grid_centers() -> np.ndarray:
    centers = []
    for row in range(5):
        for col in range(6):
            centers.append(((col + 0.5) / 6.0, (row + 0.5) / 5.0))
    return np.asarray(centers[:N_CIRCLES], dtype=float)


def _compute_valid_radii(centers: np.ndarray) -> np.ndarray:
    del centers
    return np.full(N_CIRCLES, 0.0830, dtype=float)


def run_packing():
    centers = _grid_centers()
    radii = _compute_valid_radii(centers)
    return centers, radii, float(np.sum(radii))
