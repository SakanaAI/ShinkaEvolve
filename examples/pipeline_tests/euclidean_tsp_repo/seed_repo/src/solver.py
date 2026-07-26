from __future__ import annotations

import math


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def solve_tsp(points: list[tuple[float, float]]) -> list[int]:
    if not points:
        return []

    unvisited = set(range(1, len(points)))
    tour = [0]
    current = 0
    while unvisited:
        next_idx = min(unvisited, key=lambda idx: _distance(points[current], points[idx]))
        unvisited.remove(next_idx)
        tour.append(next_idx)
        current = next_idx
    return tour
