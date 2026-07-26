"""Constructor-based circle packing for n=26 circles."""

import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    n = 26
    centers = np.zeros((n, 2))

    centers[0] = [0.5, 0.5]

    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    centers = np.clip(centers, 0.01, 0.99)

    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """
    Compute maximum possible radii for fixed centers without overlaps
    and while staying within the unit square.
    """
    n = centers.shape[0]
    radii = np.ones(n)

    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

                    if radii[i] + radii[j] > dist:
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale

    radii = np.maximum(radii, 0.0)
    return radii


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii = construct_packing()
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
