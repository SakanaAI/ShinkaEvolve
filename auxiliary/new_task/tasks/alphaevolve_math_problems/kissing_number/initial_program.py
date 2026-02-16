# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    return points


# EVOLVE-BLOCK-END
