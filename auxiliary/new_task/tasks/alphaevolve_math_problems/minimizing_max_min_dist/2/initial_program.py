# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2

    # places points randomly
    np.random.seed(42)
    points = np.random.randn(n, d)

    return points


# EVOLVE-BLOCK-END
