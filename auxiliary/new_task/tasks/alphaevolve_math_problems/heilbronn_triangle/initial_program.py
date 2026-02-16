# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    points = np.zeros((n, 2))
    return points


# EVOLVE-BLOCK-END
