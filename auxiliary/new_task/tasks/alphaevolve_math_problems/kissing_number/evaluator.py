# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the kissing number problem on dimension 11.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
from importlib import __import__
import time
import itertools
import numpy as np

DIM = 11
TOL = 1e-6
BENCHMARK = 593


def compute_squared_norm(point: list[int]) -> int:
    """Returns the squared norm of an integer vector using exact computation."""
    return sum(pow(int(x), 2) for x in point)


def verify_sphere_packing(sphere_centers: np.ndarray, tol: float = 1e-6):
    """Checks that after normalizing, the points correspond to a valid sphere packing for kissing numbers.

    Args:
        sphere_centers: the list of sphere centers, of shape [num_spheres, dimension].

    Raises:
        AssertionError: if the sphere packing is not a valid kissing configuration.
    """
    # Rounding to integers to guarantee exact computation throughout.
    sphere_centers = np.around(sphere_centers).astype(np.int64)
    squared_norms = [compute_squared_norm(list(center)) for center in sphere_centers]

    # Checks that the set doesn't contain 0.
    min_squared_norm = min(squared_norms)
    assert min_squared_norm > tol, f"Verification failed because the set contains 0."

    # Checks that the minimum pairwise distance between centers >= the maximum norm of the centers.
    max_squared_norm = max(squared_norms)
    min_squared_distance = min(
        compute_squared_norm(list(a - b)) for a, b in itertools.combinations(sphere_centers, 2)
    )
    assert (
        min_squared_distance >= max_squared_norm
    ), f"Verification failed because the minimum squared distance = {min_squared_distance} < {max_squared_norm} = maximum squared norm."


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            start_time = time.time()
            points = program.kissing_number11()
            end_time = time.time()
            eval_time = end_time - start_time
        except Exception as err:
            raise err
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape[1] != 11:
            raise ValueError(
                f"Invalid shapes: points = {points.shape}, expected ({points.shape[1]},11)"
            )

        verify_sphere_packing(points, TOL)

        num_points = len(points)
        benchmark_ratio = num_points / BENCHMARK
        return {
            "num_points": num_points,
            "combined_score": float(benchmark_ratio),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
