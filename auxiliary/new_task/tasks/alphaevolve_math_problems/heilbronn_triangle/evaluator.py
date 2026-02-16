# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the heilbronn problem for triangles, with
# 11 points.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import time
import numpy as np
import sys
import os
from importlib import __import__
import itertools

BENCHMARK = 0.036529889880030156
TOL = 1e-6
NUM_POINTS = 11


def check_inside_triangle_wtol(points: np.ndarray, tol: float = 1e-6):
    """Checks that all points are inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).

    Args:
        points: Array of 2D points to check
        tol: Tolerance for numerical errors
    """
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol

        if not (cond1 and cond2 and cond3):
            raise ValueError(
                f"Point ({x}, {y}) is outside the equilateral triangle (tolerance: {tol})."
            )


def triangle_area(a: np.array, b: np.array, c: np.array) -> float:
    return np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        points = None
        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)

            start_time = time.time()
            points = program.heilbronn_triangle11()
            end_time = time.time()
            eval_time = end_time - start_time
        except Exception as err:
            raise err
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, 2):
            raise ValueError(f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS,2)}")

        check_inside_triangle_wtol(points, TOL)

        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([0.5, np.sqrt(3) / 2])
        min_triangle_area = min(
            [triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)]
        )
        min_area_normalized = min_triangle_area / triangle_area(a, b, c)

        return {
            "min_area_normalized": float(min_area_normalized),
            "combined_score": float(min_area_normalized / BENCHMARK),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
