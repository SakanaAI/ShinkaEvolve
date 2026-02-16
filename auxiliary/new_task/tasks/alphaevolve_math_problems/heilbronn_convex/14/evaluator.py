# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the heilbronn problem for convex regions, with
# 14 points.
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
import itertools
from scipy.spatial import ConvexHull
import sys
import os
from importlib import __import__

BENCHMARK = 0.027835571458482138
NUM_POINTS = 14


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given its vertices p1, p2, and p3."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


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
            points = program.heilbronn_convex14()
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

        min_triangle_area = min(
            [triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)]
        )
        convex_hull_area = ConvexHull(points).volume
        min_area_normalized = min_triangle_area / convex_hull_area

        return {
            "min_area_normalized": float(min_area_normalized),
            "combined_score": float(min_area_normalized / BENCHMARK),
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
