"""Regression tests for the circle-packing notebook's final validator."""

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = (
    Path(__file__).parents[1]
    / "examples"
    / "circle_packing"
    / "viz_circles.ipynb"
)


@pytest.fixture
def verify_circles():
    """Load the validator from the notebook cell used by the example."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def verify_circles" in "".join(cell["source"])
    )
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_circles"
    )
    namespace = {"np": np, "itertools": itertools}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["verify_circles"]


@pytest.mark.parametrize("radius", [-1.0, -np.inf, np.inf, np.nan])
def test_rejects_invalid_radii(verify_circles, radius):
    circles = np.array([[0.25, 0.25, 0.1], [0.75, 0.75, 0.1], [0.5, 0.5, radius]])

    with pytest.raises(AssertionError, match="finite with non-negative radii"):
        verify_circles(circles)


@pytest.mark.parametrize("coordinate", [np.inf, -np.inf, np.nan])
def test_rejects_non_finite_centers(verify_circles, coordinate):
    circles = np.array([[coordinate, 0.5, 0.1]])

    with pytest.raises(AssertionError, match="finite with non-negative radii"):
        verify_circles(circles)


def test_accepts_finite_non_overlapping_circles_and_zero_radius(verify_circles):
    verify_circles(
        np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.2], [0.5, 0.5, 0.0]])
    )
