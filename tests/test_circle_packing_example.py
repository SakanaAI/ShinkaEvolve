from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from examples.circle_packing import evaluate, initial


@pytest.mark.parametrize(
    ("mutation", "expected_score", "error_fragment"),
    [
        pytest.param("", 1.04, None, id="valid"),
        pytest.param(
            "centers[:] = 0.5; radii[:] = 100.0 / 26",
            0.0,
            "outside unit square",
            id="outside-square",
        ),
        pytest.param(
            "centers[:] = 0.5; radii[:] = 0.1",
            0.0,
            "overlap",
            id="overlapping",
        ),
        pytest.param("radii[0] = -0.04", 0.0, "Negative radii", id="negative-radius"),
        pytest.param(
            "centers = centers[:-1]; radii = radii[:-1]",
            0.0,
            "Centers shape incorrect",
            id="wrong-shape",
        ),
        pytest.param("radii[0] = np.nan", 0.0, "Non-finite", id="non-finite"),
    ],
)
def test_circle_packing_persists_validated_score(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected_score: float,
    error_fragment: Optional[str],
) -> None:
    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        "import numpy as np\n\n"
        "def run_packing():\n"
        "    centers = np.array([[0.1 + 0.16 * (i % 6), "
        "0.1 + 0.16 * (i // 6)] for i in range(26)])\n"
        "    radii = np.full(26, 0.04)\n"
        f"    {mutation}\n"
        "    return centers, radii, float(np.sum(radii))\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"

    evaluate.main(str(program_path), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text())
    correct = json.loads((results_dir / "correct.json").read_text())
    is_valid = error_fragment is None
    assert correct["correct"] is is_valid
    assert metrics["combined_score"] == pytest.approx(expected_score)
    assert metrics["num_valid_runs"] == int(is_valid)
    assert metrics["num_invalid_runs"] == int(not is_valid)
    if error_fragment is not None:
        assert error_fragment in correct["error"]
        assert error_fragment in metrics["all_validation_errors"][0]
        assert "combined_score: 0.0" in capsys.readouterr().out

    # Keep the raw packing and reported score available for debugging failures.
    with np.load(results_dir / "extra.npz") as packing:
        np.testing.assert_equal(
            metrics["private"]["reported_sum_of_radii"], packing["reported_sum"]
        )
        assert metrics["public"]["num_circles"] == len(packing["radii"])


def test_circle_packing_seed_keeps_its_score(tmp_path: Path) -> None:
    program_path = Path(evaluate.__file__).with_name("initial.py")
    _, _, expected_score = initial.run_packing()

    evaluate.main(str(program_path), str(tmp_path))

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    correct = json.loads((tmp_path / "correct.json").read_text())
    assert correct["correct"] is True
    assert metrics["combined_score"] == metrics["private"]["reported_sum_of_radii"]
    assert metrics["combined_score"] == pytest.approx(float(expected_score))
    assert metrics["combined_score"] > 0.0
