from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


EVALUATOR_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "circle_packing_repo_experiment"
    / "evaluate.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("circle_repo_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "return_statement",
    [
        "return None",
        "return ([['bad']], [1.0], 'bad')",
        "return ([], [])",
        "raise RuntimeError('candidate exploded')",
    ],
)
def test_evaluator_always_writes_result_files_for_malformed_candidates(
    tmp_path, return_statement
):
    evaluator = _load_evaluator()
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "packing.py").write_text(
        f"def run_packing():\n    {return_statement}\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"

    evaluator.main(str(repo), str(results))

    metrics = json.loads((results / "metrics.json").read_text())
    correct = json.loads((results / "correct.json").read_text())
    assert metrics["combined_score"] == 0.0
    assert correct["correct"] is False
    assert correct["error"]
