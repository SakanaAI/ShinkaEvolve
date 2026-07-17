"""Regression tests for evaluation result integrity.

Covers the crash-consistency and correctness-gating fixes in
``shinka.core.wrap_eval`` / ``shinka.utils.general``:

- ``metrics.json`` is written before ``correct.json`` (commit marker last), so
  an interruption between the two writes never records a passing program with a
  missing score.
- ``correct.json`` is decoded defensively; a truncated file reads as failed.
- A run that omits ``combined_score`` is marked incorrect, not admitted as a
  score-0 "correct" program.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results
from shinka.utils.general import load_results


def test_metrics_written_before_correct_marker(tmp_path: Path) -> None:
    """The success marker (correct.json) must be the last file written."""
    written: List[str] = []
    real_open = open

    def tracking_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        name = Path(str(file)).name
        if name in {"metrics.json", "correct.json"}:
            written.append(name)
        return real_open(file, *args, **kwargs)

    import builtins

    orig = builtins.open
    builtins.open = tracking_open  # type: ignore[assignment]
    try:
        save_json_results(str(tmp_path), {"combined_score": 1.0}, True, None, False)
    finally:
        builtins.open = orig

    assert written == ["metrics.json", "correct.json"]


def test_missing_correct_marker_reads_as_failed(tmp_path: Path) -> None:
    """If the run dies after metrics.json but before correct.json, it's failed."""
    (tmp_path / "metrics.json").write_text(json.dumps({"combined_score": 5.0}))
    # correct.json intentionally absent (interrupted before the marker write)
    loaded = load_results(str(tmp_path))
    assert loaded["correct"] == {"correct": False}


def test_truncated_correct_json_reads_as_failed(tmp_path: Path) -> None:
    """A truncated correct.json must not crash postprocessing."""
    (tmp_path / "metrics.json").write_text(json.dumps({"combined_score": 5.0}))
    (tmp_path / "correct.json").write_text('{"correct": tr')  # truncated mid-write
    loaded = load_results(str(tmp_path))
    assert loaded["correct"] == {"correct": False}


def _write_program(tmp_path: Path, source: str) -> str:
    program_path = tmp_path / "program_eval.py"
    program_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(program_path)


def test_missing_combined_score_marks_run_incorrect(tmp_path: Path) -> None:
    """An aggregate that omits combined_score yields correct=False, not a 0-score elite."""
    program_path = _write_program(
        tmp_path,
        """
        def run_experiment(seed):
            return {"seed": seed}
        """,
    )

    def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Deliberately omits "combined_score".
        return {"num_runs": len(results)}

    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=str(tmp_path / "res"),
        experiment_fn_name="run_experiment",
        num_runs=2,
        aggregate_metrics_fn=aggregate_metrics,
        run_workers=1,
    )

    assert correct is False
    assert err is not None and "combined_score" in err
