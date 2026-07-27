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
import errno
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shinka.core import run_shinka_eval
from shinka.core import wrap_eval
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


def test_successful_result_syncs_files_and_correct_marker_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced_descriptors: List[int] = []
    monkeypatch.setattr(os, "fsync", synced_descriptors.append)

    save_json_results(str(tmp_path), {"combined_score": 1.0}, True, None, False)

    expected_syncs = 2 if os.name == "nt" else 3
    assert len(synced_descriptors) == expected_syncs


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_unsupported_directory_fsync_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_directory_fsync(_fd: int) -> None:
        raise OSError(errno.ENOTSUP, "directory fsync unsupported")

    monkeypatch.setattr(os, "fsync", reject_directory_fsync)

    wrap_eval._fsync_directory(str(tmp_path))


def test_final_directory_sync_failure_invalidates_correct_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_directory_fsync(_path: str) -> None:
        raise OSError(errno.EIO, "simulated directory sync failure")

    monkeypatch.setattr(wrap_eval, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="simulated directory sync failure"):
        save_json_results(str(tmp_path), {"combined_score": 1.0}, True, None, False)

    loaded = load_results(str(tmp_path))
    assert loaded["correct"] == {"correct": False}


def test_missing_correct_marker_reads_as_failed(tmp_path: Path) -> None:
    """If the run dies after metrics.json but before correct.json, it's failed."""
    (tmp_path / "metrics.json").write_text(json.dumps({"combined_score": 5.0}))
    # correct.json intentionally absent (interrupted before the marker write)
    loaded = load_results(str(tmp_path))
    assert loaded["correct"] == {"correct": False}


def test_interrupted_rerun_does_not_reuse_previous_correct_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "correct.json").write_text(
        json.dumps({"correct": True}), encoding="utf-8"
    )

    def interrupt_metrics_fsync(_fd: int) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(wrap_eval, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(os, "fsync", interrupt_metrics_fsync)

    with pytest.raises(OSError, match="simulated interruption"):
        save_json_results(
            str(tmp_path),
            {"combined_score": -5.0},
            False,
            "new evaluation failed",
            verbose=False,
        )

    loaded = load_results(str(tmp_path))
    assert loaded["correct"] == {"correct": False}


def test_evaluation_interruption_cannot_reuse_previous_correct_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "correct.json").write_text(
        json.dumps({"correct": True}), encoding="utf-8"
    )

    def interrupt_evaluation(_program_path: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(wrap_eval, "load_program", interrupt_evaluation)

    with pytest.raises(KeyboardInterrupt):
        run_shinka_eval(
            program_path="unused.py",
            results_dir=str(results_dir),
            experiment_fn_name="run",
            num_runs=1,
            verbose=False,
        )

    assert not (results_dir / "correct.json").exists()


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
