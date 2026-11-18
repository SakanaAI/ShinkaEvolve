"""Result files are written so that a partial write never reads as correct."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shinka.core import wrap_eval
from shinka.core.evaluation_results import validate_evaluation_results
from shinka.core.wrap_eval import run_shinka_eval, save_json_results
from shinka.utils import load_results


def test_metrics_are_written_before_the_correctness_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    write = wrap_eval._write_text_atomic

    def recording_write(path: str, text: str) -> None:
        written.append(os.path.basename(path))
        write(path, text)

    monkeypatch.setattr(wrap_eval, "_write_text_atomic", recording_write)

    save_json_results(str(tmp_path), {"combined_score": 1.0}, True, verbose=False)

    assert written == ["metrics.json", "correct.json"]


def test_result_files_are_replaced_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader sees the previous metrics or the complete new ones, never a part."""
    (tmp_path / "metrics.json").write_text('{"combined_score": 9.0}')
    replace = os.replace
    replaced: dict[str, tuple[Any, Any]] = {}

    def observing_replace(source: str, target: str) -> None:
        before = Path(target).read_text() if os.path.exists(target) else None
        replaced[os.path.basename(target)] = (before, json.loads(Path(source).read_text()))
        replace(source, target)

    monkeypatch.setattr(wrap_eval.os, "replace", observing_replace)

    save_json_results(str(tmp_path), {"combined_score": 1.0}, True, verbose=False)

    assert replaced == {
        "metrics.json": ('{"combined_score": 9.0}', {"combined_score": 1.0}),
        "correct.json": (None, {"correct": True, "error": None}),
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "correct.json",
        "metrics.json",
    ]


def test_interrupted_save_does_not_read_as_a_correct_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two writes leaves no verdict, old or new."""
    (tmp_path / "correct.json").write_text('{"correct": true}')
    (tmp_path / "metrics.json").write_text('{"combined_score": 9.0}')
    write = wrap_eval._write_text_atomic

    def killed_before_verdict(path: str, text: str) -> None:
        if path.endswith("correct.json"):
            raise KeyboardInterrupt
        write(path, text)

    monkeypatch.setattr(wrap_eval, "_write_text_atomic", killed_before_verdict)

    with pytest.raises(KeyboardInterrupt):
        save_json_results(str(tmp_path), {"combined_score": 1.0}, True, verbose=False)

    assert not (tmp_path / "correct.json").exists()
    results = load_results(str(tmp_path))
    assert results["correct"] == {"correct": False}
    with pytest.raises(ValueError, match="Correctness file not found"):
        validate_evaluation_results(results)


def test_unserializable_metrics_leave_no_result_behind(tmp_path: Path) -> None:
    (tmp_path / "correct.json").write_text('{"correct": true}')

    with pytest.raises(TypeError):
        save_json_results(
            str(tmp_path), {"combined_score": 1.0, "bad": object()}, True, verbose=False
        )

    assert not (tmp_path / "correct.json").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_numpy_metrics_are_saved_as_plain_json(tmp_path: Path) -> None:
    metrics: dict[str, Any] = {
        "combined_score": np.float32(1.5),
        "public": {"count": np.int64(3), "curve": np.array([1.0, 2.0])},
    }

    save_json_results(str(tmp_path), metrics, np.bool_(True), verbose=False)

    assert json.loads((tmp_path / "correct.json").read_text()) == {
        "correct": True,
        "error": None,
    }
    outcome = validate_evaluation_results(load_results(str(tmp_path)))
    assert outcome.correct is True
    assert outcome.combined_score == 1.5
    assert outcome.public_metrics == {"count": 3, "curve": [1.0, 2.0]}


def test_evaluation_with_numpy_aggregates_is_stored_correct(tmp_path: Path) -> None:
    program = tmp_path / "program.py"
    program.write_text("def run_experiment(seed):\n    return seed\n")
    results_dir = tmp_path / "results"

    run_shinka_eval(
        str(program),
        str(results_dir),
        "run_experiment",
        num_runs=2,
        aggregate_metrics_fn=lambda runs: {"combined_score": np.float32(sum(runs))},
        verbose=False,
    )

    outcome = validate_evaluation_results(load_results(str(results_dir)))
    assert outcome.correct is True
    assert outcome.combined_score == 3.0


@pytest.mark.parametrize(
    "content",
    [b'{"correct": tr', b"", b"\xff\xfe"],
    ids=["truncated", "empty", "not-text"],
)
def test_unreadable_correctness_file_is_reported_not_raised(
    tmp_path: Path, content: bytes
) -> None:
    (tmp_path / "metrics.json").write_text('{"combined_score": 1.0}')
    (tmp_path / "correct.json").write_bytes(content)

    results = load_results(str(tmp_path))

    assert results["correct"] == {"correct": False}
    assert "correct.json" in results["evaluation_error"]
    with pytest.raises(ValueError, match="Could not decode JSON"):
        validate_evaluation_results(results)


@pytest.mark.parametrize(
    "content",
    [b'{"combined_score": 1.', b"\xff\xfe"],
    ids=["truncated", "not-text"],
)
def test_unreadable_metrics_file_is_reported_not_raised(
    tmp_path: Path, content: bytes
) -> None:
    (tmp_path / "metrics.json").write_bytes(content)
    (tmp_path / "correct.json").write_text('{"correct": true}')

    results = load_results(str(tmp_path))

    assert results["metrics"] == {}
    assert "metrics.json" in results["evaluation_error"]
    with pytest.raises(ValueError, match="Could not decode JSON"):
        validate_evaluation_results(results)


def test_correctness_file_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text('{"combined_score": 1.0}')
    (tmp_path / "correct.json").write_text("true")

    with pytest.raises(ValueError, match="boolean 'correct' field"):
        validate_evaluation_results(load_results(str(tmp_path)))
