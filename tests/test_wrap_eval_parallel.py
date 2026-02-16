import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from shinka.core import run_shinka_eval


def _write_program(tmp_path: Path, source: str) -> str:
    program_path = tmp_path / "program_eval.py"
    program_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(program_path)


def _results_dir(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_run_shinka_eval_parallel_matches_sequential(tmp_path: Path) -> None:
    program_path = _write_program(
        tmp_path,
        """
        import time

        def run_experiment(seed):
            # Deliberately finish out of order to verify ordered aggregation.
            time.sleep(0.01 * (6 - seed))
            return {"seed": seed}
        """,
    )

    def get_kwargs(run_idx: int) -> Dict[str, Any]:
        return {"seed": run_idx + 1}

    def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ordered_seeds = [res["seed"] for res in results]
        return {
            "combined_score": float(sum(ordered_seeds)),
            "ordered_seeds": ordered_seeds,
        }

    def validate_result(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if result["seed"] % 2 == 0:
            return False, "even seed invalid"
        return True, None

    seq_metrics, seq_correct, seq_err = run_shinka_eval(
        program_path=program_path,
        results_dir=_results_dir(tmp_path, "seq"),
        experiment_fn_name="run_experiment",
        num_runs=5,
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_metrics,
        validate_fn=validate_result,
        run_workers=1,
    )
    par_metrics, par_correct, par_err = run_shinka_eval(
        program_path=program_path,
        results_dir=_results_dir(tmp_path, "par"),
        experiment_fn_name="run_experiment",
        num_runs=5,
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_metrics,
        validate_fn=validate_result,
        run_workers=3,
    )

    assert seq_metrics["ordered_seeds"] == [1, 2, 3, 4, 5]
    assert par_metrics["ordered_seeds"] == [1, 2, 3, 4, 5]
    assert par_metrics["combined_score"] == seq_metrics["combined_score"]
    assert par_metrics["num_valid_runs"] == seq_metrics["num_valid_runs"] == 3
    assert par_metrics["num_invalid_runs"] == seq_metrics["num_invalid_runs"] == 2
    assert par_metrics["all_validation_errors"] == seq_metrics[
        "all_validation_errors"
    ] == ["even seed invalid"]
    assert seq_correct is False
    assert par_correct is False
    assert seq_err == "Validation failed: even seed invalid"
    assert par_err == "Validation failed: even seed invalid"


def test_run_shinka_eval_parallel_worker_error_surfaces(tmp_path: Path) -> None:
    program_path = _write_program(
        tmp_path,
        """
        def run_experiment(seed):
            if seed == 3:
                raise RuntimeError("boom seed 3")
            return seed
        """,
    )

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=_results_dir(tmp_path, "worker_error"),
        experiment_fn_name="run_experiment",
        num_runs=4,
        run_workers=2,
    )

    assert correct is False
    assert error_msg is not None
    assert "Run 3/4 failed in parallel evaluation" in error_msg
    assert "boom seed 3" in error_msg
    assert metrics["combined_score"] == 0.0


def test_parallel_mode_rejects_early_stop(tmp_path: Path) -> None:
    program_path = _write_program(
        tmp_path,
        """
        def run_experiment(seed):
            return float(seed)
        """,
    )

    with pytest.raises(
        ValueError, match="Early stopping is only supported in sequential mode"
    ):
        run_shinka_eval(
            program_path=program_path,
            results_dir=_results_dir(tmp_path, "early_stop_guard"),
            experiment_fn_name="run_experiment",
            num_runs=4,
            run_workers=2,
            early_stop_method="ci",
            early_stop_threshold=0.5,
        )


def test_invalid_worker_configuration_raises(tmp_path: Path) -> None:
    program_path = _write_program(
        tmp_path,
        """
        def run_experiment(seed):
            return seed
        """,
    )

    with pytest.raises(ValueError, match="run_workers must be >= 1"):
        run_shinka_eval(
            program_path=program_path,
            results_dir=_results_dir(tmp_path, "bad_workers"),
            experiment_fn_name="run_experiment",
            num_runs=1,
            run_workers=0,
        )

    with pytest.raises(ValueError, match="max_workers_cap must be >= 1"):
        run_shinka_eval(
            program_path=program_path,
            results_dir=_results_dir(tmp_path, "bad_cap"),
            experiment_fn_name="run_experiment",
            num_runs=1,
            run_workers=1,
            max_workers_cap=0,
        )
