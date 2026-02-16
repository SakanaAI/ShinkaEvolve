import importlib.util
import json
import os
import time
import numpy as np
import pickle
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Callable, Any, Dict, List, Tuple, Optional, Union

from shinka.utils.eval_stop import (
    EarlyStopMethod,
    EarlyStopDecision,
    create_early_stop_method,
)

DEFAULT_METRICS_ON_ERROR = {
    "combined_score": 0.0,
    "execution_time_mean": 0.0,
    "execution_time_std": 0.0,
    "num_successful_runs": 0,
    "num_valid_runs": 0,
    "num_invalid_runs": 0,
    "all_validation_errors": [],
}


def load_program(program_path: str) -> Any:
    """Loads a Python module dynamically from a given file path."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {program_path}")
    if spec.loader is None:
        raise ImportError(f"Spec loader is None for module at {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_single_evaluation(
    program_path: str,
    experiment_fn_name: str,
    run_index: int,
    kwargs: Dict[str, Any],
) -> Tuple[int, Any, float]:
    """
    Execute one evaluation run in an isolated process.
    Returns run index so parent can restore deterministic ordering.
    """
    module = load_program(program_path)
    if not hasattr(module, experiment_fn_name):
        raise AttributeError(
            f"Experiment function '{experiment_fn_name}' not found in {program_path}"
        )
    experiment_fn = getattr(module, experiment_fn_name)

    start_time = time.perf_counter()
    run_result = experiment_fn(**kwargs)
    end_time = time.perf_counter()
    return run_index, run_result, end_time - start_time


def _extract_early_stop_score(
    run_result: Any,
    early_stop_score_fn: Optional[Callable[[Any], float]],
) -> Optional[float]:
    """Extract score for early stopping from run output."""
    if early_stop_score_fn is not None:
        try:
            return early_stop_score_fn(run_result)
        except Exception as e:
            print(f"Warning: early_stop_score_fn failed: {e}")
            return None

    if isinstance(run_result, (int, float)):
        return float(run_result)
    if isinstance(run_result, dict) and "score" in run_result:
        return float(run_result["score"])
    return None


def save_json_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
) -> None:
    """Saves metrics and correctness status to JSON files."""
    os.makedirs(results_dir, exist_ok=True)

    correct_payload = {"correct": correct, "error": error}
    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w") as f:
        json.dump(correct_payload, f, indent=4)
    print(f"Correctness and error status saved to {correct_file}")

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")


def run_shinka_eval(
    program_path: str,
    results_dir: str,
    experiment_fn_name: str,
    num_runs: int,
    get_experiment_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
    aggregate_metrics_fn: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    validate_fn: Optional[Callable[[Any], Tuple[bool, Optional[str]]]] = None,
    plotting_fn: Optional[Callable[[Any], List[Any]]] = None,
    default_metrics_on_error: Optional[Dict[str, Any]] = None,
    # Early stopping parameters
    early_stop_method: Optional[Union[str, EarlyStopMethod]] = None,
    early_stop_threshold: Optional[float] = None,
    early_stop_score_fn: Optional[Callable[[Any], float]] = None,
    early_stop_kwargs: Optional[Dict[str, Any]] = None,
    # Parallel execution parameters
    run_workers: int = 1,
    max_workers_cap: Optional[int] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Runs an experiment multiple times, collects results, optionally validates,
    computes metrics, and saves them. Supports early stopping.

    Args:
        program_path: Path to the Python script/module to evaluate.
        results_dir: Directory to save `metrics.json` and `correct.json`.
        experiment_fn_name: Name of function to call in the loaded module.
        num_runs: Number of times to run the experiment function.
        get_experiment_kwargs: Opt. fn (run_idx_0_based -> kwargs_dict)
                               for experiment args. Seed passed if None.
        aggregate_metrics_fn: Opt. fn (raw_results_list -> metrics_dict)
                              for aggregation. If None, basic run stats
                              (count, time) are recorded.
        validate_fn: Opt. fn (result -> (is_valid, error_msg)) to validate
                       each run. Affects overall correctness.
        plotting_fn: Opt. fn (extra_data) -> List[(Figure|Animation, title)].
                     Returns list of (item, title) tuples. Title used as
                     filename. Figures saved as PNG/PDF, animations as GIF.
        default_metrics_on_error: Metrics for eval failure. Uses predefined
                                  default if None.
        early_stop_method: Early stopping method. Either a string
                          ("none", "bayesian", "ci", "hybrid") or an
                          EarlyStopMethod instance. None disables early stop.
        early_stop_threshold: Target threshold to beat. Required if
                             early_stop_method is set.
        early_stop_score_fn: Function to extract score from run result.
                            If None, assumes result is a numeric score.
        early_stop_kwargs: Additional kwargs for create_early_stop_method
                          (e.g., prob_cutoff, ci_confidence, min_trials).
        run_workers: Number of worker processes for per-run evaluations.
                    `1` keeps sequential behavior. Values > 1 enable
                    process-based parallelism.
        max_workers_cap: Optional upper bound on effective worker count.
                        Applied after `run_workers` and `num_runs`.
                        Useful for externally constraining CPU use.

    Returns:
        A tuple: (metrics, overall_correct_flag, first_error_message)
    """
    effective_default_metrics = (
        default_metrics_on_error.copy()
        if default_metrics_on_error
        else DEFAULT_METRICS_ON_ERROR.copy()
    )
    if run_workers < 1:
        raise ValueError("run_workers must be >= 1")
    if max_workers_cap is not None and max_workers_cap < 1:
        raise ValueError("max_workers_cap must be >= 1 when provided")

    effective_run_workers = run_workers
    if max_workers_cap is not None:
        effective_run_workers = min(effective_run_workers, max_workers_cap)
    if num_runs > 0:
        effective_run_workers = min(effective_run_workers, num_runs)
    else:
        effective_run_workers = 1

    parallel_enabled = num_runs > 1 and effective_run_workers > 1

    overall_correct_flag = True
    first_error_message: Optional[str] = None

    all_validation_errors_list: List[str] = []
    num_valid_runs = 0
    num_invalid_runs = 0

    all_run_results: List[Any] = []
    execution_times: List[float] = []

    # Early stopping setup
    early_stopper: Optional[EarlyStopMethod] = None
    early_stop_scores: List[float] = []
    early_stop_decision: Optional[EarlyStopDecision] = None

    if parallel_enabled and early_stop_method is not None:
        raise ValueError(
            "Early stopping is only supported in sequential mode "
            "(set run_workers=1)."
        )

    if early_stop_method is not None:
        if early_stop_threshold is None:
            raise ValueError(
                "early_stop_threshold is required when early_stop_method is set"
            )
        if isinstance(early_stop_method, str):
            es_kwargs = early_stop_kwargs or {}
            early_stopper = create_early_stop_method(early_stop_method, **es_kwargs)
        else:
            early_stopper = early_stop_method
        early_stopper.reset()
        print(
            f"Early stopping enabled: method={early_stopper.name}, "
            f"threshold={early_stop_threshold}"
        )

    try:
        module = load_program(program_path)
        if not hasattr(module, experiment_fn_name):
            raise AttributeError(
                f"Experiment function '{experiment_fn_name}' not found in "
                f"{program_path}"
            )
        experiment_fn = getattr(module, experiment_fn_name)

        if parallel_enabled:
            print(
                f"Parallel evaluation enabled with {effective_run_workers} worker(s) "
                f"for {num_runs} run(s)"
            )
            ordered_run_results: List[Any] = [None] * num_runs
            ordered_execution_times: List[float] = [0.0] * num_runs
            run_completed: List[bool] = [False] * num_runs
            futures_to_indices: Dict[Future[Tuple[int, Any, float]], int] = {}

            with ProcessPoolExecutor(max_workers=effective_run_workers) as executor:
                for i in range(num_runs):
                    print(
                        f"{10 * '='}Running program evaluation {i + 1}/{num_runs}...{10 * '='}"
                    )
                    kwargs: Dict[str, Any] = (
                        get_experiment_kwargs(i)
                        if get_experiment_kwargs
                        else {"seed": i + 1}
                    )
                    try:
                        future = executor.submit(
                            _run_single_evaluation,
                            program_path,
                            experiment_fn_name,
                            i,
                            kwargs,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to submit run {i + 1}/{num_runs} for "
                            f"parallel execution: {e}. Ensure kwargs are "
                            "pickle-serializable."
                        ) from e
                    futures_to_indices[future] = i

                for future in as_completed(futures_to_indices):
                    submitted_idx = futures_to_indices[future]
                    try:
                        completed_idx, run_result, run_time = future.result()
                    except Exception as e:
                        err_msg = str(e)
                        pickle_hint = ""
                        if "pickle" in err_msg.lower():
                            pickle_hint = (
                                " Ensure experiment kwargs and return values "
                                "are pickle-serializable."
                            )
                        raise RuntimeError(
                            f"Run {submitted_idx + 1}/{num_runs} failed in "
                            f"parallel evaluation: {err_msg}.{pickle_hint}"
                        ) from e

                    ordered_run_results[completed_idx] = run_result
                    ordered_execution_times[completed_idx] = run_time
                    run_completed[completed_idx] = True

            for i in range(num_runs):
                if not run_completed[i]:
                    raise RuntimeError(
                        f"Run {i + 1}/{num_runs} did not complete in parallel mode."
                    )

                run_result = ordered_run_results[i]
                run_time = ordered_execution_times[i]
                all_run_results.append(run_result)
                execution_times.append(run_time)

                if validate_fn:
                    is_valid, validation_err_msg = validate_fn(run_result)
                    if not is_valid:
                        num_invalid_runs += 1
                        overall_correct_flag = False
                        if validation_err_msg:
                            if not first_error_message:
                                first_error_message = (
                                    f"Validation failed: {validation_err_msg}"
                                )
                            if validation_err_msg not in all_validation_errors_list:
                                all_validation_errors_list.append(validation_err_msg)
                    else:
                        num_valid_runs += 1

                print(
                    f"{10 * '='}Run {i + 1}/{num_runs} completed in "
                    f"{run_time:.2f} seconds{10 * '='}"
                )
        else:
            for i in range(num_runs):
                print(
                    f"{10 * '='}Running program evaluation {i + 1}/{num_runs}...{10 * '='}"
                )
                run_kwargs: Dict[str, Any] = {}
                if get_experiment_kwargs:
                    run_kwargs = get_experiment_kwargs(i)
                else:
                    run_kwargs = {"seed": i + 1}

                start_time = time.perf_counter()
                run_result = experiment_fn(**run_kwargs)
                end_time = time.perf_counter()

                all_run_results.append(run_result)
                execution_times.append(end_time - start_time)

                if validate_fn:
                    is_valid, validation_err_msg = validate_fn(run_result)
                    if not is_valid:
                        num_invalid_runs += 1
                        overall_correct_flag = False
                        if validation_err_msg:
                            if not first_error_message:
                                first_error_message = (
                                    f"Validation failed: {validation_err_msg}"
                                )
                            if validation_err_msg not in all_validation_errors_list:
                                all_validation_errors_list.append(validation_err_msg)
                    else:
                        num_valid_runs += 1
                print(
                    f"{10 * '='}Run {i + 1}/{num_runs} completed in {end_time - start_time:.2f} seconds{10 * '='}"
                )

                # Early stopping check
                if early_stopper is not None and early_stop_threshold is not None:
                    score = _extract_early_stop_score(run_result, early_stop_score_fn)
                    if score is not None:
                        early_stop_scores.append(score)
                        early_stop_decision = early_stopper.check(
                            early_stop_scores, early_stop_threshold
                        )
                        print(
                            f"Early stop check: {early_stop_decision.prediction} "
                            f"(confidence={early_stop_decision.confidence:.3f}, "
                            f"reason={early_stop_decision.reason})"
                        )

                        if early_stop_decision.should_stop:
                            print(
                                f"Early stopping triggered after {i + 1}/{num_runs} runs: "
                                f"{early_stop_decision.reason}"
                            )
                            break

        metrics: Dict[str, Any]
        if aggregate_metrics_fn:
            metrics = aggregate_metrics_fn(all_run_results)
        else:
            metrics = {"num_successful_runs": len(all_run_results)}
            if all_run_results:
                metrics["first_run_result_type"] = str(type(all_run_results[0]))
                metrics["raw_results_preview"] = str(all_run_results[:2])
            else:
                metrics["first_run_result_type"] = "N/A"
                metrics["raw_results_preview"] = "N/A"

        metrics["execution_time_mean"] = (
            float(np.mean(execution_times)) if execution_times else 0.0
        )
        metrics["execution_time_std"] = (
            float(np.std(execution_times)) if execution_times else 0.0
        )
        if validate_fn:
            metrics["num_valid_runs"] = num_valid_runs
            metrics["num_invalid_runs"] = num_invalid_runs
            metrics["all_validation_errors"] = all_validation_errors_list

        # Add early stopping metadata
        if early_stopper is not None:
            metrics["early_stop"] = {
                "method": early_stopper.name,
                "threshold": early_stop_threshold,
                "triggered": early_stop_decision.should_stop
                if early_stop_decision
                else False,
                "runs_completed": len(all_run_results),
                "runs_requested": num_runs,
                "prediction": early_stop_decision.prediction
                if early_stop_decision
                else "uncertain",
                "confidence": early_stop_decision.confidence
                if early_stop_decision
                else 0.0,
                "reason": early_stop_decision.reason if early_stop_decision else None,
                "scores": early_stop_scores,
            }

        # Check if combined_score is NaN or inf/-inf and mark as incorrect
        if "combined_score" in metrics:
            combined_score = metrics["combined_score"]
            if np.isnan(combined_score):
                overall_correct_flag = False
                if not first_error_message:
                    first_error_message = "combined_score is NaN"
            elif np.isinf(combined_score):
                overall_correct_flag = False
                if not first_error_message:
                    first_error_message = f"combined_score is inf ({combined_score})"

    except Exception as e:
        print(f"Evaluation error: {e}")
        metrics = {
            k: effective_default_metrics.get(k, v_default)
            for k, v_default in DEFAULT_METRICS_ON_ERROR.items()
        }
        if validate_fn:
            metrics.setdefault("num_valid_runs", 0)
            # Best guess for invalid runs if an exception occurs mid-evaluation
            num_potential_runs = num_runs
            if all_run_results is not None:
                num_potential_runs = len(all_run_results)
            metrics.setdefault("num_invalid_runs", num_potential_runs)
            metrics.setdefault("all_validation_errors", [str(e)])

        first_error_message = str(e)
        overall_correct_flag = False

    if "extra_data" in metrics:
        os.makedirs(results_dir, exist_ok=True)
        extra_data = metrics.pop("extra_data")
        extra_file = os.path.join(results_dir, "extra.pkl")
        with open(extra_file, "wb") as f:
            pickle.dump(extra_data, f)
        print(f"Extra data saved to {extra_file}")

        if plotting_fn is not None:
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            try:
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation, ArtistAnimation

                items = plotting_fn(extra_data)
                fig_count, anim_count = 0, 0
                for item, title in items:
                    if isinstance(item, (FuncAnimation, ArtistAnimation)):
                        gif_path = os.path.join(plots_dir, f"{title}.gif")
                        item.save(gif_path, writer="pillow", fps=30)
                        plt.close(item._fig)
                        anim_count += 1
                    else:
                        # Assume it's a Figure
                        png_path = os.path.join(plots_dir, f"{title}.png")
                        pdf_path = os.path.join(plots_dir, f"{title}.pdf")
                        item.savefig(png_path, bbox_inches="tight", dpi=150)
                        item.savefig(pdf_path, bbox_inches="tight")
                        plt.close(item)
                        fig_count += 1
                print(
                    f"Saved {fig_count} plot(s) and {anim_count} animation(s) "
                    f"to {plots_dir}"
                )
            except Exception as e:
                print(f"Error generating plots: {e}")

    save_json_results(results_dir, metrics, overall_correct_flag, first_error_message)
    return metrics, overall_correct_flag, first_error_message
