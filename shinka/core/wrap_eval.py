import importlib.util
import json
import os
import time
import numpy as np
import pickle
from typing import Callable, Any, Dict, List, Tuple, Optional

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
    default_metrics_on_error: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Runs an experiment multiple times, collects results, optionally validates,
    computes metrics, and saves them.

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
        default_metrics_on_error: Metrics for eval failure. Uses predefined
                                  default if None.

    Returns:
        A tuple: (metrics, overall_correct_flag, first_error_message)
    """
    effective_default_metrics = (
        default_metrics_on_error.copy()
        if default_metrics_on_error
        else DEFAULT_METRICS_ON_ERROR.copy()
    )

    overall_correct_flag = True
    first_error_message: Optional[str] = None

    all_validation_errors_list: List[str] = []
    num_valid_runs = 0
    num_invalid_runs = 0

    all_run_results: List[Any] = []
    execution_times: List[float] = []

    try:
        module = load_program(program_path)
        if not hasattr(module, experiment_fn_name):
            raise AttributeError(
                f"Experiment function '{experiment_fn_name}' not found in "
                f"{program_path}"
            )
        experiment_fn = getattr(module, experiment_fn_name)

        for i in range(num_runs):
            kwargs: Dict[str, Any] = {}
            if get_experiment_kwargs:
                kwargs = get_experiment_kwargs(i)
            else:
                kwargs = {"seed": i + 1}

            start_time = time.perf_counter()
            run_result = experiment_fn(**kwargs)
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
                f"Run {i + 1}/{num_runs} completed in {end_time - start_time:.2f} seconds"
            )

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

    save_json_results(results_dir, metrics, overall_correct_flag, first_error_message)
    return metrics, overall_correct_flag, first_error_message
