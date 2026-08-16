"""Safe metric payloads for the optional W&B integration."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from shinka.database import Program
from shinka.telemetry_safety import is_sensitive_telemetry_key

GENERATION_METRIC = "generation"
INDIVIDUAL_SCORE_METRIC = "score/individual"
EVALUATED_COUNT_METRIC = "population/evaluated_count"
PROGRAM_TABLE_KEY = "population/final_table"
MAX_PUBLIC_METRICS = 64
MAX_METRIC_KEY_LENGTH = 128
PROGRAM_TABLE_COLUMNS = [
    "id",
    "generation",
    "score",
    "correct",
    "parent_id",
    "island_idx",
    "is_copy",
    "in_archive",
    "patch_type",
    "model_name",
    "cost",
]

_COST_KEYS = {
    "api": "api_costs",
    "embed": "embed_cost",
    "novelty": "novelty_cost",
    "meta": "meta_cost",
}
_HEADLESS_MODEL_PREFIX = "headless/"
_TIMING_KEYS = (
    "sampling_seconds",
    "evaluation_seconds",
    "postprocess_seconds",
    "pipeline_seconds",
)


def build_program_log_payload(program: Program) -> Dict[str, Any]:
    """Build one raw W&B history event for an evaluated individual.

    Individual events use W&B's internal history step. They are candidate
    scatter data, not the run's canonical progress series.
    """
    metadata = program.metadata or {}
    headless_used = _is_headless_program(program)
    logged_costs = _reported_program_costs(program)
    individual_score = _finite_float(program.combined_score)
    payload: Dict[str, Any] = {
        GENERATION_METRIC: program.generation,
        INDIVIDUAL_SCORE_METRIC: individual_score,
        "individual/id": program.id,
        "individual/parent_id": program.parent_id,
        "individual/island_idx": program.island_idx,
        "individual/is_copy": is_island_copy(program),
        "individual/correct": bool(program.correct),
        "individual/in_archive": bool(program.in_archive),
        "individual/patch_type": metadata.get("patch_type"),
        "individual/model_name": _program_model_name(program),
        **{f"individual/cost/{name}": value for name, value in logged_costs.items()},
    }
    if headless_used:
        payload.update(
            {
                "headless/usage_status": metadata.get("headless_usage_status"),
                "headless/usage_unknown": metadata.get("headless_usage_unknown"),
                "headless/pricing_status": metadata.get("headless_pricing_status"),
                "headless/pricing_unknown": metadata.get("headless_pricing_unknown"),
                "headless/cost_basis": metadata.get("headless_cost_basis"),
                "headless/pricing_source": metadata.get("headless_pricing_source"),
            }
        )

    for key in _TIMING_KEYS:
        value = _finite_float(metadata.get(key))
        if value is not None:
            payload[f"individual/timing/{key}"] = value

    for key, value in flatten_numeric_metrics(
        program.public_metrics,
        "public_metrics",
        max_metrics=MAX_PUBLIC_METRICS,
        max_key_length=MAX_METRIC_KEY_LENGTH,
    ).items():
        leaf_name = key.rsplit("/", 1)[-1]
        if leaf_name in {"score", "combined_score"} and value == individual_score:
            continue
        payload[key] = value
    return {key: value for key, value in payload.items() if value is not None}


def is_island_copy(program: Program) -> bool:
    """Return whether a DB row is an administrative copy."""
    metadata = program.metadata or {}
    return bool(metadata.get("_is_island_copy") or metadata.get("_spawned_island"))


def flatten_numeric_metrics(
    data: Any,
    prefix: str,
    *,
    max_depth: int = 3,
    max_metrics: int = MAX_PUBLIC_METRICS,
    max_key_length: int = MAX_METRIC_KEY_LENGTH,
) -> Dict[str, float]:
    """Flatten only numeric evaluation metrics, excluding bulky text and arrays."""
    flattened: Dict[str, float] = {}

    def visit(value: Any, parts: List[str], depth: int) -> None:
        if depth > max_depth or len(flattened) >= max_metrics:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if len(flattened) >= max_metrics:
                    break
                if len(str(key)) > max_key_length or _is_sensitive_metric_key(key):
                    continue
                visit(child, [*parts, _metric_segment(key)], depth + 1)
            return
        numeric_value = _finite_float(value)
        if numeric_value is not None:
            metric_key = "/".join([prefix, *parts])
            if len(metric_key) <= max_key_length:
                flattened[metric_key] = numeric_value

    if isinstance(data, dict):
        for key, value in data.items():
            if len(flattened) >= max_metrics:
                break
            if len(str(key)) > max_key_length or _is_sensitive_metric_key(key):
                continue
            visit(value, [_metric_segment(key)], 1)
    return flattened


def program_costs(program: Program) -> Dict[str, float]:
    """Return the non-duplicated cost breakdown for one individual."""
    metadata = program.metadata or {}
    return {
        name: _finite_float(metadata.get(metadata_key)) or 0.0
        for name, metadata_key in _COST_KEYS.items()
    }


def program_table_row(program: Program) -> List[Any]:
    """Return a compact row; detailed program data remains in the WebUI database."""
    metadata = program.metadata or {}
    return [
        program.id,
        program.generation,
        _finite_float(program.combined_score),
        bool(program.correct),
        program.parent_id,
        program.island_idx,
        is_island_copy(program),
        bool(program.in_archive),
        metadata.get("patch_type"),
        _program_model_name(program),
        0.0
        if is_island_copy(program)
        else sum(_reported_program_costs(program).values()),
    ]


def build_population_progress_payload(
    programs: Sequence[Program],
    *,
    evaluated_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build cumulative population and island metrics for one snapshot.

    Administrative copies remain visible in population counts, while the
    evaluation axis, costs, and timings describe unique evaluated programs.
    """
    all_programs = list(programs)
    evaluated_programs = [
        program for program in all_programs if not is_island_copy(program)
    ]
    scores = _scores(all_programs)
    correct_programs = [program for program in all_programs if program.correct]
    correct_scores = _scores(correct_programs)
    costs = {
        name: sum(
            program_costs(program)[name]
            for program in evaluated_programs
            if not _has_unknown_headless_api_cost(program, name)
        )
        for name in _COST_KEYS
    }
    unknown_pricing_count = sum(
        1
        for program in evaluated_programs
        if _has_unknown_headless_api_cost(program, "api")
    )
    payload: Dict[str, Any] = {
        EVALUATED_COUNT_METRIC: (
            len(evaluated_programs)
            if evaluated_count is None
            else max(0, int(evaluated_count))
        ),
        "population/count": len(all_programs),
        "population/correct_count": len(correct_programs),
        "population/correct_rate": (
            len(correct_programs) / len(all_programs) if all_programs else 0.0
        ),
        "population/best_score": max(scores) if scores else None,
        "population/best_correct_score": (
            max(correct_scores) if correct_scores else None
        ),
        "population/mean_score": sum(scores) / len(scores) if scores else None,
        "cost/api": costs["api"],
        "cost/embed": costs["embed"],
        "cost/novelty": costs["novelty"],
        "cost/meta": costs["meta"],
        "cost/total": sum(costs.values()),
        "cost/pricing_unknown_count": unknown_pricing_count,
    }

    for key in _TIMING_KEYS:
        payload[f"timing/{key}_total"] = sum(
            _finite_float((program.metadata or {}).get(key)) or 0.0
            for program in evaluated_programs
        )

    islands: Dict[str, List[Program]] = {}
    for program in all_programs:
        island_key = (
            str(program.island_idx) if program.island_idx is not None else "unknown"
        )
        islands.setdefault(island_key, []).append(program)
    for island_key, island_programs in islands.items():
        payload.update(_island_payload(island_key, island_programs))

    return {key: value for key, value in payload.items() if value is not None}


def build_population_progress_payload_from_telemetry(
    telemetry: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build population metrics from compact database aggregate rows."""
    rows = list(telemetry)
    count = sum(int(row["count"]) for row in rows)
    evaluated_count = sum(int(row["evaluated_count"]) for row in rows)
    correct_count = sum(int(row["correct_count"]) for row in rows)
    score_count = sum(int(row["score_count"]) for row in rows)
    score_sum = sum(float(row["score_sum"]) for row in rows)
    costs = {
        name: sum(float(row[f"{name}_cost"]) for row in rows) for name in _COST_KEYS
    }

    payload: Dict[str, Any] = {
        EVALUATED_COUNT_METRIC: evaluated_count,
        "population/count": count,
        "population/correct_count": correct_count,
        "population/correct_rate": correct_count / count if count else 0.0,
        "population/best_score": _maximum_aggregate(rows, "best_score"),
        "population/best_correct_score": _maximum_aggregate(rows, "best_correct_score"),
        "population/mean_score": score_sum / score_count if score_count else None,
        **{f"cost/{name}": value for name, value in costs.items()},
        "cost/total": sum(costs.values()),
        "cost/pricing_unknown_count": sum(
            int(row["pricing_unknown_count"]) for row in rows
        ),
    }
    for key in _TIMING_KEYS:
        payload[f"timing/{key}_total"] = sum(float(row[key]) for row in rows)
    for row in rows:
        island_key = (
            str(row["island_idx"]) if row["island_idx"] is not None else "unknown"
        )
        island_count = int(row["count"])
        island_score_count = int(row["score_count"])
        prefix = f"island/{_metric_segment(island_key)}"
        payload.update(
            {
                f"{prefix}/count": island_count,
                f"{prefix}/evaluated_count": int(row["evaluated_count"]),
                f"{prefix}/correct_count": int(row["correct_count"]),
                f"{prefix}/correct_rate": (
                    int(row["correct_count"]) / island_count if island_count else 0.0
                ),
                f"{prefix}/best_score": _finite_float(row["best_score"]),
                f"{prefix}/mean_score": (
                    float(row["score_sum"]) / island_score_count
                    if island_score_count
                    else None
                ),
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _maximum_aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [value for row in rows if (value := _finite_float(row[key])) is not None]
    return max(values) if values else None


def build_run_summary(
    programs: Sequence[Program],
    *,
    total_proposals_generated: Optional[int] = None,
    total_api_cost: Optional[float] = None,
    total_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Build concise final values for the W&B run summary."""
    population = build_population_progress_payload(programs)
    correct_scores = _scores([program for program in programs if program.correct])
    summary = {
        "run/program_count": len(programs),
        "run/evaluated_count": population[EVALUATED_COUNT_METRIC],
        "run/correct_rate": population["population/correct_rate"],
        "run/max_generation": max(
            (program.generation for program in programs), default=0
        ),
        "run/best_score": max(correct_scores) if correct_scores else None,
        "run/total_proposals_generated": total_proposals_generated,
        "run/total_api_cost": _finite_float(total_api_cost),
        "run/total_cost": _finite_float(
            total_cost if total_cost is not None else total_api_cost
        ),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _island_payload(island_key: str, programs: Sequence[Program]) -> Dict[str, Any]:
    scores = _scores(programs)
    correct_count = sum(1 for program in programs if program.correct)
    prefix = f"island/{_metric_segment(island_key)}"
    return {
        f"{prefix}/count": len(programs),
        f"{prefix}/evaluated_count": sum(
            1 for program in programs if not is_island_copy(program)
        ),
        f"{prefix}/correct_count": correct_count,
        f"{prefix}/correct_rate": correct_count / len(programs) if programs else 0.0,
        f"{prefix}/best_score": max(scores) if scores else None,
        f"{prefix}/mean_score": sum(scores) / len(scores) if scores else None,
    }


def _scores(programs: Sequence[Program]) -> List[float]:
    return [
        score
        for program in programs
        if (score := _finite_float(program.combined_score)) is not None
    ]


def _has_unknown_headless_api_cost(program: Program, cost_name: str) -> bool:
    return (
        cost_name == "api"
        and _is_headless_program(program)
        and (program.metadata or {}).get("headless_pricing_unknown") is True
    )


def _reported_program_costs(program: Program) -> Dict[str, float]:
    return {
        name: value
        for name, value in program_costs(program).items()
        if not _has_unknown_headless_api_cost(program, name)
    }


def _program_model_name(program: Program) -> Optional[str]:
    metadata = program.metadata or {}
    llm_result = metadata.get("llm_result")
    nested_model = None
    if isinstance(llm_result, dict):
        nested_model = llm_result.get("model_name") or llm_result.get("model")
    value = metadata.get("model_name") or metadata.get("model") or nested_model
    return str(value) if value is not None else None


def _is_headless_program(program: Program) -> bool:
    model_name = _program_model_name(program)
    return model_name is not None and model_name.startswith(_HEADLESS_MODEL_PREFIX)


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            return None
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _metric_segment(value: Any) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return segment.strip("_") or "value"


def _is_sensitive_metric_key(value: Any) -> bool:
    return is_sensitive_telemetry_key(value)
