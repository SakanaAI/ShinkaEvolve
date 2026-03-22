from __future__ import annotations

from typing import Any, Dict, Optional


def _duration_seconds(start_at: float, end_at: float) -> float:
    """Return a non-negative wall-clock duration in seconds."""
    return max(0.0, float(end_at) - float(start_at))


def with_pipeline_timing(
    metadata: Optional[Dict[str, Any]],
    *,
    pipeline_started_at: float,
    sampling_started_at: float,
    sampling_finished_at: float,
    evaluation_started_at: float,
    evaluation_finished_at: float,
    postprocess_started_at: float,
    postprocess_finished_at: float,
) -> Dict[str, Any]:
    """Merge pipeline stage boundaries and durations into program metadata."""
    merged = dict(metadata or {})

    merged.update(
        {
            "pipeline_started_at": pipeline_started_at,
            "sampling_started_at": sampling_started_at,
            "sampling_finished_at": sampling_finished_at,
            "evaluation_started_at": evaluation_started_at,
            "evaluation_finished_at": evaluation_finished_at,
            "postprocess_started_at": postprocess_started_at,
            "postprocess_finished_at": postprocess_finished_at,
        }
    )

    merged["sampling_seconds"] = _duration_seconds(
        sampling_started_at, sampling_finished_at
    )
    merged["evaluation_seconds"] = _duration_seconds(
        evaluation_started_at, evaluation_finished_at
    )
    merged["postprocess_seconds"] = _duration_seconds(
        postprocess_started_at, postprocess_finished_at
    )
    merged["pipeline_seconds"] = _duration_seconds(
        pipeline_started_at, postprocess_finished_at
    )
    # Preserve legacy semantics used across the existing UI and examples.
    merged["compute_time"] = merged["evaluation_seconds"]
    return merged
