import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from shinka.plots import (
    plot_generation_runtime_timeline,
    plot_normalized_occupancy_over_time,
)
from shinka.plots.plot_throughput import (
    _compute_occupancy_series,
    _prepare_pool_runtime_data,
)


def _runtime_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "job-a-main",
                "source_job_id": "job-a",
                "is_island_copy": False,
                "correct": True,
                "combined_score": 0.9,
                "timestamp": 10,
                "generation": 1,
                "patch_name": "patch-a",
                "model_name": "model-a",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 0,
                "sampling_started_at": 0,
                "sampling_finished_at": 2,
                "evaluation_started_at": 2,
                "evaluation_finished_at": 6,
                "postprocess_started_at": 6,
                "postprocess_finished_at": 8,
                "sampling_worker_id": 1,
                "evaluation_worker_id": 1,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 2,
                "evaluation_worker_capacity": 2,
                "postprocess_worker_capacity": 1,
            },
            {
                "id": "job-a-copy",
                "source_job_id": "job-a",
                "is_island_copy": True,
                "correct": False,
                "combined_score": 0.1,
                "timestamp": 11,
                "generation": 1,
                "patch_name": "patch-a-copy",
                "model_name": "model-a",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 0,
                "sampling_started_at": 0,
                "sampling_finished_at": 2,
                "evaluation_started_at": 2,
                "evaluation_finished_at": 6,
                "postprocess_started_at": 6,
                "postprocess_finished_at": 8,
                "sampling_worker_id": 1,
                "evaluation_worker_id": 1,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 2,
                "evaluation_worker_capacity": 2,
                "postprocess_worker_capacity": 1,
            },
            {
                "id": "job-b",
                "source_job_id": "job-b",
                "is_island_copy": False,
                "correct": True,
                "combined_score": 0.8,
                "timestamp": 20,
                "generation": 2,
                "patch_name": "patch-b",
                "model_name": "model-b",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 1,
                "sampling_started_at": 1,
                "sampling_finished_at": 3,
                "evaluation_started_at": 4,
                "evaluation_finished_at": 8,
                "postprocess_started_at": 8,
                "postprocess_finished_at": 10,
                "sampling_worker_id": 2,
                "evaluation_worker_id": 2,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 2,
                "evaluation_worker_capacity": 2,
                "postprocess_worker_capacity": 1,
            },
            {
                "id": "job-missing",
                "source_job_id": "job-missing",
                "is_island_copy": False,
                "correct": True,
                "combined_score": 0.7,
                "timestamp": 30,
                "generation": 3,
                "patch_name": "patch-missing",
                "model_name": "model-c",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 2,
                "sampling_started_at": 2,
                "sampling_finished_at": 4,
                "evaluation_started_at": 5,
                "evaluation_finished_at": 9,
                "postprocess_started_at": 9,
                "postprocess_finished_at": None,
                "sampling_worker_id": 1,
                "evaluation_worker_id": 1,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 2,
                "evaluation_worker_capacity": 2,
                "postprocess_worker_capacity": 1,
            },
        ]
    )


def test_prepare_pool_runtime_data_dedupes_rows_and_computes_capacities():
    prepared = _prepare_pool_runtime_data(_runtime_df())

    assert prepared is not None
    assert prepared.capacities == {"sampling": 2, "evaluation": 2, "postprocess": 1}
    assert prepared.lane_labels == [
        "Sampling W1",
        "Evaluation W1",
        "Postprocess W1",
        "Sampling W2",
        "Evaluation W2",
    ]
    assert list(prepared.rows["id"]) == ["job-a-main", "job-b"]
    assert prepared.peaks == {"sampling": 2, "evaluation": 2, "postprocess": 1}


def test_prepare_pool_runtime_data_handles_missing_optional_columns():
    runtime_df = _runtime_df().drop(
        columns=["source_job_id", "is_island_copy", "patch_name", "model_name"]
    )

    prepared = _prepare_pool_runtime_data(runtime_df)

    assert prepared is not None
    assert list(prepared.rows["id"]) == ["job-a-main", "job-a-copy", "job-b"]
    assert list(prepared.rows["source_job_id"]) == ["job-a-main", "job-a-copy", "job-b"]
    assert list(prepared.rows["patch_name"]) == ["unnamed", "unnamed", "unnamed"]
    assert list(prepared.rows["model_name"]) == ["N/A", "N/A", "N/A"]
    assert prepared.capacities == {"sampling": 2, "evaluation": 2, "postprocess": 1}


def test_compute_occupancy_series_matches_expected_utilization_stats():
    prepared = _prepare_pool_runtime_data(_runtime_df())
    assert prepared is not None

    series = _compute_occupancy_series(
        prepared.rows,
        start_key="evaluation_started_at",
        end_key="evaluation_finished_at",
        capacity=prepared.capacities["evaluation"],
    )

    assert series is not None
    assert series.total_duration == pytest.approx(6.0)
    assert series.avg_occupied == pytest.approx(8.0 / 6.0)
    assert series.utilization_pct == pytest.approx((8.0 / 12.0) * 100.0)
    assert series.full_occupancy_pct == pytest.approx((2.0 / 6.0) * 100.0)
    assert series.idle_pct == pytest.approx(0.0)


def test_prepare_pool_runtime_data_reconciles_legacy_worker_overlap():
    runtime_df = pd.DataFrame(
        [
            {
                "id": "job-a",
                "source_job_id": "job-a",
                "is_island_copy": False,
                "correct": True,
                "combined_score": 0.9,
                "timestamp": 10,
                "generation": 1,
                "patch_name": "patch-a",
                "model_name": "model-a",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 0,
                "sampling_started_at": 0,
                "sampling_finished_at": 2,
                "evaluation_started_at": 2,
                "evaluation_finished_at": 8,
                "postprocess_started_at": 8,
                "postprocess_finished_at": 9,
                "sampling_worker_id": 1,
                "evaluation_worker_id": 1,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 1,
                "evaluation_worker_capacity": 1,
                "postprocess_worker_capacity": 1,
            },
            {
                "id": "job-b",
                "source_job_id": "job-b",
                "is_island_copy": False,
                "correct": True,
                "combined_score": 0.8,
                "timestamp": 20,
                "generation": 2,
                "patch_name": "patch-b",
                "model_name": "model-b",
                "timeline_lane_mode": "pool_slots",
                "pipeline_started_at": 1,
                "sampling_started_at": 1,
                "sampling_finished_at": 5,
                "evaluation_started_at": 5,
                "evaluation_finished_at": 10,
                "postprocess_started_at": 10,
                "postprocess_finished_at": 11,
                "sampling_worker_id": 1,
                "evaluation_worker_id": 1,
                "postprocess_worker_id": 1,
                "sampling_worker_capacity": 1,
                "evaluation_worker_capacity": 1,
                "postprocess_worker_capacity": 1,
            },
        ]
    )

    prepared = _prepare_pool_runtime_data(runtime_df)

    assert prepared is not None
    assert prepared.peaks["evaluation"] == 1
    assert list(prepared.rows["evaluation_started_at"]) == [2.0, 8.0]
    assert list(prepared.rows["evaluation_finished_at"]) == [8.0, 10.0]

    series = _compute_occupancy_series(
        prepared.rows,
        start_key="evaluation_started_at",
        end_key="evaluation_finished_at",
        capacity=prepared.capacities["evaluation"],
    )

    assert series is not None
    assert max(series.y) == 1.0
    assert series.utilization_pct == pytest.approx(100.0)


def test_plot_generation_runtime_timeline_uses_deduped_pool_rows():
    fig, ax = plot_generation_runtime_timeline(_runtime_df(), title="Runtime Timeline")

    assert fig is not None
    assert ax is not None
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "Sampling W1",
        "Evaluation W1",
        "Postprocess W1",
        "Sampling W2",
        "Evaluation W2",
    ]
    assert len(ax.patches) == 6
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {
        "Sampling",
        "Evaluation",
        "Postprocess",
    }
    assert ax.get_legend()._ncols == 3
    assert ax.get_legend()._loc == 9
    assert {text.get_fontsize() for text in ax.get_legend().get_texts()} == {10.0}
    assert ax.get_legend().get_bbox_to_anchor()._bbox.y0 < 0


def test_plot_normalized_occupancy_over_time_adds_reference_line():
    fig, ax = plot_normalized_occupancy_over_time(
        _runtime_df(), title="Normalized Occupancy"
    )

    assert fig is not None
    assert ax is not None
    labels = [line.get_label() for line in ax.lines]
    assert labels == [
        "Sampling Occupancy",
        "Evaluation Occupancy",
        "Postprocess Occupancy",
        "100% Capacity",
    ]
    assert ax.get_ylim()[1] >= 100
    assert list(ax.lines[-1].get_ydata()) == [100, 100]
    assert ax.get_legend()._ncols == 2
    assert ax.get_legend()._loc == 9
    assert {text.get_fontsize() for text in ax.get_legend().get_texts()} == {10.0}
    assert ax.get_legend().get_bbox_to_anchor()._bbox.y0 < 0
