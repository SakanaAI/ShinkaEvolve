from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass(frozen=True)
class StageConfig:
    key: str
    label: str
    color: str
    start_key: str
    end_key: str
    worker_id_key: str
    capacity_key: str


@dataclass(frozen=True)
class OccupancySeries:
    x: list[pd.Timestamp]
    y: list[float]
    min_start: float
    max_end: float
    capacity: int
    total_duration: float
    duration_by_count: dict[int, float]
    avg_occupied: float
    utilization_pct: float
    full_occupancy_pct: float
    idle_pct: float


@dataclass(frozen=True)
class PoolRuntimeData:
    rows: pd.DataFrame
    capacities: dict[str, int]
    lane_labels: list[str]
    peaks: dict[str, int]


STAGE_CONFIGS: tuple[StageConfig, ...] = (
    StageConfig(
        key="sampling",
        label="Sampling",
        color="#3b82f6",
        start_key="sampling_started_at",
        end_key="sampling_finished_at",
        worker_id_key="sampling_worker_id",
        capacity_key="sampling_worker_capacity",
    ),
    StageConfig(
        key="evaluation",
        label="Evaluation",
        color="#f59e0b",
        start_key="evaluation_started_at",
        end_key="evaluation_finished_at",
        worker_id_key="evaluation_worker_id",
        capacity_key="evaluation_worker_capacity",
    ),
    StageConfig(
        key="postprocess",
        label="Postprocess",
        color="#10b981",
        start_key="postprocess_started_at",
        end_key="postprocess_finished_at",
        worker_id_key="postprocess_worker_id",
        capacity_key="postprocess_worker_capacity",
    ),
)

REQUIRED_RUNTIME_COLUMNS: tuple[str, ...] = (
    "id",
    "timeline_lane_mode",
    "pipeline_started_at",
    "sampling_started_at",
    "sampling_finished_at",
    "evaluation_started_at",
    "evaluation_finished_at",
    "postprocess_started_at",
    "postprocess_finished_at",
)

OPTIONAL_NUMERIC_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "generation",
    "combined_score",
    "sampling_worker_id",
    "evaluation_worker_id",
    "postprocess_worker_id",
    "sampling_worker_capacity",
    "evaluation_worker_capacity",
    "postprocess_worker_capacity",
)


def _coerce_time_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").astype(float) / 1_000_000_000.0
    return pd.to_numeric(series, errors="coerce")


def _coerce_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _get_series_with_default(
    df: pd.DataFrame, column: str, default: object, dtype: Optional[str] = None
) -> pd.Series:
    if column in df.columns:
        series = df[column].copy()
    else:
        series = pd.Series(default, index=df.index)
    if dtype is not None:
        return series.astype(dtype)
    return series


def _get_runtime_row_priority(row: pd.Series) -> int:
    priority = 0
    if not bool(row.get("is_island_copy", False)):
        priority += 10
    if bool(row.get("correct", False)):
        priority += 2
    if pd.notna(row.get("combined_score")):
        priority += 1
    return priority


def _get_tie_break_timestamp(row: pd.Series) -> float:
    value = row.get("timestamp")
    if pd.isna(value):
        return float("inf")
    return float(value)


def _compute_peak_concurrency(
    rows: pd.DataFrame, start_key: str, end_key: str
) -> int:
    events: list[tuple[float, int]] = []
    for start, end in rows[[start_key, end_key]].itertuples(index=False):
        if not pd.notna(start) or not pd.notna(end) or end <= start:
            continue
        events.append((float(start), 1))
        events.append((float(end), -1))

    events.sort(key=lambda item: (item[0], item[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _format_worker_label(stage_label: str, worker_id: float) -> str:
    return f"{stage_label} W{int(worker_id)}"


def _reconcile_stage_worker_overlaps(
    rows: pd.DataFrame, stage: StageConfig
) -> pd.DataFrame:
    reconciled_rows = rows.copy()
    if stage.worker_id_key not in reconciled_rows.columns:
        return reconciled_rows

    valid_rows = reconciled_rows.loc[
        reconciled_rows[stage.worker_id_key].notna()
        & reconciled_rows[stage.start_key].notna()
        & reconciled_rows[stage.end_key].notna()
    ].copy()
    if valid_rows.empty:
        return reconciled_rows

    valid_rows = valid_rows.sort_values(
        by=[
            stage.worker_id_key,
            stage.start_key,
            stage.end_key,
            "generation",
            "pipeline_started_at",
        ],
        kind="stable",
    )

    previous_end_by_worker: dict[float, float] = {}
    for row_index, row in valid_rows.iterrows():
        worker_id = float(row[stage.worker_id_key])
        start_at = float(row[stage.start_key])
        end_at = float(row[stage.end_key])
        previous_end = previous_end_by_worker.get(worker_id)
        if previous_end is None:
            normalized_start = start_at
        else:
            normalized_start = max(start_at, previous_end)
        normalized_end = max(normalized_start, end_at)
        reconciled_rows.at[row_index, stage.start_key] = normalized_start
        reconciled_rows.at[row_index, stage.end_key] = normalized_end
        previous_end_by_worker[worker_id] = normalized_end

    return reconciled_rows


def _prepare_pool_runtime_data(df: pd.DataFrame) -> Optional[PoolRuntimeData]:
    missing_columns = [col for col in REQUIRED_RUNTIME_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required runtime columns: " + ", ".join(sorted(missing_columns))
        )

    runtime_df = df.copy()
    for column in REQUIRED_RUNTIME_COLUMNS[2:]:
        runtime_df[column] = _coerce_time_series(runtime_df[column])
    for column in OPTIONAL_NUMERIC_COLUMNS:
        runtime_df[column] = _coerce_numeric_series(runtime_df, column)

    runtime_df["source_job_id"] = runtime_df.get("source_job_id", runtime_df["id"])
    runtime_df["source_job_id"] = runtime_df["source_job_id"].where(
        runtime_df["source_job_id"].notna(), runtime_df["id"]
    )
    runtime_df["is_island_copy"] = _get_series_with_default(
        runtime_df, "is_island_copy", False
    ).fillna(False).astype(bool)
    runtime_df["correct"] = _get_series_with_default(
        runtime_df, "correct", False
    ).fillna(False).astype(bool)
    runtime_df["patch_name"] = _get_series_with_default(
        runtime_df, "patch_name", "unnamed"
    ).fillna("unnamed")
    runtime_df["model_name"] = _get_series_with_default(
        runtime_df, "model_name", "N/A"
    ).fillna("N/A")

    complete_mask = runtime_df[list(REQUIRED_RUNTIME_COLUMNS[2:])].notna().all(axis=1)
    runtime_df = runtime_df.loc[complete_mask].copy()
    runtime_df = runtime_df.loc[runtime_df["timeline_lane_mode"] == "pool_slots"].copy()
    if runtime_df.empty:
        return None

    deduped_rows: dict[str, pd.Series] = {}
    for _, row in runtime_df.iterrows():
        dedupe_key = str(row["source_job_id"] or row["id"])
        existing = deduped_rows.get(dedupe_key)
        if existing is None:
            deduped_rows[dedupe_key] = row
            continue

        row_priority = _get_runtime_row_priority(row)
        existing_priority = _get_runtime_row_priority(existing)
        if row_priority > existing_priority:
            deduped_rows[dedupe_key] = row
            continue

        if row_priority == existing_priority and _get_tie_break_timestamp(
            row
        ) < _get_tie_break_timestamp(existing):
            deduped_rows[dedupe_key] = row

    deduped_df = pd.DataFrame(deduped_rows.values())
    if deduped_df.empty:
        return None

    deduped_df = deduped_df.sort_values(
        by=["evaluation_started_at", "generation", "pipeline_started_at"],
        kind="stable",
    ).reset_index(drop=True)
    for stage in STAGE_CONFIGS:
        deduped_df = _reconcile_stage_worker_overlaps(deduped_df, stage)

    capacities = {
        stage.key: int(
            max(
                0.0,
                (
                    deduped_df[stage.capacity_key]
                    .fillna(deduped_df[stage.worker_id_key])
                    .fillna(0)
                    .max()
                ),
            )
        )
        for stage in STAGE_CONFIGS
    }
    if all(capacity == 0 for capacity in capacities.values()):
        return None

    lane_labels: list[str] = []
    max_lane_count = max(capacities.values())
    for idx in range(1, max_lane_count + 1):
        for stage in STAGE_CONFIGS:
            if idx <= capacities[stage.key]:
                lane_labels.append(f"{stage.label} W{idx}")

    peaks = {
        stage.key: _compute_peak_concurrency(
            deduped_df, stage.start_key, stage.end_key
        )
        for stage in STAGE_CONFIGS
    }
    return PoolRuntimeData(
        rows=deduped_df, capacities=capacities, lane_labels=lane_labels, peaks=peaks
    )


def _compute_occupancy_series(
    rows: pd.DataFrame, start_key: str, end_key: str, capacity: int
) -> Optional[OccupancySeries]:
    if rows.empty or capacity <= 0:
        return None

    intervals = rows[[start_key, end_key]].copy()
    intervals = intervals.dropna()
    intervals = intervals.loc[intervals[end_key] > intervals[start_key]]
    if intervals.empty:
        return None

    min_start = float(intervals[start_key].min())
    max_end = float(intervals[end_key].max())

    events: list[tuple[float, int]] = []
    for start, end in intervals.itertuples(index=False):
        events.append((float(start), 1))
        events.append((float(end), -1))
    events.sort(key=lambda item: (item[0], item[1]))

    x_seconds = [min_start]
    y_values = [0.0]
    duration_by_count: dict[int, float] = {}
    current = 0
    previous = min_start

    for time_value, delta in events:
        if time_value > previous:
            duration_by_count[current] = duration_by_count.get(current, 0.0) + (
                time_value - previous
            )
        x_seconds.append(time_value)
        y_values.append(float(current))
        current += delta
        x_seconds.append(time_value)
        y_values.append(float(current))
        previous = time_value

    if max_end > previous:
        duration_by_count[current] = duration_by_count.get(current, 0.0) + (
            max_end - previous
        )

    total_duration = max_end - min_start
    occupied_time = sum(count * duration for count, duration in duration_by_count.items())
    x_datetimes = pd.to_datetime(x_seconds, unit="s").to_pydatetime().tolist()

    return OccupancySeries(
        x=x_datetimes,
        y=y_values,
        min_start=min_start,
        max_end=max_end,
        capacity=capacity,
        total_duration=total_duration,
        duration_by_count=duration_by_count,
        avg_occupied=(occupied_time / total_duration) if total_duration > 0 else 0.0,
        utilization_pct=(
            (occupied_time / (capacity * total_duration)) * 100.0
            if total_duration > 0
            else 0.0
        ),
        full_occupancy_pct=(
            (duration_by_count.get(capacity, 0.0) / total_duration) * 100.0
            if total_duration > 0
            else 0.0
        ),
        idle_pct=(
            (duration_by_count.get(0, 0.0) / total_duration) * 100.0
            if total_duration > 0
            else 0.0
        ),
    )


def _configure_time_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _render_empty_plot(ax: Axes, title: str, message: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=28, weight="bold", pad=24)
    ax.set_xlabel("Wall Clock Time", fontsize=18, weight="bold")
    ax.set_ylabel(ylabel, fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="#6b7280",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_generation_runtime_timeline(
    df: pd.DataFrame,
    title: str = "Generation Runtime Timeline",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """
    Plot pool-slot runtime bars for sampling, evaluation, and postprocess stages.

    Args:
        df: Flattened runtime dataframe with pool-slot metadata columns.
        title: Plot title.
        fig: Optional existing figure.
        ax: Optional existing axes.

    Returns:
        Tuple of figure and axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 9))

    prepared = _prepare_pool_runtime_data(df)
    if prepared is None:
        _render_empty_plot(
            ax,
            title=title,
            message="No pool-based runtime timeline data available.",
            ylabel="Resource Pool",
        )
        fig.tight_layout()
        return fig, ax

    lane_to_index = {label: idx for idx, label in enumerate(prepared.lane_labels)}
    min_start = float(prepared.rows["pipeline_started_at"].min())
    max_end = float(prepared.rows["postprocess_finished_at"].max())
    min_visible_seconds = max(0.25, (max_end - min_start) * 0.003)
    bar_height = max(0.12, min(0.75, 3.6 / max(len(prepared.lane_labels), 1)))

    legend_handles: list[Patch] = []
    for stage in STAGE_CONFIGS:
        stage_rows = prepared.rows.loc[prepared.rows[stage.worker_id_key].notna()].copy()
        if stage_rows.empty:
            continue

        stage_rows["lane_label"] = stage_rows[stage.worker_id_key].map(
            lambda worker_id: _format_worker_label(stage.label, worker_id)
        )
        stage_rows = stage_rows.loc[
            stage_rows["lane_label"].isin(prepared.lane_labels)
        ].copy()
        if stage_rows.empty:
            continue

        y_positions = stage_rows["lane_label"].map(lane_to_index).to_numpy()
        left = mdates.date2num(pd.to_datetime(stage_rows[stage.start_key], unit="s"))
        durations = (
            stage_rows[stage.end_key] - stage_rows[stage.start_key]
        ).clip(lower=0.0)
        widths = np.maximum(durations.to_numpy(dtype=float), min_visible_seconds) / (
            24 * 3600
        )
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=bar_height,
            color=stage.color,
            edgecolor="#ffffff",
            linewidth=0.8,
            alpha=0.92,
            label=stage.label,
        )
        legend_handles.append(Patch(facecolor=stage.color, label=stage.label))

    summary = "  |  ".join(
        f"{stage.label}: peak {prepared.peaks[stage.key]}/{prepared.capacities[stage.key]}"
        for stage in STAGE_CONFIGS
    )
    ax.text(
        0.0,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#4b5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
    )
    ax.set_yticks(range(len(prepared.lane_labels)))
    ax.set_yticklabels(prepared.lane_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall Clock Time", fontsize=18, weight="bold")
    ax.set_ylabel("Resource Pool", fontsize=18, weight="bold")
    ax.set_title(title, fontsize=28, weight="bold", pad=24)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _configure_time_axis(ax)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=True,
            columnspacing=1.2,
            handlelength=1.5,
            borderaxespad=0.0,
            fancybox=True,
            framealpha=0.95,
            facecolor="#f8fafc",
            edgecolor="#d1d5db",
        )

    fig.tight_layout()
    return fig, ax


def plot_normalized_occupancy_over_time(
    df: pd.DataFrame,
    title: str = "Normalized Occupancy Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """
    Plot normalized worker occupancy over wall-clock time for each pool stage.

    Args:
        df: Flattened runtime dataframe with pool-slot metadata columns.
        title: Plot title.
        fig: Optional existing figure.
        ax: Optional existing axes.

    Returns:
        Tuple of figure and axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))

    prepared = _prepare_pool_runtime_data(df)
    if prepared is None:
        _render_empty_plot(
            ax,
            title=title,
            message="No normalized occupancy data available.",
            ylabel="Occupancy (%)",
        )
        fig.tight_layout()
        return fig, ax

    min_start = float("inf")
    max_end = float("-inf")
    plotted_any = False
    for stage in STAGE_CONFIGS:
        series = _compute_occupancy_series(
            prepared.rows,
            start_key=stage.start_key,
            end_key=stage.end_key,
            capacity=prepared.capacities[stage.key],
        )
        if series is None:
            continue

        min_start = min(min_start, series.min_start)
        max_end = max(max_end, series.max_end)
        ax.step(
            series.x,
            [(value / series.capacity) * 100.0 for value in series.y],
            where="post",
            linewidth=2.2,
            color=stage.color,
            label=f"{stage.label} Occupancy",
        )
        plotted_any = True

    if not plotted_any:
        _render_empty_plot(
            ax,
            title=title,
            message="No normalized occupancy data available.",
            ylabel="Occupancy (%)",
        )
        fig.tight_layout()
        return fig, ax

    ax.plot(
        pd.to_datetime([min_start, max_end], unit="s"),
        [100, 100],
        linestyle=":",
        linewidth=1.2,
        color="#6b7280",
        alpha=0.8,
        label="100% Capacity",
    )
    ax.set_xlabel("Wall Clock Time", fontsize=18, weight="bold")
    ax.set_ylabel("Occupancy (%)", fontsize=18, weight="bold")
    ax.set_title(title, fontsize=28, weight="bold", pad=24)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _configure_time_axis(ax)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=stage.color,
                linewidth=2.2,
                label=f"{stage.label} Occupancy",
            )
            for stage in STAGE_CONFIGS
        ]
        + [
            Line2D(
                [0],
                [0],
                color="#6b7280",
                linewidth=1.2,
                linestyle=":",
                label="100% Capacity",
            )
        ],
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
        columnspacing=1.2,
        handlelength=1.5,
        borderaxespad=0.0,
        fancybox=True,
        framealpha=0.95,
        facecolor="#f8fafc",
        edgecolor="#d1d5db",
    )

    fig.tight_layout()
    return fig, ax
