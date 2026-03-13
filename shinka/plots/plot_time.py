import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


def plot_time_performance(
    df: pd.DataFrame,
    title: str = "Best Combined Score Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Elapsed ShinkaEvolve Runtime",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    score_column: str = "combined_score",
    scatter_improvements_only: bool = False,
):
    """
    Plots the cumulative maximum score over elapsed time.

    The x-axis shows elapsed time from the first timestamp in relative units
    (seconds, minutes, hours, or days) depending on the scale.

    Args:
        df: DataFrame containing timestamp and score columns
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ylim: Optional tuple of (min, max) for y-axis limits
        score_column: Name of the score column to plot
        scatter_improvements_only: If True, only show scatter points where
            the cummax improves. If False, show all evaluation points.

    Returns:
        Tuple of (figure, axes)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Sort by timestamp and filter for correct programs
    df = df.sort_values(by="timestamp")
    df_filtered = df[df["correct"]].copy()

    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_filtered["timestamp"]):
        df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])

    # Calculate time difference from start in seconds
    start_time = df_filtered["timestamp"].iloc[0]
    time_diff_seconds = (df_filtered["timestamp"] - start_time).dt.total_seconds()

    # Convert to days, hours, minutes
    time_diff_days = time_diff_seconds / (24 * 3600)

    # Calculate cumulative maximum
    cummax_scores = df_filtered[score_column].cummax()

    # Plot the cummax line
    line1 = ax.plot(
        time_diff_days,
        cummax_scores,
        linewidth=3,
        color="red",
        label="Best Score",
    )

    # Plot individual evaluations as scatter points
    if scatter_improvements_only:
        # Only plot points where cummax changes (improvements)
        improvements_mask = cummax_scores != cummax_scores.shift(1).fillna(
            -float("inf")
        )
        improvements = df_filtered[improvements_mask]
        improvements_time = time_diff_days[improvements_mask]
        scatter1 = ax.scatter(
            improvements_time,
            improvements[score_column],
            alpha=1.0,
            s=100,
            color="red",
            marker="o",
            zorder=5,
        )
    else:
        # Plot all evaluation points
        scatter1 = ax.scatter(
            time_diff_days,
            df_filtered[score_column],
            alpha=1.0,
            s=40,
            color="black",
            label="Individual Evals",
        )

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Format x-axis with custom formatter for days, hours, minutes
    def format_time(x, pos):
        """Format time as days, hours, or minutes depending on scale."""
        total_seconds = x * 24 * 3600  # Convert days back to seconds

        if total_seconds < 60:  # Less than 1 minute
            return f"{total_seconds:.0f}s"
        elif total_seconds < 3600:  # Less than 1 hour
            minutes = total_seconds / 60
            return f"{minutes:.1f}m"
        elif total_seconds < 86400:  # Less than 1 day
            hours = total_seconds / 3600
            return f"{hours:.1f}h"
        else:  # Days
            days = total_seconds / 86400
            return f"{days:.1f}d"

    ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    handles = line1 + [scatter1]
    # Filter out matplotlib auto-generated labels (starting with _)
    labels = [h.get_label() for h in handles if not h.get_label().startswith("_")]
    handles = [h for h in handles if not h.get_label().startswith("_")]
    ax.legend(handles, labels, fontsize=25, loc="lower right")

    fig.tight_layout()

    return fig, ax


def plot_time_performance_compare(
    dfs: List[pd.DataFrame],
    labels: List[str],
    title: str = "Best Combined Score Over Time - Comparison",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Elapsed ShinkaEvolve Runtime",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    score_column: str = "combined_score",
    scatter_improvements_only: bool = False,
    colors: Optional[List[str]] = None,
):
    """
    Plots comparison of multiple runs: best score over elapsed time.

    Args:
        dfs: List of DataFrames with timestamp and score columns
        labels: List of labels for each dataset
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ylim: Optional tuple of (min, max) for y-axis limits
        score_column: Name of the score column to plot
        scatter_improvements_only: If True, only show scatter at improvements
        colors: Optional list of colors for each dataset

    Returns:
        Tuple of (figure, axes)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Default colors if not provided
    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    handles = []
    all_labels = []

    for idx, (df, label) in enumerate(zip(dfs, labels)):
        color = colors[idx % len(colors)]

        # Sort by timestamp
        df = df.sort_values(by="timestamp")
        df_filtered = df[df["correct"]].copy()

        if df_filtered.empty:
            continue

        # Convert timestamp to datetime and calculate elapsed time
        df_filtered["timestamp_dt"] = pd.to_datetime(df_filtered["timestamp"])
        start_time = df_filtered["timestamp_dt"].min()
        df_filtered["elapsed_time"] = (
            df_filtered["timestamp_dt"] - start_time
        ).dt.total_seconds() / (24 * 3600)  # Convert to days

        # Calculate cumulative maximum score
        cummax_scores = df_filtered[score_column].cummax()

        # Plot the cummax line
        line = ax.plot(
            df_filtered["elapsed_time"],
            cummax_scores,
            linewidth=3,
            color=color,
            label=f"{label}",
            alpha=0.8,
        )
        handles.extend(line)
        all_labels.append(f"{label}")

        # Plot scatter points
        if scatter_improvements_only:
            improvements = df_filtered[
                cummax_scores != cummax_scores.shift(1).fillna(-float("inf"))
            ]
            if not improvements.empty:
                ax.scatter(
                    improvements["elapsed_time"],
                    improvements[score_column],
                    alpha=0.6,
                    s=80,
                    color=color,
                    marker="*",
                    zorder=5,
                )
        else:
            ax.scatter(
                df_filtered["elapsed_time"],
                df_filtered[score_column],
                alpha=0.4,
                s=30,
                color=color,
            )

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Format x-axis to show time in appropriate units
    def format_time(x, pos):
        total_seconds = x * 24 * 3600
        if total_seconds < 60:
            return f"{total_seconds:.0f}s"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f}m"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = total_seconds / 86400
            return f"{days:.1f}d"

    ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend(handles, all_labels, fontsize=25, loc="lower right")

    fig.tight_layout()

    return fig, ax


def plot_time_throughput_compare(
    dfs: List[pd.DataFrame],
    labels: List[str],
    title: str = "Number of Evaluated Programs Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Elapsed ShinkaEvolve Runtime",
    ylabel: str = "Number of Evaluated Programs",
    ylim: Optional[Tuple[float, float]] = None,
    colors: Optional[List[str]] = None,
    only_correct: bool = False,
):
    """
    Plots comparison of multiple runs: cumulative count of evaluated programs over elapsed time.

    This function shows throughput (number of evaluations) rather than performance.

    Args:
        dfs: List of DataFrames with timestamp columns
        labels: List of labels for each dataset
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ylim: Optional tuple of (min, max) for y-axis limits
        colors: Optional list of colors for each dataset
        only_correct: If True, only count correct programs. If False, count all evaluations.

    Returns:
        Tuple of (figure, axes)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Default colors if not provided
    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    handles = []
    all_labels = []

    for idx, (df, label) in enumerate(zip(dfs, labels)):
        color = colors[idx % len(colors)]

        # Sort by timestamp
        df = df.sort_values(by="timestamp")

        # Filter for correct programs if requested
        if only_correct:
            df_filtered = df[df["correct"]].copy()
        else:
            df_filtered = df.copy()

        if df_filtered.empty:
            continue

        # Convert timestamp to datetime and calculate elapsed time
        df_filtered["timestamp_dt"] = pd.to_datetime(df_filtered["timestamp"])
        start_time = df_filtered["timestamp_dt"].min()
        df_filtered["elapsed_time"] = (
            df_filtered["timestamp_dt"] - start_time
        ).dt.total_seconds() / (24 * 3600)  # Convert to days

        # Calculate cumulative count of evaluations
        df_filtered["cumulative_count"] = range(1, len(df_filtered) + 1)

        # Plot the cumulative count line
        line = ax.plot(
            df_filtered["elapsed_time"],
            df_filtered["cumulative_count"],
            linewidth=3,
            color=color,
            label=f"{label}",
            alpha=0.8,
        )
        handles.extend(line)
        all_labels.append(f"{label}")

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Format x-axis to show time in appropriate units
    def format_time(x, pos):
        total_seconds = x * 24 * 3600
        if total_seconds < 60:
            return f"{total_seconds:.0f}s"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f}m"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = total_seconds / 86400
            return f"{days:.1f}d"

    ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend(handles, all_labels, fontsize=25, loc="lower right")

    fig.tight_layout()

    return fig, ax
