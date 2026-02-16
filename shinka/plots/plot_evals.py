import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from shinka.utils import get_path_to_best_node
import matplotlib.transforms as transforms


def plot_evals_performance(
    df: pd.DataFrame,
    title: str = "Best Combined Score Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "# Evaluated Programs",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    plot_path_to_best_node: bool = True,
    scatter_improvements_only: bool = False,
    annotate: bool = True,
    show_cost: bool = True,
):
    """
    Plots the improvement of a program over generations.

    Args:
        df: DataFrame containing program evaluation data
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ylim: Optional tuple of (min, max) for y-axis limits
        plot_path_to_best_node: Whether to plot path to best node
        scatter_improvements_only: If True, only show scatter points where
            the cummax improves. If False, show all evaluation points.
        annotate: If True, annotate points with patch names. If False,
            no annotations are shown.
        show_cost: If True, show cumulative cost on second y-axis. If False,
            hide the cost information.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Plot best score line
    # Calculate cumulative maximum and back-fill leading NaNs
    # to ensure the line is continuous from the start.
    df = df.sort_values(by="generation")
    df_filtered = df[df["correct"]].copy()

    cummax_scores = df_filtered["combined_score"].cummax()

    line1 = ax.plot(
        df_filtered["generation"],
        cummax_scores,
        linewidth=3,
        color="red",
        label="Best Score",
    )

    # Plot individual evaluations as scatter points
    if scatter_improvements_only:
        # Only plot points where cummax changes (improvements)
        improvements = df_filtered[
            cummax_scores != cummax_scores.shift(1).fillna(-float("inf"))
        ]
        scatter1 = ax.scatter(
            improvements["generation"],
            improvements["combined_score"],
            alpha=1.0,
            s=100,
            color="red",
            marker="o",
            zorder=5,
        )
    else:
        # Plot all evaluation points
        scatter1 = ax.scatter(
            df_filtered["generation"],
            df_filtered["combined_score"],
            alpha=1.0,
            s=40,
            color="black",
            label="Individual Evals",
        )

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Get the path to the best node
    if plot_path_to_best_node:
        best_path_df = get_path_to_best_node(df_filtered, score_column="combined_score")
    else:
        best_path_df = pd.DataFrame()
    line_best_path_plot = []  # Initialize to empty list

    if not best_path_df.empty:
        # Plot the path to the best node
        line_best_path_plot = ax.plot(
            best_path_df["generation"],  # Use generation for x-axis
            best_path_df["combined_score"],
            linestyle="-.",
            marker="o",
            color="blue",
            label="Path to Best Node",
            markersize=5,
            linewidth=2,
        )
        # Add annotations if 'patch_name' column exists and annotate is True
        if annotate and "patch_name" in best_path_df.columns:
            _place_non_overlapping_annotations(
                ax, best_path_df, "generation", "combined_score", "patch_name"
            )

    # Create handles and labels for legend
    handles = line1 + [scatter1]
    if line_best_path_plot:  # If the best path was plotted
        handles.extend(line_best_path_plot)

    # Get labels and filter out matplotlib auto-generated ones (starting _)
    labels = [h.get_label() for h in handles if not h.get_label().startswith("_")]
    handles = [h for h in handles if not h.get_label().startswith("_")]

    # Create a second y-axis for cumulative API cost (if show_cost is True)
    if show_cost and "api_costs" in df_filtered.columns:
        ax2 = ax.twinx()
        cumulative_api_cost = df["api_costs"].cumsum().bfill()
        cumulative_embed_cost = (
            df["embed_cost"].cumsum().bfill() if "embed_cost" in df.columns else 0
        )
        cumulative_novelty_cost = (
            df["novelty_cost"].cumsum().bfill() if "novelty_cost" in df.columns else 0
        )

        cumulative_meta_cost = (
            df["meta_cost"].cumsum().bfill() if "meta_cost" in df.columns else 0
        )

        # Sum all costs together
        total_cumulative_cost = (
            cumulative_api_cost
            + cumulative_embed_cost
            + cumulative_novelty_cost
            + cumulative_meta_cost
        )

        line2 = ax2.plot(
            df["generation"],
            total_cumulative_cost,
            linewidth=2,
            color="orange",
            linestyle="--",
            label="Cumulative Cost",
        )
        ax2.set_ylabel(
            "Cumulative API Cost ($)",
            fontsize=22,
            weight="bold",
            color="orange",
            labelpad=15,
        )
        ax2.tick_params(axis="y", which="major", labelsize=25)
        handles.extend(line2)
        labels = [h.get_label() for h in handles]  # Recreate labels

        # Configure ax2 spines
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", which="major", labelsize=30)

    ax.legend(handles, labels, fontsize=25, loc="lower right")

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines for the primary axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()  # Adjust layout to prevent overlapping labels

    return fig, ax


def plot_evals_performance_compare(
    dfs: List[pd.DataFrame],
    labels: List[str],
    title: str = "Best Combined Score Over Time - Comparison",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "# Evaluated Programs",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    scatter_improvements_only: bool = False,
    colors: Optional[List[str]] = None,
):
    """
    Plots comparison of multiple runs over generations.

    Args:
        dfs: List of DataFrames containing program evaluation data
        labels: List of labels for each dataset
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ylim: Optional tuple of (min, max) for y-axis limits
        scatter_improvements_only: If True, only show scatter at improvements
        colors: Optional list of colors for each dataset
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

        # Sort and filter
        df = df.sort_values(by="generation")
        df_filtered = df[df["correct"]].copy()

        if df_filtered.empty:
            continue

        cummax_scores = df_filtered["combined_score"].cummax()

        # Plot best score line
        line = ax.plot(
            df_filtered["generation"],
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
                    improvements["generation"],
                    improvements["combined_score"],
                    alpha=0.6,
                    s=80,
                    color=color,
                    marker="*",
                    zorder=5,
                )
        else:
            ax.scatter(
                df_filtered["generation"],
                df_filtered["combined_score"],
                alpha=0.4,
                s=30,
                color=color,
            )

    if ylim is not None:
        ax.set_ylim(*ylim)

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


def _place_non_overlapping_annotations(
    ax: Axes, df: pd.DataFrame, x_col: str, y_col: str, text_col: str
):
    """
    Places annotations with minimal overlap using a systematic approach.
    """
    # Define multiple offset positions to try (in order of preference)
    offset_positions = [
        (40, -30),  # bottom-right
        (40, 30),  # top-right
        (-40, 30),  # top-left
        (-40, -30),  # bottom-left
        (60, 0),  # right
        (-60, 0),  # left
        (0, 40),  # top
        (0, -40),  # bottom
        (70, -50),  # far bottom-right
        (-70, 50),  # far top-left
    ]

    placed_boxes = []  # Store bounding boxes of placed annotations

    for _, row in df.iterrows():
        patch_name_val = str(row.get(text_col, ""))
        if pd.notna(patch_name_val) and patch_name_val != "":
            if patch_name_val == "nan" or patch_name_val == "none":
                patch_name_val = "Base"

            # Wrap long patch names
            patch_name_to_plot = _wrap_text(patch_name_val, max_length=15)

            x_pos = float(row[x_col])
            y_pos = float(row[y_col])

            # Find the best position with minimal overlap
            best_offset, best_ha, best_va = _find_best_position(
                ax, x_pos, y_pos, patch_name_to_plot, offset_positions, placed_boxes
            )

            # Place the annotation
            annotation = ax.annotate(
                patch_name_to_plot,
                (x_pos, y_pos),
                textcoords="offset points",
                xytext=best_offset,
                ha=best_ha,
                va=best_va,
                fontsize=11,
                fontweight="bold",
                color="darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="lightyellow",
                    ec="black",
                    alpha=0.7,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    shrinkA=5,
                    shrinkB=5,
                    connectionstyle="arc3,rad=0.2",
                    color="black",
                ),
                zorder=10,
            )

            # Store the bounding box for future collision detection
            try:
                # Get the bounding box in data coordinates
                bbox = annotation.get_window_extent()
                inv_transform = ax.transData.inverted()
                bbox_data = inv_transform.transform_bbox(bbox)
                placed_boxes.append(bbox_data)
            except Exception:
                # Fallback: approximate bounding box
                approx_width = len(patch_name_to_plot) * 0.01  # rough estimate
                approx_height = patch_name_to_plot.count("\n") * 0.02 + 0.02
                placed_boxes.append(
                    transforms.Bbox.from_bounds(
                        x_pos - approx_width / 2,
                        y_pos - approx_height / 2,
                        approx_width,
                        approx_height,
                    )
                )


def _wrap_text(text: str, max_length: int = 15) -> str:
    """
    Wraps text at word boundaries for better readability.
    """
    if len(text) <= max_length:
        return text

    # Try to find a good breaking point
    mid_point = len(text) // 2

    # Look for a space near the middle
    for offset in range(min(5, mid_point)):
        # Check before midpoint
        if mid_point - offset > 0 and text[mid_point - offset] == " ":
            break_point = mid_point - offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

        # Check after midpoint
        if mid_point + offset < len(text) and text[mid_point + offset] == " ":
            break_point = mid_point + offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

    # No good space found, break at midpoint
    return f"{text[:mid_point]}\n{text[mid_point:]}"


def _find_best_position(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset_positions: List[Tuple[int, int]],
    placed_boxes: List[transforms.Bbox],
) -> Tuple[Tuple[int, int], str, str]:
    """
    Finds the best annotation position with minimal overlap.
    """
    best_offset = offset_positions[0]
    best_overlap_count = float("inf")

    for offset in offset_positions:
        # Determine alignment based on offset
        ha = "left" if offset[0] >= 0 else "right"
        va = "bottom" if offset[1] >= 0 else "top"

        # Estimate the bounding box for this position
        estimated_bbox = _estimate_annotation_bbox(
            ax, x_pos, y_pos, text, offset, ha, va
        )

        # Count overlaps with existing annotations
        overlap_count = sum(1 for bbox in placed_boxes if estimated_bbox.overlaps(bbox))

        # If no overlaps, use this position
        if overlap_count == 0:
            return offset, ha, va

        # Track the position with minimum overlaps
        if overlap_count < best_overlap_count:
            best_overlap_count = overlap_count
            best_offset = offset

    # Return the alignment for the best offset
    ha = "left" if best_offset[0] >= 0 else "right"
    va = "bottom" if best_offset[1] >= 0 else "top"

    return best_offset, ha, va


def _estimate_annotation_bbox(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset: Tuple[int, int],
    ha: str,
    va: str,
) -> transforms.Bbox:
    """
    Estimates the bounding box of an annotation in data coordinates.
    """
    # Rough estimation based on text length and number of lines
    lines = text.split("\n")
    max_line_length = max(len(line) for line in lines)
    num_lines = len(lines)

    # Approximate dimensions (these are rough estimates)
    char_width_data = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100
    line_height_data = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 50

    width = max_line_length * char_width_data
    height = num_lines * line_height_data

    # Convert offset from points to data coordinates (approximate)
    x_offset_data = offset[0] * char_width_data / 8  # rough conversion
    y_offset_data = offset[1] * line_height_data / 12  # rough conversion

    # Calculate annotation position based on alignment
    if ha == "left":
        left = x_pos + x_offset_data
        right = left + width
    else:  # ha == "right"
        right = x_pos + x_offset_data
        left = right - width

    if va == "bottom":
        bottom = y_pos + y_offset_data
        top = bottom + height
    else:  # va == "top"
        top = y_pos + y_offset_data
        bottom = top - height

    return transforms.Bbox.from_bounds(left, bottom, width, height)
