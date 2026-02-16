import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from .plot_evals import _wrap_text


# Helper function to identify Pareto-optimal points
# Assumes higher values are better for all metrics
def get_pareto_mask(points: np.ndarray) -> np.ndarray:
    """
    Finds the Pareto-efficient points. Assumes all objectives are
    to be maximized. A point is Pareto efficient if it is not
    dominated by any other point. Point A dominates point B if A is
    at least as good as B on all objectives, and strictly better
    than B on at least one objective.

    :param points: An (n_points, n_objectives) NumPy array.
    :return: A (n_points, ) boolean NumPy array indicating Pareto optimality.
    """
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        if not is_pareto[i]:  # If already marked as dominated, skip
            continue
        for j in range(num_points):
            if i == j:
                continue

            # Check if point j dominates point i
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False  # Point i is dominated by point j
                break
    return is_pareto


def _place_pareto_annotations_with_connections(
    ax, pareto_df, x_col, y_col, x_maximize=True
):
    """
    Place patch name annotations for Pareto points using adjustText for
    optimal positioning. Deduplicates based on coordinates and patch_name
    to avoid multiple annotations for the same point.
    """
    # Force axis limits to be updated after any inversion
    ax.figure.canvas.draw_idle()

    annotations = []

    # Deduplicate based on coordinates and patch_name to avoid multiple
    # annotations for the same program point (e.g., island copies)
    unique_points = {}
    for _, row in pareto_df.iterrows():
        patch_name_val = str(row.get("patch_name", ""))
        if (
            pd.notna(patch_name_val)
            and patch_name_val != ""
            and patch_name_val not in ["nan", "none"]
        ):
            x_pos = float(row[x_col])
            y_pos = float(row[y_col])

            # Create a key based on coordinates and patch name
            key = (x_pos, y_pos, patch_name_val)

            # Only keep the first occurrence of each unique point
            if key not in unique_points:
                unique_points[key] = row

    # Now create annotations for unique points only
    for (x_pos, y_pos, patch_name_val), row in unique_points.items():
        # Wrap long patch names
        patch_name_to_plot = _wrap_text(patch_name_val, max_length=12)

        # Calculate initial offset position based on x_maximize
        # Use smaller, more conservative offsets to stay within bounds
        x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

        # Get axis limits for constraint checking
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if x_maximize:
            # For maximization, place labels to the left (worse direction)
            x_offset = -x_range * 0.15  # Reduced from 0.3 to 0.15
        else:
            # For minimization, place labels to the right (worse direction)
            x_offset = x_range * 0.15  # Reduced from 0.3 to 0.15

        # Calculate proposed text position
        text_x = x_pos + x_offset
        text_y = y_pos

        # Ensure text position stays within bounds with margin
        margin_x = x_range * 0.05  # 5% margin
        margin_y = y_range * 0.05  # 5% margin

        text_x = max(x_min + margin_x, min(x_max - margin_x, text_x))
        text_y = max(y_min + margin_y, min(y_max - margin_y, text_y))

        # Create annotation with constrained position
        annotation = ax.annotate(
            patch_name_to_plot,
            xy=(x_pos, y_pos),  # Point to connect arrow to
            xytext=(text_x, text_y),  # Constrained text position
            fontsize=20,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="lightyellow",
                ec="black",
                alpha=0.7,
            ),
            zorder=4.0,
            arrowprops=dict(
                arrowstyle="->",
                shrinkA=5,
                shrinkB=5,
                # connectionstyle="arc3,rad=0.4",
                color="black",
                linewidth=3,  # Make arrow thick
            ),
        )
        annotations.append(annotation)

    # Simple grid-based positioning to avoid overlaps completely
    if annotations:
        # Set clipping on annotations to ensure they don't extend beyond axes
        for annotation in annotations:
            annotation.set_clip_on(True)

        # Sort annotations by x-coordinate of their data points for consistent ordering
        annotations_with_points = []
        for annotation in annotations:
            # Get the xy position (data point) from the annotation
            xy_pos = annotation.xy
            annotations_with_points.append((xy_pos[0], annotation))

        # Sort by x-coordinate
        annotations_with_points.sort(key=lambda x: x[0])

        # Position annotations close to points but with smart vertical spacing
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Create vertical slots to prevent overlaps
        n_annotations = len(annotations_with_points)

        # Calculate vertical spacing for labels
        label_zone_height = y_range * 0.6  # Use 60% of plot height for labels
        label_zone_bottom = y_min + y_range * 0.15  # Start 15% from bottom

        if n_annotations > 1:
            y_spacing = label_zone_height / (n_annotations - 1)
        else:
            y_spacing = 0

        # Position each annotation
        for i, (data_x, annotation) in enumerate(annotations_with_points):
            # Get the original data point position
            data_point_x, data_point_y = annotation.xy

            # Calculate y position in the vertical slot system
            if n_annotations == 1:
                label_y = label_zone_bottom + label_zone_height / 2
            else:
                label_y = label_zone_bottom + i * y_spacing

            # Position horizontally close to the data point but with some offset
            if x_maximize:
                # For maximization, place labels to the left
                label_x = data_point_x - x_range * 0.03
            else:
                # For minimization, place labels to the right
                label_x = data_point_x + x_range * 0.03

            # Ensure labels stay within bounds
            margin_x = x_range * 0.05
            margin_y = y_range * 0.05

            label_x = data_point_x - min(x_range * 0.03, margin_x)
            # + max(x_min + margin_x, min(x_max - margin_x, label_x))
            label_y = max(y_min + margin_y, min(y_max - margin_y, label_y))

            # Set the new position
            annotation.set_position((label_x, label_y))


def plot_pareto_curve(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_maximize: bool = True,
    y_maximize: bool = True,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    title: str = "Pareto Front Analysis",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    annotate: bool = True,
    scatter_improvements_only: bool = False,
):
    """
    Plots a 2D Pareto front with lineage connections, aiming for
    clarity and aesthetics. Axes are inverted as needed so that better
    is always higher and to the right.

    Args:
        df: DataFrame containing program data with metrics
        x_variable: Name of the column for x-axis metric
        y_variable: Name of the column for y-axis metric
        x_maximize: If True, higher x values are better
        y_maximize: If True, higher y values are better
        x_lim: Optional tuple of (min, max) for x-axis limits
        y_lim: Optional tuple of (min, max) for y-axis limits
        title: Plot title
        xlabel: Label for x-axis (defaults to x_variable if not provided)
        ylabel: Label for y-axis (defaults to y_variable if not provided)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        annotate: If True, annotate Pareto points with patch names.
            If False, no annotations are shown.
        scatter_improvements_only: If True, only plot Pareto-optimal points
            and the Pareto front line. If False, plot all points.
    """
    x_metric_col_name, y_metric_col_name = x_variable, y_variable

    # Set labels
    if xlabel is None:
        xlabel = x_variable.replace("_", " ").title()
    if ylabel is None:
        ylabel = y_variable.replace("_", " ").title()

    # Determine axis labels
    final_xlabel = xlabel if xlabel is not None else x_metric_col_name
    final_ylabel = ylabel if ylabel is not None else y_metric_col_name

    required_plotting_cols = [x_metric_col_name, y_metric_col_name]
    missing_metrics = [col for col in required_plotting_cols if col not in df.columns]
    if missing_metrics:
        raise ValueError(
            f"DataFrame missing required metric columns: {missing_metrics}"
        )

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    df_plot = df.copy()

    if "correct" in df_plot.columns:
        try:
            df_plot["correct"] = df_plot["correct"].astype(bool)
        except Exception as e:
            print(
                f"Warning: Could not convert 'correct' column to boolean: "
                f"{e}. Using as is."
            )

        original_row_count = len(df_plot)
        df_plot = df_plot[df_plot["correct"]]
        if len(df_plot) < original_row_count:
            print(
                f"Filtered to {len(df_plot)} 'correct' rows from "
                f"{original_row_count} total."
            )
        if df_plot.empty:
            print("No 'correct' points found to plot.")
            ax.set_title(title, fontsize=32, fontweight="bold", pad=15)
            ax.set_xlabel(final_xlabel, fontsize=25, fontweight="bold", labelpad=15)
            ax.set_ylabel(final_ylabel, fontsize=25, fontweight="bold", labelpad=15)
            ax.grid(
                True, linestyle=":", alpha=0.9, color="lightgray"
            )  # User preference for grid alpha
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if fig:
                fig.tight_layout()
            return fig, ax
    else:
        print("Warning: 'correct' column not found. Plotting all points.")

    for col in [x_variable, y_variable]:
        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    df_plot = df_plot.dropna(subset=[x_variable, y_variable])
    ax.tick_params(axis="both", which="major", labelsize=20)
    if df_plot.empty:
        print("No data to plot after processing metric columns.")
        ax.set_title(title, fontsize=32, fontweight="bold", pad=15)
        ax.set_xlabel(final_xlabel, fontsize=25, fontweight="bold", labelpad=15)
        ax.set_ylabel(final_ylabel, fontsize=25, fontweight="bold", labelpad=15)
        ax.grid(
            True, linestyle=":", alpha=0.9, color="lightgray"
        )  # User preference for grid alpha
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if fig:
            fig.tight_layout()
        return fig, ax

    # Prepare metric values for Pareto calculation
    # Invert values for minimization objectives so higher is always better
    metric_values = df_plot[[x_variable, y_variable]].values.copy()
    if not x_maximize:
        metric_values[:, 0] = -metric_values[:, 0]
    if not y_maximize:
        metric_values[:, 1] = -metric_values[:, 1]

    pareto_mask = get_pareto_mask(metric_values)
    df_plot["is_pareto"] = pareto_mask

    pareto_df = df_plot[df_plot["is_pareto"]].copy()
    non_pareto_df = df_plot[~df_plot["is_pareto"]].copy()

    # Plot non-Pareto points (unless scatter_improvements_only is True)
    if not scatter_improvements_only and not non_pareto_df.empty:
        ax.scatter(
            non_pareto_df[x_metric_col_name],
            non_pareto_df[y_metric_col_name],
            color="dimgray",
            s=100,
            alpha=1.0,
            zorder=1,
            label="Dominated/Other",
        )

    # Plot Pareto points on top
    if not pareto_df.empty:
        # Use different styling when scatter_improvements_only is True
        if scatter_improvements_only:
            marker_style = "*"
            marker_size = 250
            marker_color = "red"
        else:
            marker_style = "o"
            marker_size = 200
            marker_color = "orangered"

        ax.scatter(
            pareto_df[x_metric_col_name],
            pareto_df[y_metric_col_name],
            color=marker_color,
            s=marker_size,
            alpha=1.0,
            marker=marker_style,
            edgecolor="black",
            linewidth=1,
            zorder=3,
            # label="Pareto Optimal",
        )
    # Draw connections for Pareto frontier
    if not pareto_df.empty and len(pareto_df) > 1:
        # Sort Pareto points by x-coordinate to form proper frontier
        pareto_sorted = pareto_df.sort_values(x_metric_col_name)

        # Connect consecutive points along the sorted frontier
        x_coords = pareto_sorted[x_metric_col_name].values
        y_coords = pareto_sorted[y_metric_col_name].values

        ax.plot(
            x_coords,
            y_coords,
            color="red",
            linewidth=4,
            alpha=0.7,
            zorder=2,
        )

    # Invert axes BEFORE annotations if needed so better is always higher
    # and to the right
    if not x_maximize:
        ax.invert_xaxis()
    if not y_maximize:
        ax.invert_yaxis()

    # Annotate Pareto points with patch names using optimization
    if annotate and not pareto_df.empty and "patch_name" in pareto_df.columns:
        _place_pareto_annotations_with_connections(
            ax, pareto_df, x_metric_col_name, y_metric_col_name, x_maximize
        )

    ax.set_xlabel(final_xlabel, fontsize=30, fontweight="bold", labelpad=15)
    ax.set_ylabel(final_ylabel, fontsize=30, fontweight="bold", labelpad=15)
    ax.set_title(title, fontsize=40, fontweight="bold", pad=15)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=25)

    ax.grid(
        True, linestyle=":", alpha=0.9, color="lightgray"
    )  # User preference for grid alpha
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig:
        fig.tight_layout()
    return fig, ax


def plot_pareto_compare(
    dfs: List[pd.DataFrame],
    labels: List[str],
    x_variable: str,
    y_variable: str,
    x_maximize: bool = True,
    y_maximize: bool = True,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    title: str = "Pareto Front Analysis",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colors: Optional[List[str]] = None,
):
    """
    Plots comparison of Pareto fronts from multiple runs.

    Args:
        dfs: List of DataFrames with columns for x_variable and y_variable
        labels: List of labels for each dataset
        x_variable: Column name for x-axis
        y_variable: Column name for y-axis
        x_maximize: If True, higher x values are better
        y_maximize: If True, higher y values are better
        x_lim: Optional tuple for x-axis limits
        y_lim: Optional tuple for y-axis limits
        title: Plot title
        xlabel: Label for x-axis (defaults to x_variable name)
        ylabel: Label for y-axis (defaults to y_variable name)
        fig: Optional existing figure
        ax: Optional existing axes
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

    # Set labels
    if xlabel is None:
        xlabel = x_variable.replace("_", " ").title()
    if ylabel is None:
        ylabel = y_variable.replace("_", " ").title()

    for idx, (df, label) in enumerate(zip(dfs, labels)):
        color = colors[idx % len(colors)]

        # Filter for valid data
        df_filtered = df[df[x_variable].notna() & df[y_variable].notna()].copy()
        try:
            df_filtered["correct"] = df_filtered["correct"].astype(bool)
        except Exception as e:
            print(
                f"Warning: Could not convert 'correct' column to boolean: "
                f"{e}. Using as is."
            )

        df_filtered = df_filtered[df_filtered["correct"]]

        if df_filtered.empty:
            continue

        # Extract x and y values
        x_vals = df_filtered[x_variable].values
        y_vals = df_filtered[y_variable].values

        # Adjust signs for minimization objectives
        x_adjusted = x_vals if x_maximize else -x_vals
        y_adjusted = y_vals if y_maximize else -y_vals

        # Stack into points array for Pareto calculation
        points = np.column_stack([x_adjusted, y_adjusted])

        # Get Pareto mask
        pareto_mask = get_pareto_mask(points)
        pareto_df = df_filtered[pareto_mask].copy()

        # Sort Pareto front for connected line
        pareto_df_sorted = pareto_df.sort_values(
            by=x_variable, ascending=not x_maximize
        )

        # Plot Pareto line
        ax.plot(
            pareto_df_sorted[x_variable],
            pareto_df_sorted[y_variable],
            linewidth=3,
            color=color,
            label=f"{label}",
            alpha=0.8,
            zorder=5,
        )

        # Plot Pareto points
        ax.scatter(
            pareto_df[x_variable],
            pareto_df[y_variable],
            s=150,
            color=color,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.8,
            zorder=6,
        )

    # Set axis limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Invert axes if minimizing
    if not x_maximize:
        ax.invert_xaxis()
    if not y_maximize:
        ax.invert_yaxis()

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, linestyle=":", alpha=0.9, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend(fontsize=25, loc="best")

    if fig:
        fig.tight_layout()

    return fig, ax
