import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_costs(
    df: pd.DataFrame,
    title: str = "Cumulative Costs Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "# Evaluated Programs",
    ylabel: str = "Cumulative Cost ($)",
    plot_individual: bool = True,
    plot_total: bool = True,
):
    """
    Plots the cumulative costs (api_costs, embed_cost, novelty_cost)
    over program evaluations.

    Args:
        df: DataFrame containing cost columns and generation information
        title: Plot title
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        plot_individual: Whether to plot individual cost components
        plot_total: Whether to plot the total cumulative cost

    Returns:
        Tuple of (figure, axes)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Sort by generation
    df = df.sort_values(by="generation")

    # Calculate cumulative costs
    cumulative_api_cost = (
        df["api_costs"].cumsum().bfill()
        if "api_costs" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_embed_cost = (
        df["embed_cost"].cumsum().bfill()
        if "embed_cost" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_novelty_cost = (
        df["novelty_cost"].cumsum().bfill()
        if "novelty_cost" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_meta_cost = (
        df["meta_cost"].fillna(0).cumsum().bfill()
        if "meta_cost" in df.columns
        else pd.Series([0] * len(df))
    )

    # Calculate total cumulative cost
    total_cumulative_cost = (
        cumulative_api_cost
        + cumulative_embed_cost
        + cumulative_novelty_cost
        + cumulative_meta_cost
    )

    handles = []
    labels = []

    # Get final total cost for percentage calculations
    final_total_cost = total_cumulative_cost.iloc[-1]

    # Plot Total and API costs on primary y-axis (left)
    if plot_total:
        line_total = ax.plot(
            df["generation"],
            total_cumulative_cost,
            linewidth=3.5,
            color="red",
            linestyle="-",
            label=f"Total Cost (${final_total_cost:.2f}, 100%)",
            alpha=1.0,
        )
        handles.extend(line_total)
        labels.append(f"Total Cost (${final_total_cost:.2f}, 100%)")

    if plot_individual:
        if "api_costs" in df.columns and cumulative_api_cost.max() > 0:
            final_api_cost = cumulative_api_cost.iloc[-1]
            api_pct = (
                (final_api_cost / final_total_cost * 100) if final_total_cost > 0 else 0
            )
            api_label = f"Program Costs (${final_api_cost:.2f}, {api_pct:.1f}%)"
            line1 = ax.plot(
                df["generation"],
                cumulative_api_cost,
                linewidth=2.5,
                color="blue",
                linestyle="-",
                label=api_label,
                alpha=0.8,
            )
            handles.extend(line1)
            labels.append(api_label)

    # Create secondary y-axis for Embedding and Novelty costs
    ax2 = ax.twinx()

    if plot_individual:
        if "embed_cost" in df.columns and cumulative_embed_cost.max() > 0:
            final_embed_cost = cumulative_embed_cost.iloc[-1]
            embed_pct = (
                (final_embed_cost / final_total_cost * 100)
                if final_total_cost > 0
                else 0
            )
            embed_label = f"Embedding Costs (${final_embed_cost:.2f}, {embed_pct:.1f}%)"
            line2 = ax2.plot(
                df["generation"],
                cumulative_embed_cost,
                linewidth=2.5,
                color="green",
                linestyle="--",
                label=embed_label,
                alpha=0.8,
            )
            handles.extend(line2)
            labels.append(embed_label)

        if "novelty_cost" in df.columns and cumulative_novelty_cost.max() > 0:
            final_novelty_cost = cumulative_novelty_cost.iloc[-1]
            novelty_pct = (
                (final_novelty_cost / final_total_cost * 100)
                if final_total_cost > 0
                else 0
            )
            novelty_label = (
                f"Novelty Costs (${final_novelty_cost:.2f}, {novelty_pct:.1f}%)"
            )
            line3 = ax2.plot(
                df["generation"],
                cumulative_novelty_cost,
                linewidth=2.5,
                color="purple",
                linestyle="--",
                label=novelty_label,
                alpha=0.8,
            )
            handles.extend(line3)
            labels.append(novelty_label)

        if "meta_cost" in df.columns and cumulative_meta_cost.max() > 0:
            final_meta_cost = cumulative_meta_cost.iloc[-1]
            meta_pct = (
                (final_meta_cost / final_total_cost * 100)
                if final_total_cost > 0
                else 0
            )
            meta_label = f"Meta Costs (${final_meta_cost:.2f}, {meta_pct:.1f}%)"
            line4 = ax2.plot(
                df["generation"],
                cumulative_meta_cost,
                linewidth=2.5,
                color="orange",
                linestyle="--",
                label=meta_label,
                alpha=0.8,
            )
            handles.extend(line4)
            labels.append(meta_label)

    # Customize primary y-axis (left)
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(
        "Total & Program Generation ($)",
        fontsize=30,
        weight="bold",
        labelpad=15,
    )
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Customize secondary y-axis (right)
    ax2.set_ylabel(
        "Cum. Cost - Embedding, Novelty & Meta ($)",
        fontsize=22,
        weight="bold",
        labelpad=15,
        color="darkgreen",
    )
    ax2.tick_params(axis="y", which="major", labelsize=20, labelcolor="darkgreen")

    # Remove top spines
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Add legend
    if handles:
        ax.legend(handles, labels, fontsize=25, loc="upper left")

    fig.tight_layout()

    return fig, ax


def plot_cost_performance(
    df: pd.DataFrame,
    title: str = "Best Combined Score vs Total Cost",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Cumulative Total Cost ($)",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    score_column: str = "combined_score",
    scatter_improvements_only: bool = False,
):
    """
    Plots the cumulative maximum performance against cumulative total costs.

    Args:
        df: DataFrame containing cost columns and score information
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

    # Sort by generation and filter for correct programs
    df = df.sort_values(by="generation")
    df_filtered = df[df["correct"]].copy()

    # Calculate cumulative costs
    cumulative_api_cost = (
        df["api_costs"].cumsum().bfill()
        if "api_costs" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_embed_cost = (
        df["embed_cost"].cumsum().bfill()
        if "embed_cost" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_novelty_cost = (
        df["novelty_cost"].cumsum().bfill()
        if "novelty_cost" in df.columns
        else pd.Series([0] * len(df))
    )
    cumulative_meta_cost = (
        df["meta_cost"].fillna(0).cumsum().bfill()
        if "meta_cost" in df.columns
        else pd.Series([0] * len(df))
    )

    # Calculate total cumulative cost
    total_cumulative_cost = (
        cumulative_api_cost
        + cumulative_embed_cost
        + cumulative_novelty_cost
        + cumulative_meta_cost
    )

    # Align filtered indices with full df for cost lookup
    df_filtered["total_cost"] = total_cumulative_cost.loc[df_filtered.index]

    # Calculate cumulative maximum score
    cummax_scores = df_filtered[score_column].cummax()

    # Plot the cummax line
    line1 = ax.plot(
        df_filtered["total_cost"],
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
            improvements["total_cost"],
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
            df_filtered["total_cost"],
            df_filtered[score_column],
            alpha=1.0,
            s=40,
            color="black",
            label="Individual Evals",
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
    handles = line1 + [scatter1]
    # Filter out matplotlib auto-generated labels (starting with _)
    labels = [h.get_label() for h in handles if not h.get_label().startswith("_")]
    handles = [h for h in handles if not h.get_label().startswith("_")]
    ax.legend(handles, labels, fontsize=25, loc="lower right")

    fig.tight_layout()

    return fig, ax


def plot_cost_performance_compare(
    dfs: List[pd.DataFrame],
    labels: List[str],
    title: str = "Best Combined Score vs Total Cost - Comparison",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Cumulative Total Cost ($)",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    score_column: str = "combined_score",
    scatter_improvements_only: bool = False,
    colors: Optional[List[str]] = None,
):
    """
    Plots comparison of multiple runs: performance vs cumulative costs.

    Args:
        dfs: List of DataFrames containing cost and score information
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

        # Sort by generation and filter for correct programs
        df = df.sort_values(by="generation")
        df_filtered = df[df["correct"]].copy()

        if df_filtered.empty:
            continue

        # Calculate cumulative costs
        cumulative_api_cost = (
            df["api_costs"].cumsum().bfill()
            if "api_costs" in df.columns
            else pd.Series([0] * len(df))
        )
        cumulative_embed_cost = (
            df["embed_cost"].cumsum().bfill()
            if "embed_cost" in df.columns
            else pd.Series([0] * len(df))
        )
        cumulative_novelty_cost = (
            df["novelty_cost"].cumsum().bfill()
            if "novelty_cost" in df.columns
            else pd.Series([0] * len(df))
        )
        cumulative_meta_cost = (
            df["meta_cost"].fillna(0).cumsum().bfill()
            if "meta_cost" in df.columns
            else pd.Series([0] * len(df))
        )

        # Calculate total cumulative cost
        total_cumulative_cost = (
            cumulative_api_cost
            + cumulative_embed_cost
            + cumulative_novelty_cost
            + cumulative_meta_cost
        )

        # Align filtered indices with full df for cost lookup
        df_filtered["total_cost"] = total_cumulative_cost.loc[df_filtered.index]

        # Calculate cumulative maximum score
        cummax_scores = df_filtered[score_column].cummax()

        # Plot the cummax line
        line = ax.plot(
            df_filtered["total_cost"],
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
                    improvements["total_cost"],
                    improvements[score_column],
                    alpha=0.6,
                    s=80,
                    color=color,
                    marker="*",
                    zorder=5,
                )
        else:
            ax.scatter(
                df_filtered["total_cost"],
                df_filtered[score_column],
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
