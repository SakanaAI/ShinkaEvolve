import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Get the number of LLM calls for each program and add inset bar plot for cumulative api costs
def plot_cumulative_llm_calls(
    df,
    title="Cumulative LLM Calls Over Time by Model",
    fig=None,
    ax=None,
):
    """
    Plots cumulative LLM calls over time for each model,
    and an inset bar plot for total api_costs by model.
    Orders model names by average API cost (descending) in the inset.
    Args:
        df: pandas DataFrame with 'llm_result' and 'api_costs' columns.
        ax: matplotlib axes object to plot on.
    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Extract model names for all rows
    model_names = df["llm_result"].apply(
        lambda x: x.get("model_name")
        if isinstance(x, dict) and "model_name" in x
        else None
    )

    # Defensive extraction of costs, treat missing as 0
    api_costs = df["api_costs"].apply(
        lambda x: float(x) if x not in [None, "None"] else 0.0
    )

    # Compute average cost per model (for sorting)
    model_avg_cost = (
        df.assign(model_name=model_names, api_cost=api_costs)
        .groupby("model_name")["api_cost"]
        .mean()
        .sort_values(ascending=False)  # descending: highest cost first
    )
    # Order of models for plotting
    ordered_models = model_avg_cost.index.tolist()

    # Plot cumulative LLM calls for each model in sorted order
    for model in ordered_models:
        if model is not None:
            model_mask = model_names == model
            model_cumsum = model_mask.astype(int).cumsum()
            ax.plot(df.index, model_cumsum, label=f"{model}")

    # Cumulative total cost per model in this order
    cost_per_model = {
        model: api_costs[model_names == model].sum()
        for model in ordered_models
        if model is not None
    }

    # Inset axes: position = [x0, y0, width, height] (in fraction of parent)
    # Change loc to 'lower right' to position in bottom right
    inset_ax = inset_axes(ax, width="40%", height="35%", loc="lower right", borderpad=2)
    model_names_list = list(cost_per_model.keys())
    total_costs_list = [cost_per_model[m] for m in model_names_list]

    # Bar plot in inset
    # Use same colors for bars as the line plot uses for their lines
    model_to_color = {}
    for line in ax.get_lines():
        lbl = line.get_label()
        if lbl in model_names_list:
            model_to_color[lbl] = line.get_color()
    # Ensure color order matches sorted model order
    bar_colors = [model_to_color.get(m, None) for m in model_names_list]
    bars = inset_ax.bar(model_names_list, total_costs_list, color=bar_colors)
    inset_ax.set_ylabel("Total API Cost", fontsize=12)
    total_sum_cost = sum(total_costs_list)
    inset_ax.set_title(
        f"Cumulative API Cost (Total: ${total_sum_cost:,.2f})",
        fontsize=13,
        weight="bold",
    )
    inset_ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    inset_ax.tick_params(axis="y", labelsize=11)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)

    # Clean up styling of main axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("# Evaluated LLM Program Proposals", fontsize=20, weight="bold")
    ax.set_ylabel("Cumulative LLM Calls", fontsize=20, weight="bold")
    ax.set_title(title, fontsize=25, weight="bold")
    ax.legend(fontsize=20, loc="upper left")
    fig.tight_layout()
