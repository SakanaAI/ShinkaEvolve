from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import matplotlib.cm as cm_module
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_lineage_tree(
    df: pd.DataFrame,
    title="Program Lineage Tree",
    fig: Figure | None = None,
    ax: Axes | None = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Generates a tree visualization of program lineage using matplotlib and
    NetworkX.

    Args:
        df: Pandas DataFrame containing program data. Must include 'id' and
            'parent_id'.
        figsize: Size of the figure (width, height) in inches.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None. Cannot draw tree.")
        return

    # Handle island copies: map them to their original program
    island_copy_mapping = {}
    if "_is_island_copy" in df.columns:
        # Find island copies and their originals
        is_copy_mask = df["_is_island_copy"].fillna(False).astype(bool)
        island_copies = df[is_copy_mask]

        # Create mapping from island copy id to original id
        for idx, row in island_copies.iterrows():
            copy_id = str(row["id"])
            # Find the original by matching generation and looking for non-copy
            if "generation" in df.columns:
                same_gen = df[df["generation"] == row["generation"]]
                same_gen_non_copy = same_gen[
                    ~same_gen["_is_island_copy"].fillna(False).astype(bool)
                ]
                if not same_gen_non_copy.empty:
                    original_id = str(same_gen_non_copy.iloc[0]["id"])
                    island_copy_mapping[copy_id] = original_id

        # Filter out island copies from the dataframe
        df = df[~is_copy_mask].copy()

    # set combined score to 0 for incorrect programs
    df.loc[~df["correct"], "combined_score"] = 0

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with attributes for labels
    for idx, row in df.iterrows():
        node_id = str(row["id"])
        node_attrs = {}

        # Add available metrics as node attributes
        for col in df.columns:
            if col in row:
                # Skip the code column as it's usually too long
                if col != "code":
                    node_attrs[col] = row[col]

        G.add_node(node_id, **node_attrs)

    # Add edges with parent remapping for island copies
    for idx, row in df.iterrows():
        child_id = str(row["id"])
        if "parent_id" in row and pd.notna(row["parent_id"]):
            parent_id = str(row["parent_id"])

            # Remap parent if it was an island copy
            if parent_id in island_copy_mapping:
                parent_id = island_copy_mapping[parent_id]

            # Check if parent exists and is not self-referential
            if parent_id in G.nodes() and parent_id != child_id:
                G.add_edge(parent_id, child_id)

    # Create figure with a specific axes for the graph and colorbar
    # Ensure both fig and ax are created together
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 16))

    # Group nodes by generation to ensure proper ordering
    generation_groups = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        gen = attrs.get("generation", 0)  # Default to 0 if no generation found
        if gen not in generation_groups:
            generation_groups[gen] = []
        generation_groups[gen].append(node)

    # Identify the roots (earliest generation nodes)
    roots = [n for n, d in G.in_degree() if d == 0]

    if not roots:
        # If no root is found, use nodes from earliest generation
        min_gen = min(generation_groups.keys()) if generation_groups else 0
        roots = generation_groups.get(min_gen, [list(G.nodes())[0]])

    root = roots[0]

    # Initialize levels dict (will be populated if using manual layout)
    levels = {}

    # Try to use a hierarchical layout that respects parent-child relationships
    # Focus on creating a clean tree structure like the reference image
    try:
        # Use dot layout for hierarchical tree structure with edge crossing
        # minimization
        pos = nx.nx_agraph.graphviz_layout(
            G,
            prog="dot",
            root=root,
            args="-Grankdir=TB -Gsplines=true -Goverlap=false -Gsep=1.0",
        )
    except ImportError:
        try:
            # If pygraphviz not available, try pydot (no args parameter)
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot", root=root)
        except ImportError:
            print("GraphViz not available, using hierarchical layout")
            # Create a clean hierarchical layout manually
            pos = {}

            # Find node depths based on distance from root
            depths = {}
            for node in G.nodes():
                try:
                    path_len = len(nx.shortest_path(G, root, node)) - 1
                    depths[node] = path_len
                except nx.NetworkXNoPath:
                    # If no path, use generation if available
                    if "generation" in G.nodes[node]:
                        depths[node] = G.nodes[node]["generation"]
                    else:
                        depths[node] = 0

            # Group nodes by depth/generation
            levels = {}
            for node, depth in depths.items():
                if depth not in levels:
                    levels[depth] = []
                levels[depth].append(node)

            # Create clean hierarchical positioning
            max_depth = max(levels.keys()) if levels else 0
            # Total nodes in graph for base spacing
            total_nodes = len(G.nodes())
            for depth in sorted(levels.keys()):
                nodes_at_level = levels[depth]
                num_nodes_at_level = len(nodes_at_level)

                if depth == 0:
                    # Root node at center top
                    if num_nodes_at_level == 1:
                        pos[nodes_at_level[0]] = (0, 0)
                    else:
                        # Multiple roots - space them out horizontally
                        spacing = 25.0
                        total_width = (num_nodes_at_level - 1) * spacing
                        start_x = -total_width / 2
                        for i, node in enumerate(nodes_at_level):
                            pos[node] = (start_x + i * spacing, 0)
                else:
                    # For non-root levels, try to position children near
                    # parents
                    # to minimize crossings

                    # First, collect parent positions for each node
                    node_parent_info = {}
                    for node in nodes_at_level:
                        parent_x_positions = []
                        for parent in G.predecessors(node):
                            if parent in pos:
                                parent_x_positions.append(pos[parent][0])

                        if parent_x_positions:
                            avg_parent_x = sum(parent_x_positions) / len(
                                parent_x_positions
                            )
                            node_parent_info[node] = avg_parent_x
                        else:
                            node_parent_info[node] = 0

                    # Sort nodes by their parent positions to reduce crossings
                    sorted_nodes = sorted(
                        nodes_at_level, key=lambda n: node_parent_info[n]
                    )

                    # Position nodes with adequate spacing - more aggressive
                    # for early levels
                    # Base spacing should prevent overlapping at all levels
                    base_spacing = max(25.0, 15.0 * (total_nodes**0.5))
                    # Extra spacing for early levels where nodes tend to
                    # cluster
                    depth_multiplier = max(1.0, 3.0 / (depth + 1))

                    spacing = base_spacing * depth_multiplier

                    y_pos = -depth * 5.0  # Increased vertical spacing

                    if num_nodes_at_level == 1:
                        pos[sorted_nodes[0]] = (0, y_pos)
                    else:
                        total_width = (num_nodes_at_level - 1) * spacing
                        start_x = -total_width / 2

                        for i, node in enumerate(sorted_nodes):
                            x_pos = start_x + i * spacing
                            pos[node] = (x_pos, y_pos)

                            # Fine-tune position to be closer to parent if possible
                            if node in node_parent_info:
                                preferred_x = node_parent_info[node]
                                # Check if we can move closer to parent without
                                # overlapping other nodes - use stricter minimum distance
                                min_distance = spacing * 0.8
                                can_move = True

                                for other_node in sorted_nodes:
                                    if other_node != node and other_node in pos:
                                        other_x = pos[other_node][0]
                                        if abs(preferred_x - other_x) < min_distance:
                                            can_move = False
                                            break

                                if can_move:
                                    # Move towards parent but very conservatively
                                    adjustment = (preferred_x - x_pos) * 0.1
                                    pos[node] = (x_pos + adjustment, y_pos)

    # Additional fine-tuning to reduce crossings
    if pos and len(pos) > 1:
        # Try to reduce crossings by adjusting positions within each level
        for depth in sorted(levels.keys()):
            if depth == 0:  # Skip root
                continue

            nodes_at_level = levels[depth]
            if len(nodes_at_level) <= 1:
                continue

            # Calculate crossing score for current arrangement
            def count_crossings(node_positions):
                crossings = 0
                for i, node1 in enumerate(nodes_at_level):
                    for j, node2 in enumerate(nodes_at_level):
                        if i >= j:
                            continue

                        # Get parents of both nodes
                        parents1 = list(G.predecessors(node1))
                        parents2 = list(G.predecessors(node2))

                        for p1 in parents1:
                            for p2 in parents2:
                                if p1 in pos and p2 in pos:
                                    # Check if edges cross
                                    p1_x, p1_y = pos[p1]
                                    p2_x, p2_y = pos[p2]
                                    n1_x = node_positions[node1][0]
                                    n2_x = node_positions[node2][0]

                                    # Simple crossing check
                                    if (p1_x < p2_x and n1_x > n2_x) or (
                                        p1_x > p2_x and n1_x < n2_x
                                    ):
                                        crossings += 1
                return crossings

            # Try to improve by swapping adjacent nodes
            improved = True
            max_iterations = 10
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(len(nodes_at_level) - 1):
                    node1 = nodes_at_level[i]
                    node2 = nodes_at_level[i + 1]

                    # Create temporary positions with swapped nodes
                    temp_positions = dict(pos)
                    temp_positions[node1] = pos[node2]
                    temp_positions[node2] = pos[node1]

                    # Check if this reduces crossings
                    original_crossings = count_crossings(pos)
                    new_crossings = count_crossings(temp_positions)

                    if new_crossings < original_crossings:
                        # Apply the swap
                        pos[node1] = temp_positions[node1]
                        pos[node2] = temp_positions[node2]
                        # Also swap in the nodes list
                        nodes_at_level[i], nodes_at_level[i + 1] = (
                            nodes_at_level[i + 1],
                            nodes_at_level[i],
                        )
                        improved = True

    # Center the initial node (root) at x=0 after all layout adjustments
    if pos and root in pos:
        # Get the current position of the root node
        root_x, root_y = pos[root]

        # Calculate offset to place root at x=0
        x_offset = -root_x

        # Apply offset to all nodes to center the root
        for node in pos:
            x, y = pos[node]
            pos[node] = (x + x_offset, y)

    # Calculate base node sizes based on number of nodes and axes size
    num_nodes = len(G.nodes())

    # Get axes size in inches (works with subplots)
    bbox = ax.get_position()  # Get axes position in figure coordinates
    fig_width, fig_height = fig.get_size_inches()
    # Calculate actual axes size in inches
    ax_width = bbox.width * fig_width
    ax_height = bbox.height * fig_height

    # Calculate a size scale based on axes dimensions
    # Normalize to a reference size of 20x16 inches
    ax_scale = ((ax_width / 20.0) + (ax_height / 16.0)) / 2.0

    # Scale node sizes based on both number of nodes and axes size
    size_factor = 0.65 * max(0.3, min(1.0, 20 / (num_nodes**0.4)))
    # Apply axes scaling
    size_factor = size_factor * ax_scale

    best_node_size = int(1500 * size_factor)
    path_node_size = int(800 * size_factor)
    regular_node_size = int(600 * size_factor)

    # Find min and max combined_score to create color map
    score_values = []
    score_field = "combined_score"  # As per user's request

    # Find the best node (highest score)
    best_node = None
    best_score = float("-inf")

    for node in G.nodes():
        if score_field in G.nodes[node]:
            score = G.nodes[node][score_field]
            if isinstance(score, (int, float)):
                score_values.append(score)
                if score > best_score:
                    best_score = score
                    best_node = node

    # Find the path from root to the best node (if it exists)
    path_to_best = []
    best_path_edges = []
    if best_node:
        try:
            # Find shortest path from root to best node
            path_to_best = nx.shortest_path(G, root, best_node)
            # Create list of edge tuples in the path
            best_path_edges = list(zip(path_to_best[:-1], path_to_best[1:]))
        except nx.NetworkXNoPath:
            # No path exists, keep the lists empty
            pass

    # Draw regular edges first (thinner, black)
    regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in best_path_edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=regular_edges,
        arrows=False,
        arrowsize=12,
        width=1.5,
        edge_color="black",
        alpha=0.6,
        ax=ax,
    )

    # Draw the edges in the path to the best node (thicker, black like reference)
    if best_path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_path_edges,
            arrows=False,
            arrowsize=20,
            width=3.5,
            edge_color="black",
            alpha=0.9,
            ax=ax,
        )

    if score_values:
        if vmin is None:
            vmin = min(score_values)
        if vmax is None:
            vmax = max(score_values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Create colormap using proper method - viridis has good contrast with black text
        color_map = cm_module.get_cmap("viridis")

        # Draw nodes with colors based on combined_score
        for node in G.nodes():
            node_attrs = G.nodes[node]
            current_node_size = regular_node_size
            # Default for nodes without valid score
            current_node_color = "lightgray"
            current_edge_color = "black"
            current_linewidth = 1.5
            current_node_shape = "o"  # Default shape, circle for "diff"

            # Check if node is incorrect first (overrides other shape logic)
            is_correct = node_attrs.get("correct", True)  # Default to True
            if not is_correct:
                current_node_shape = "x"  # X shape for incorrect nodes
                current_node_color = "red"
                current_edge_color = "darkred"
                current_linewidth = 4.0  # Thicker line for X
                current_node_size = int(current_node_size * 1.5)  # Larger size
            else:
                # Determine shape based on patch_type (only for correct nodes)
                patch_type = node_attrs.get("patch_type")
                if patch_type == "full":
                    current_node_shape = "s"  # Square
                elif patch_type == "init":
                    current_node_shape = "^"  # Triangle up
                # elif patch_type == "paper":
                #     current_node_shape = "d"  # Diamond
                elif patch_type == "cross":
                    current_node_shape = "P"  # Plus (filled)

                if score_field in node_attrs:
                    score = node_attrs[score_field]
                    if pd.isna(score):  # Check for NaN
                        current_node_color = "purple"  # Highlight NaN scores
                    elif isinstance(score, (int, float)):
                        color = color_map(norm(score))
                        current_node_color = mcolors.to_hex(color)

            # Check if this is the best node (overrides all except incorrect)
            if node == best_node and is_correct:  # Only if correct
                current_node_size = best_node_size
                current_node_color = "gold"
                current_edge_color = "black"
                current_linewidth = 2.5
                current_node_shape = "*"  # Star shape for best node
            elif node in path_to_best and is_correct:  # Only if correct
                current_node_size = path_node_size
                # Color for path nodes:
                # - Score color if valid
                # - Purple if NaN
                # - Lightgray otherwise
                node_score = node_attrs.get(score_field)
                if node_score is not None and not pd.isna(node_score):
                    # Valid numeric score?
                    if isinstance(node_score, (int, float)):
                        color = color_map(norm(node_score))
                        current_node_color = mcolors.to_hex(color)
                current_edge_color = "black"
                current_linewidth = 2.0
                # Keep shape determined by patch_type unless best node

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=current_node_size,
                node_color=current_node_color,
                edgecolors=current_edge_color,
                linewidths=current_linewidth,
                ax=ax,
                node_shape=current_node_shape,
            )

        # Add colorbar with proper axes reference
        sm = cm_module.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(
            sm,
            ax=ax,  # type: ignore[arg-type]
            pad=-0.05,
            shrink=0.6,
        )
        cb.set_label(label="Combined Fitness Score", size=20, weight="bold")
        cb.ax.tick_params(labelsize=16)
    else:
        # Draw all nodes with default color if no scores available
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=regular_node_size,
            node_color="lightblue",
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

    # Prepare simple node labels with generation and combined_score
    node_labels = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        label_parts = []

        # Add generation if available
        if "generation" in attrs:
            label_parts.append(f"{attrs['generation']}")

        # # Add combined_score if available
        # if score_field in attrs:
        #     value = attrs[score_field]
        #     if isinstance(value, float):
        #         label_parts.append(f"{value:.1f}")
        #     else:
        #         label_parts.append(f"{value}")

        # # Join with newline
        if label_parts:
            # node_labels[node] = "\n".join(label_parts)
            node_labels[node] = label_parts[0]
        # else:
        #     # Just use a short version of the ID if no other info available
        #     node_labels[node] = f"{node[:8]}"

    # Create a new position dictionary for labels with adjusted y-coordinates
    # to place labels above nodes
    label_pos = {}
    for node, (x, y) in pos.items():
        # Move labels slightly above the nodes
        label_pos[node] = (x, y + 0.0)

    # Draw the labels with better font options at adjusted positions
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=node_labels,
        font_size=12,
        font_weight="bold",
        font_color="white",
        ax=ax,
    )

    # Add legend for the star shape and paths
    if best_node:
        star_patch = Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=20,
            label="Best Score",
        )
        # Create line legend with appropriate width
        path_line = Line2D([0], [0], color="red", linewidth=4, label=r"Path$\to$Best")
        # Legend for patch types
        diff_patch = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Diff Edit",
            markerfacecolor="gray",
            markersize=10,
        )
        full_patch = Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Full Edit",
            markerfacecolor="gray",
            markersize=10,
        )
        init_patch = Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Initial",
            markerfacecolor="gray",
            markersize=10,
        )
        # paper_patch = Line2D(
        #     [0],
        #     [0],
        #     marker="d",  # Diamond
        #     color="w",
        #     label="Paper Edit",
        #     markerfacecolor="gray",
        #     markersize=10,
        # )
        crossover_patch = Line2D(
            [0],
            [0],
            marker="P",  # Plus (filled)
            color="w",
            label="Cross-Over",
            markerfacecolor="gray",
            markersize=10,
        )
        incorrect_patch = Line2D(
            [0],
            [0],
            marker="x",  # X shape
            color="w",
            label="Incorrect",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=15,
            markeredgewidth=3,
        )

        legend_handles = [
            star_patch,
            # path_line,
            diff_patch,
            full_patch,
            init_patch,
            # paper_patch,
            crossover_patch,
            incorrect_patch,
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            fontsize=25,
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )

    ax.set_title(title, fontsize=40, fontweight="bold")
    ax.axis("off")
    # Use subplots_adjust for more control over margins
    # Increase bottom margin to make room for legend
    fig.subplots_adjust(left=0.02, right=0.85, top=0.95, bottom=0.08)
    return fig, ax
