import os
import openai
import google.generativeai as genai
import pandas as pd
from typing import Union, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


M = 1_000_000

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AZURE_EMBEDDING_MODELS = [
    "azure-text-embedding-3-small",
    "azure-text-embedding-3-large",
]

GEMINI_EMBEDDING_MODELS = [
    "gemini-embedding-exp-03-07",
    "gemini-embedding-001",
]

OPENAI_EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02 / M,
    "text-embedding-3-large": 0.13 / M,
}

# Gemini embedding costs (approximate - check current pricing)
GEMINI_EMBEDDING_COSTS = {
    "gemini-embedding-exp-03-07": 0.0 / M,  # Experimental model, often free
    "gemini-embedding-001": 0.0 / M,  # Check current pricing
}

def get_client_model(model_name: str) -> tuple[Union[openai.OpenAI, str], str]:
    if model_name in OPENAI_EMBEDDING_MODELS:
        client = openai.OpenAI()
        model_to_use = model_name
    elif model_name in AZURE_EMBEDDING_MODELS:
        # get rid of the azure- prefix
        model_to_use = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif model_name in GEMINI_EMBEDDING_MODELS:
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini models")
        genai.configure(api_key=api_key)
        client = "gemini"  # Use string identifier for Gemini
        model_to_use = model_name
    else:
        raise ValueError(f"Invalid embedding model: {model_name}")

    return client, model_to_use


class EmbeddingClient:
    def __init__(
        self, model_name: str = "text-embedding-3-small", verbose: bool = False
    ):
        """
        Initialize the EmbeddingClient.

        Args:
            model (str): The OpenAI, Azure, or Gemini embedding model name to use.
        """
        self.client, self.model = get_client_model(model_name)
        self.model_name = model_name
        self.verbose = verbose

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Union[Tuple[List[float], float], Tuple[List[List[float]], float]]:
        """
        Computes the text embedding for a CUDA kernel string.

        Args:
            code (str, list[str]): The CUDA kernel code as a string or list
                of strings.

        Returns:
            list: Embedding vector for the kernel code or None if an error
                occurs.
        """
        if isinstance(code, str):
            code = [code]
            single_code = True
        else:
            single_code = False
        # Handle Gemini models
        if self.model_name in GEMINI_EMBEDDING_MODELS:
            try:
                embeddings = []
                total_tokens = 0
                
                for text in code:
                    result = genai.embed_content(
                        model=f"models/{self.model}",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                    total_tokens += len(text.split())
                
                cost = total_tokens * GEMINI_EMBEDDING_COSTS.get(self.model, 0.0)
                
                if single_code:
                    return embeddings[0] if embeddings else [], cost
                else:
                    return embeddings, cost
            except Exception as e:
                logger.error(f"Error getting Gemini embedding: {e}")
                if single_code:
                    return [], 0.0
                else:
                    return [[]], 0.0
        # Handle OpenAI and Azure models (same interface)
        try:
            response = self.client.embeddings.create(
                model=self.model, input=code, encoding_format="float"
            )
            cost = response.usage.total_tokens * OPENAI_EMBEDDING_COSTS[self.model]
            # Extract embedding from response
            if single_code:
                return response.data[0].embedding, cost
            else:
                return [d.embedding for d in response.data], cost
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            if single_code:
                return [], 0.0
            else:
                return [[]], 0.0

    def get_column_embedding(
        self,
        df: pd.DataFrame,
        column_name: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Computes the text embedding for a batch of CUDA kernel strings.

        Args:
            df (pd.DataFrame): A pandas DataFrame with the column to embed.
            column_name (str, list): The name of the columns to embed.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the column to embed.
        """
        if isinstance(column_name, str):
            column_name = [column_name]

        for column_name in column_name:
            model_name_str = self.model.replace("-", "_")
            new_col_name = f"{column_name}_embedding_{model_name_str}"
            df[new_col_name] = df[column_name].apply(
                lambda x: self.get_embedding(x),
            )
        return df

    def get_closest_k_neighbors(
        self,
        new_str_query: str,
        embeddings: list,
        top_k: Union[int, str] = 5,
    ) -> tuple[list, list]:
        """Get k closest neighbors from the embeddings list

        Args:
            new_str_query: The string to get the closest neighbors for.
            embeddings: The list of embeddings to compare against.
            top_k: The number of closest neighbors to return.

        Returns:
            A tuple of the top k indices and the top k similarities.
        """
        # get embedding of the new string
        new_embedding, _ = self.get_embedding(new_str_query)

        if not new_embedding:  # Handle case where embedding fails
            return [], []

        # define cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # compute the cosine similarity between the new embed
        similarities = [
            cosine_similarity(new_embedding, embedding) for embedding in embeddings
        ]

        # get the top k neighbors or random rows
        if top_k == "random":
            if len(similarities) < 5:
                top_idx = np.random.choice(
                    len(similarities), size=len(similarities), replace=False
                )
            else:
                top_idx = np.random.choice(len(similarities), size=5, replace=False)
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx.tolist(), similarities_subset
        elif isinstance(top_k, int):
            top_idx = np.argsort(similarities)[-top_k:]
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx[::-1].tolist(), similarities_subset[::-1]
        else:
            raise ValueError("top_k must be an int or 'random'")

    def get_dim_reduction(
        self,
        embeddings: list,
        method: str = "pca",
        dims: int = 2,
    ):
        """Performs dimensionality reduction on a list of embeddings using
        various methods.

        Args:
            embeddings: List of embedding vectors
            method: Dimensionality reduction method ('pca', 'umap', or 'tsne')
            dims: Number of dimensions to reduce to

        Returns:
            The transformed embeddings in reduced dimensionality
        """
        if isinstance(embeddings, pd.Series):
            embeddings = embeddings.tolist()

        # Convert list to numpy array if needed
        X = np.array(embeddings) if isinstance(embeddings, list) else embeddings
        # preprocess the embeddings using standard scaler
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if method.lower() == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=dims)
            return model.fit_transform(X)
        elif method.lower() == "umap":
            from umap import UMAP

            model = UMAP(n_components=dims, random_state=42)
            return model.fit_transform(X)
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE

            model = TSNE(n_components=dims, random_state=42)
            return model.fit_transform(X)
        else:
            raise ValueError("Method must be one of: 'pca', 'umap', 'tsne'")

    def get_embedding_clusters(
        self,
        embeddings: list,
        num_clusters: int = 4,
        verbose: bool = False,
    ) -> list:
        """
        Performs clustering on a list of embeddings using Gaussian Mixture Model.

        Args:
            embeddings: List of embedding vectors
            num_clusters: Number of clusters to form with GMM.
            top_k_candidates: Number of top kernels to select per cluster.
            verbose: If True, prints detailed cluster information.

        Returns:
            pd.DataFrame: A DataFrame with top candidate kernels from each
            cluster.
        """
        from sklearn.mixture import GaussianMixture

        # Perform GMM clustering on the PCA-reduced embeddings
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(embeddings)
        clusters = gmm.predict(embeddings)

        # Optionally display detailed cluster information
        if verbose:
            logger.info(
                f"GMM {num_clusters} Clusters ==> Got {len(embeddings)} "
                f"embeddings with cluster assignments:"
            )
            num_members = pd.Series(clusters).value_counts()
            logger.info(num_members)

        return clusters

    def plot_reduced_embeddings(
        self,
        embeddings: list,
        method: str = "pca",
        num_dims: int = 3,
        title="Embedding",
        cluster_ids: Optional[list] = None,
        cluster_label: str = "Cluster",
        patch_type: Optional[list] = None,
    ):
        transformed = self.get_dim_reduction(embeddings, method, num_dims)

        if num_dims == 2:
            fig, ax = plot_2d_scatter(
                transformed, title, cluster_ids, cluster_label, patch_type
            )
        elif num_dims == 3:
            fig, ax = plot_3d_scatter(
                transformed, title, cluster_ids, cluster_label, patch_type
            )
        else:
            raise ValueError(f"Invalid number of dimensions: {num_dims}")

        return fig, ax


def plot_2d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    # Create figure and 2D axes with adjusted size and spacing
    fig, ax = plt.subplots(figsize=(10, 7))

    # Prepare cluster IDs and colormap
    if cluster_ids is not None:
        original_unique_ids, cluster_ids_for_coloring = np.unique(
            cluster_ids, return_inverse=True
        )
        num_distinct_colors = len(original_unique_ids)
        # Ensure cluster_ids_array for c= in scatter is the 0-indexed version
        # This was previously cluster_ids_array = np.array(cluster_ids)
        # Now it's cluster_ids_for_coloring
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        original_unique_ids = [
            0
        ]  # For consistent ticks if colorbar is ever shown for no clusters
        num_distinct_colors = 1

    # Create discrete colormap
    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    if num_distinct_colors > 0:
        multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
        extended_colors = base_colors * multiplier
        colors_for_cmap = extended_colors[:num_distinct_colors]
    else:  # Should not happen if original_unique_ids is at least [0]
        colors_for_cmap = ["blue"]

    cmap = ListedColormap(colors_for_cmap)

    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            current_marker = marker_shapes[i % len(marker_shapes)]

            c_val_scatter = None
            cmap_val_scatter = (
                None  # Define cmap_val_scatter to avoid UnboundLocalError
            )
            if cluster_ids is not None:
                c_val_scatter = cluster_ids_for_coloring[patch_mask]
                cmap_val_scatter = cmap

            label_text = str(patch_val)

            scatter_args = {
                "marker": current_marker,
                "alpha": 0.6,
                "s": 100,
                "label": label_text,
            }
            if c_val_scatter is not None:  # Check c_val_scatter
                scatter_args["c"] = c_val_scatter
                scatter_args["cmap"] = cmap_val_scatter

            ax.scatter(
                transformed[patch_mask, 0],  # PC1
                transformed[patch_mask, 1],  # PC2
                **scatter_args,
            )
    else:  # No patch_type
        c_val_scatter_else = None
        if cluster_ids is not None:
            c_val_scatter_else = (
                cluster_ids_for_coloring  # Use 0-indexed IDs for coloring
            )

        # cmap is already defined based on cluster_ids_for_coloring

        scatter_args_else = {"marker": "o", "alpha": 0.6, "s": 100}
        if (
            c_val_scatter_else is not None
        ):  # Check c_val_scatter_else instead of cluster_ids
            scatter_args_else["c"] = c_val_scatter_else
            scatter_args_else["cmap"] = cmap  # Use the globally defined cmap

        ax.scatter(
            transformed[:, 0],  # PC1
            transformed[:, 1],  # PC2
            **scatter_args_else,
        )

    # Add labels and title with adjusted padding
    ax.set_xlabel("1st Latent Dim.", fontsize=20)
    ax.set_ylabel("2nd Latent Dim.", fontsize=20)
    ax.set_title(title, fontsize=30)

    # no spines for right and top
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Add colorbar with discrete levels
    if (
        cluster_ids is not None
    ):  # Simplified condition: show colorbar if cluster_ids are present
        try:
            # Use an invisible scatter plot with all data points for a robust
            # colorbar. Ensure this uses the 0-indexed
            # cluster_ids_for_coloring for correct color mapping
            ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                c=cluster_ids_for_coloring,  # Use 0-indexed IDs for mapping
                cmap=cmap,  # Use the main cmap
                s=0,
                alpha=0,
            )
            # # Ticks should correspond to original unique cluster ID values
            # if len(original_unique_ids) > 1 or (
            #     len(original_unique_ids) == 1 and original_unique_ids[0] != 0
            # ):  # Only show colorbar if meaningful clusters
            #     colorbar = plt.colorbar(
            #         temp_scatter_for_colorbar,
            #         ticks=original_unique_ids,
            #         shrink=0.4,
            #     )
            #     colorbar.set_label(cluster_label, fontsize=20)
        except Exception:
            pass  # Silently pass

    if patch_type is not None:
        # Create custom legend handles for black markers
        legend_handles = []
        unique_patches_for_legend = np.unique(np.array(patch_type))
        for i, patch_val in enumerate(unique_patches_for_legend):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_shapes[i % len(marker_shapes)],
                    color="black",
                    label=str(patch_val),
                    linestyle="None",
                    markersize=10,
                )
            )
        if legend_handles:
            ax.legend(handles=legend_handles, title="Patch Types", loc="best")

    fig.tight_layout()
    # Remove subplot_adjust for legend as it's now inside
    # if patch_type is not None and legend_handles:
    #     plt.subplots_adjust(right=0.75)

    return fig, ax


def plot_3d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap

    # Create figure and 3D axes with adjusted size and spacing
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Prepare cluster IDs and colormap
    if cluster_ids is not None:
        original_unique_ids, cluster_ids_for_coloring = np.unique(
            cluster_ids, return_inverse=True
        )
        num_distinct_colors = len(original_unique_ids)
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        original_unique_ids = [0]
        num_distinct_colors = 1

    # Create discrete colormap
    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    if num_distinct_colors > 0:
        multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
        extended_colors = base_colors * multiplier
        colors_for_cmap = extended_colors[:num_distinct_colors]
    else:
        colors_for_cmap = ["blue"]

    cmap = ListedColormap(colors_for_cmap)

    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            current_marker = marker_shapes[i % len(marker_shapes)]

            c_val_scatter = None
            cmap_val_scatter = None
            if cluster_ids is not None:
                c_val_scatter = cluster_ids_for_coloring[patch_mask]
                cmap_val_scatter = cmap

            label_text = str(patch_val)

            scatter_args = {
                "marker": current_marker,
                "alpha": 0.6,
                "s": 20,  # Keep user's marker size for 3D
                "label": label_text,
                # Removed vmin and vmax
            }
            if c_val_scatter is not None:
                scatter_args["c"] = c_val_scatter
                scatter_args["cmap"] = cmap_val_scatter

            scatter = ax.scatter(
                transformed[patch_mask, 0],  # PC1
                transformed[patch_mask, 1],  # PC2
                transformed[patch_mask, 2],  # PC3
                **scatter_args,
            )
    else:  # No patch_type
        c_val_scatter_else = None
        if cluster_ids is not None:
            c_val_scatter_else = cluster_ids_for_coloring

        scatter_args_else = {
            "marker": "o",
            "alpha": 0.6,
            "s": 20,  # Keep user's marker size for 3D
            # Removed vmin and vmax
        }
        if c_val_scatter_else is not None:
            scatter_args_else["c"] = c_val_scatter_else
            scatter_args_else["cmap"] = cmap

        scatter = ax.scatter(
            transformed[:, 0],  # PC1
            transformed[:, 1],  # PC2
            transformed[:, 2],  # PC3
            **scatter_args_else,
        )

    # Add labels and title with adjusted padding
    ax.set_xlabel("1st Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_ylabel("2nd Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_zlabel(
        "3rd Latent Dim.", labelpad=-17, rotation=90, fontsize=8
    )  # Increased labelpad and rotated label
    ax.set_title(title, y=0.95)

    # Add colorbar with discrete levels
    if cluster_ids is not None:  # Simplified condition
        try:
            temp_scatter_for_colorbar = ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                c=cluster_ids_for_coloring,  # Use 0-indexed IDs
                cmap=cmap,
                s=0,
                alpha=0,
            )
            # if len(original_unique_ids) > 1 or (
            #     len(original_unique_ids) == 1 and original_unique_ids[0] != 0
            # ):
            #     colorbar = plt.colorbar(
            #         temp_scatter_for_colorbar, ticks=original_unique_ids, shrink=0.4
            #     )
            #     colorbar.set_label(cluster_label)
        except Exception:
            pass  # Silently pass

    if patch_type is not None:
        # Create custom legend handles for black markers
        legend_handles_3d = []
        unique_patches_for_legend_3d = np.unique(np.array(patch_type))
        for i, patch_val in enumerate(unique_patches_for_legend_3d):
            legend_handles_3d.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_shapes[i % len(marker_shapes)],
                    color="black",
                    label=str(patch_val),
                    linestyle="None",
                    markersize=10,
                )
            )
        if legend_handles_3d:
            ax.legend(
                handles=legend_handles_3d,
                title="Patch Types",
                loc="best",
                bbox_to_anchor=(0.9, 0.5),
            )

    # Adjust the view angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Adjust layout with specific spacing - remove specific adjustments for
    # external legend
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)
    fig.tight_layout()
    return fig, ax
