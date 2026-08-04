import random
from typing import List, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from shinka.database import Program
from .prompts_base import perf_str


CROSS_SYS_FORMAT = """
You are given multiple code scripts implementing the same algorithm.
You are tasked with generating a new code snippet that combines these code scripts in a way that is more efficient. 
I.e. perform crossover between the code scripts.
Provide the complete new program code.
You MUST respond using a short summary name, description, and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
* Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.
""".rstrip()


CROSS_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Task

Perform a cross-over between the code script above and the one below. Aim to combine the best parts of both code implementations that improves the score.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
""".rstrip()


def _embedding_distance(
    parent_embedding: List[float],
    inspiration_embedding: List[float],
    metric: str,
) -> Optional[float]:
    """Return the configured distance, or None for unusable embeddings."""
    try:
        parent_vector = np.asarray(parent_embedding, dtype=float)
        inspiration_vector = np.asarray(inspiration_embedding, dtype=float)
    except (TypeError, ValueError):
        return None

    if (
        parent_vector.ndim != 1
        or inspiration_vector.ndim != 1
        or parent_vector.shape != inspiration_vector.shape
        or parent_vector.size == 0
        or not np.isfinite(parent_vector).all()
        or not np.isfinite(inspiration_vector).all()
    ):
        return None

    distance = pairwise_distances(
        parent_vector.reshape(1, -1),
        inspiration_vector.reshape(1, -1),
        metric=metric,
    )[0, 0]
    return float(distance) if np.isfinite(distance) else None


def _select_inspiration(
    parent: Optional[Program],
    inspirations: List[Program],
    metric: str,
) -> Program:
    """Select the most distant inspiration, falling back to random sampling."""
    if not inspirations:
        raise ValueError("At least one crossover inspiration is required")

    if parent is not None:
        distances = []
        for index, inspiration in enumerate(inspirations):
            distance = _embedding_distance(
                parent.embedding, inspiration.embedding, metric
            )
            if distance is not None:
                distances.append((distance, index))

        if distances:
            _, selected_index = max(distances, key=lambda item: item[0])
            return inspirations[selected_index]

    return random.choice(inspirations)


def get_cross_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
    *,
    parent: Optional[Program] = None,
    distance_metric: str = "cosine",
) -> str:
    all_inspirations = archive_inspirations + top_k_inspirations

    inspiration = _select_inspiration(parent, all_inspirations, distance_metric)

    crossover_inspiration = "# Crossover Inspiration Programs\n"
    crossover_inspiration += f"```{language}\n{inspiration.code}\n```\n\n"
    crossover_inspiration += f"Performance metrics: {perf_str(inspiration.combined_score, inspiration.public_metrics)}\n\n"

    return crossover_inspiration
