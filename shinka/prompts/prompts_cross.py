import random
from typing import List

from shinka.database import Program
from .prompts_base import perf_str


CROSS_SYS_FORMAT = """
You are given multiple repository individuals for the same task.
Combine useful ideas from these individuals in a way that is more efficient.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Make sure your changes maintain the evaluator-facing behavior expected by the original repository while improving the internal implementation.
* Make sure the repository still runs after your changes.
""".rstrip()


CROSS_ITER_MSG = """# Current repository individual

Here is the current repository summary:
{code_content}

Here are the performance metrics of the repository individual:

{performance_metrics}{text_feedback_section}

# Task

Perform a cross-over between the current repository individual and the inspiration below. Aim to combine the best parts of both implementations to improve the score.

IMPORTANT: Make sure your changes maintain the evaluator-facing behavior expected by the original repository while improving the internal implementation.
""".rstrip()


def get_cross_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
) -> str:
    all_inspirations = archive_inspirations + top_k_inspirations

    # TODO(RobertTLange): Compute embedding distance between all inspirations and parent - max?! for more diversity

    # Sample a random inspiration
    inspiration = random.choice(all_inspirations)

    inspiration_summary = inspiration.repo_summary or inspiration.code or "No summary recorded."
    crossover_inspiration = "# Crossover Inspiration Repository Individual\n"
    crossover_inspiration += f"{inspiration_summary}\n\n"
    crossover_inspiration += f"Performance metrics: {perf_str(inspiration.combined_score, inspiration.public_metrics)}\n\n"

    return crossover_inspiration
