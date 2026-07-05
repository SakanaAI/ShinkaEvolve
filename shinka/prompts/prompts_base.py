from __future__ import annotations

import json
from typing import Iterable

from shinka.database import Program


BASE_SYSTEM_MSG = """You are an expert programmer.
Your goal is to improve the current repository while preserving correctness.
"""


def perf_str(combined_score: float | None, public_metrics: dict | None) -> str:
    """Format compact performance metrics for prompts."""
    metrics = public_metrics or {}
    parts = [f"combined_score={combined_score if combined_score is not None else 0.0}"]
    if metrics:
        parts.append(f"public_metrics={json.dumps(metrics, sort_keys=True)}")
    return "; ".join(parts)


def format_text_feedback_section(text_feedback) -> str:
    if not text_feedback:
        return ""
    if isinstance(text_feedback, list):
        text_feedback = "\n".join(str(item) for item in text_feedback)
    return f"\nText feedback:\n{text_feedback}"


def construct_individual_program_msg(
    program: Program,
    *,
    language: str = "python",
    include_text_feedback: bool = False,
) -> str:
    """Render one Program as prompt context."""
    summary = program.repo_summary or program.code or "No summary recorded."
    sections = [
        f"Repository individual ID: {program.id}",
        f"Generation: {program.generation}",
        f"Language: {language}",
        f"Correct: {program.correct}",
        f"Performance: {perf_str(program.combined_score, program.public_metrics)}",
        f"Repository summary:\n{summary}",
    ]
    if include_text_feedback:
        feedback = format_text_feedback_section(program.text_feedback)
        if feedback:
            sections.append(feedback)
    return "\n".join(sections)


def construct_eval_history_msg(
    programs: Iterable[Program],
    *,
    language: str = "python",
    include_text_feedback: bool = False,
    correct: bool | None = None,
) -> str:
    """Render inspiration repository individuals for mutation prompts."""
    rendered = []
    for idx, program in enumerate(programs, start=1):
        if correct is not None and bool(program.correct) != correct:
            continue
        rendered.append(
            f"## Inspiration {idx}\n"
            + construct_individual_program_msg(
                program,
                language=language,
                include_text_feedback=include_text_feedback,
            )
        )
    if not rendered:
        return ""
    return "# Prior Repository Individual Context\n\n" + "\n\n".join(rendered)
