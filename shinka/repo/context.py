from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class RepoContext:
    task_objective: str
    parent_summary: str
    parent_id: Optional[str]
    parent_commit: Optional[str]
    parent_metrics: dict[str, Any] = field(default_factory=dict)
    parent_feedback: str = ""
    archive_summaries: list[str] = field(default_factory=list)
    top_k_summaries: list[str] = field(default_factory=list)
    mutable_paths: list[str] = field(default_factory=list)
    immutable_paths: list[str] = field(default_factory=list)
    summary_filename: str = ".shinka/individual.md"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n\n[truncated for context]\n"


def _render_summary_list(title: str, summaries: Iterable[str], max_chars: int) -> str:
    rendered = [f"## {title}"]
    found = False
    for index, summary in enumerate(summaries, start=1):
        found = True
        rendered.append(f"### Inspiration {index}\n\n{_truncate(summary, max_chars)}")
    if not found:
        rendered.append("None selected.")
    return "\n\n".join(rendered)


def render_repo_context(context: RepoContext, *, max_summary_chars: int = 6000) -> str:
    mutable_paths = "\n".join(f"- {path}" for path in context.mutable_paths) or "- Any path not marked immutable"
    immutable_paths = "\n".join(f"- {path}" for path in context.immutable_paths) or "- None configured"
    parent_metrics = "\n".join(
        f"- {key}: {value}" for key, value in context.parent_metrics.items()
    ) or "- No parent metrics recorded"

    return textwrap.dedent(
        f"""\
        ## Objective

        {context.task_objective or "Improve the repository under the configured evaluator."}

        ## Parent Metrics

        {parent_metrics}

        ## Parent Feedback

        {context.parent_feedback or "No text feedback recorded."}

        ## Parent Summary

        {_truncate(context.parent_summary or "No parent summary recorded.", max_summary_chars)}

        {_render_summary_list("Archive Inspiration Summaries", context.archive_summaries, max_summary_chars)}

        {_render_summary_list("Top-K Inspiration Summaries", context.top_k_summaries, max_summary_chars)}

        ## Mutable Paths

        {mutable_paths}

        ## Immutable Paths

        {immutable_paths}

        ## Acceptance Criteria

        Run the cheapest validation or testing you judge is most useful before
        returning. Prefer fast compile, import, lint, or smoke checks over
        expensive end-to-end runs, and record what you validated plus any gaps
        in the summary.

        ## Required Output

        - Edit the worktree directly.
        - Do not edit immutable paths.
        - Run affordable validation and fix failures within your retry budget.
        - Write `{context.summary_filename}` using the required Shinka individual summary schema.
        - The summary must preserve the essential lineage context from the parent.
        """ # TODO: determine how to send sequential messages to the agent
    ).strip() + "\n"
