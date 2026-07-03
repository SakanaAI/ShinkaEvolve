from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Iterable, Optional

SUMMARY_SCHEMA_VERSION = "repo-individual-v1"

# TODO: Simplify the summary schema
REQUIRED_HEADINGS = [
    "# Individual Summary",
    "## Parent",
    "## Core Idea",
    "## Lineage Context",
    "## Changed Files",
    "## Validation Performed",
    "## Performance Hypothesis",
    "## Risks and Followups",
    "## Minimal Snippets",
]


@dataclass
class SummaryValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: Optional[str] = None



def _extract_field(content: str, field_name: str) -> Optional[str]:
    pattern = rf"(?m)^[-*]?\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None


def validate_summary(
    content: str,
    *,
    max_chars: int = 12000,
) -> SummaryValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not content.strip():
        errors.append("summary is empty")
    if len(content) > max_chars:
        errors.append(f"summary exceeds max_chars={max_chars}")

    for heading in REQUIRED_HEADINGS:
        if heading not in content:
            errors.append(f"missing required heading: {heading}")

    schema_version = _extract_field(content, "Schema-Version")
    if schema_version != SUMMARY_SCHEMA_VERSION:
        errors.append(
            f"Schema-Version must be {SUMMARY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    if "```" in content and content.count("```") % 2 != 0:
        errors.append("unbalanced fenced code block")

    if len(content) > max_chars * 0.8:
        warnings.append("summary is close to configured size limit")

    return SummaryValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        schema_version=schema_version,
    )


def _bullet_lines(values: Iterable[str]) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return "- None recorded"
    return "\n".join(f"- {item}" for item in items)


def build_initial_summary(
    *,
    individual_id: str,
    generation: int,
    commit_sha: str,
    changed_files: Iterable[str] = (),
    parent_id: Optional[str] = None,
    parent_commit: Optional[str] = None,
    parent_digest: str = "Root individual.",
    core_idea: str = "Initial repository seed.",
    validation: str = "Initial candidate has not run agent self-validation.",
) -> str:
    """Build a schema-valid fallback summary for seed or degraded candidates."""

    return textwrap.dedent(
        f"""\
        # Individual Summary

        - Schema-Version: {SUMMARY_SCHEMA_VERSION}
        - Individual: {individual_id}
        - Generation: {generation}
        - Commit: {commit_sha}

        ## Parent

        {parent_digest.strip() or "Root individual."}

        ## Core Idea

        {core_idea.strip() or "No core idea recorded."}

        ## Lineage Context

        This summary preserves the inherited context needed for future mutation.

        ## Diff Essence

        Initial or fallback summary. See the persisted git diff for exact changes.

        ## Changed Files

        {_bullet_lines(changed_files)}

        ## Validation Performed

        {validation.strip() or "No validation recorded."}

        ## Performance Hypothesis

        This individual is expected to preserve or improve the repository behavior under the configured evaluator.

        ## Risks and Followups

        - Future agents should verify the evaluator-specific assumptions before making large changes.

        ## Minimal Snippets

        - No minimal snippets recorded.
        """
    ).strip() + "\n"
