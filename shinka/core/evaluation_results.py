"""Validation shared by every path that turns evaluator output into a program row."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EvaluationOutcome:
    """What one evaluation contributes to its program row."""

    correct: bool
    combined_score: float
    public_metrics: dict[str, Any] = field(default_factory=dict)
    private_metrics: dict[str, Any] = field(default_factory=dict)
    text_feedback: str = ""
    # Set when the row needs an explanation: always for a failed evaluation, and
    # for an accepted one whose score had to be replaced.
    diagnostic: Optional[str] = None
    failed: bool = False


def failed_evaluation(diagnostic: str) -> EvaluationOutcome:
    """Return the outcome stored when an evaluation's results cannot be trusted."""
    return EvaluationOutcome(
        correct=False,
        combined_score=0.0,
        text_feedback=diagnostic,
        diagnostic=diagnostic,
        failed=True,
    )


def validate_evaluation_results(
    results: Mapping[str, Any],
    *,
    subject: str = "program",
    exit_status: Optional[int] = None,
) -> EvaluationOutcome:
    """Check loaded evaluator results and return what may be stored.

    Raises ValueError when the results cannot back a program row. A program only
    counts as correct with a boolean flag, a finite numeric score and, when the
    exit status is known, an evaluator that exited cleanly.
    """
    if results.get("evaluation_error"):
        raise ValueError(results["evaluation_error"])
    correctness = results.get("correct")
    if not isinstance(correctness, dict) or not isinstance(
        correctness.get("correct"), bool
    ):
        raise ValueError("correct.json must contain a boolean 'correct' field")
    correct = correctness["correct"]
    if correct and exit_status:
        raise ValueError(
            f"Evaluator exited with status {exit_status} after reporting a correct "
            f"{subject}"
        )

    metrics = results.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a JSON object")
    combined_score = metrics.get("combined_score", 0.0)
    if type(combined_score) not in (int, float):
        raise ValueError("metrics.json 'combined_score' must be numeric")
    try:
        combined_score = float(combined_score)
    except OverflowError:  # an integer too large for a float
        combined_score = math.inf if combined_score > 0 else -math.inf
    diagnostic = None
    if not math.isfinite(combined_score):
        if correct:
            raise ValueError(
                f"metrics.json 'combined_score' must be finite for a correct {subject}"
            )
        diagnostic = (
            f"Incorrect {subject} reported a non-finite combined_score "
            f"({combined_score}). Using 0.0."
        )
        combined_score = 0.0

    feedback = metrics.get("text_feedback")
    if not correct and not feedback:
        feedback = correctness.get("error")
    if isinstance(feedback, list):
        text_feedback = "\n".join(str(item) for item in feedback)
    else:
        text_feedback = "" if feedback is None else str(feedback)

    return EvaluationOutcome(
        correct=correct,
        combined_score=combined_score,
        public_metrics=_metric_group(metrics, "public"),
        private_metrics=_metric_group(metrics, "private"),
        text_feedback=text_feedback,
        diagnostic=diagnostic,
    )


def _metric_group(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    group = metrics.get(name)
    if group is None:
        return {}
    if not isinstance(group, dict):
        raise ValueError(f"metrics.json '{name}' must be a JSON object")
    return group
