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


def failed_evaluation(
    diagnostic: str, results: Optional[Mapping[str, Any]] = None
) -> EvaluationOutcome:
    """Return the outcome stored when an evaluation's results cannot be trusted.

    Whatever the evaluator said about the program is kept: it is what the next
    proposal has to go on.
    """
    feedback = _evaluator_feedback(results or {})
    return EvaluationOutcome(
        correct=False,
        combined_score=0.0,
        text_feedback=(
            f"{diagnostic}\n\nEvaluator feedback:\n{feedback}"
            if feedback
            else diagnostic
        ),
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
    counts as correct with a boolean flag, usable metrics with a finite numeric
    score and, when the exit status is known, an evaluator that exited cleanly.
    An incorrect program is stored whatever its metrics look like: the unusable
    ones are replaced by defaults and reported.
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

    problems: list[str] = []
    metrics = results.get("metrics", {})
    if not isinstance(metrics, dict):
        problems.append("is not a JSON object")
        metrics = {}
    combined_score = _combined_score(metrics.get("combined_score", 0.0), problems)
    public_metrics = _metric_group(metrics, "public", problems)
    private_metrics = _metric_group(metrics, "private", problems)
    diagnostic = None
    if problems:
        complaint = "metrics.json " + " and ".join(problems)
        if correct:
            raise ValueError(complaint)
        diagnostic = f"Incorrect {subject}: {complaint}. Using defaults instead."

    return EvaluationOutcome(
        correct=correct,
        combined_score=combined_score,
        public_metrics=public_metrics,
        private_metrics=private_metrics,
        text_feedback=(
            _as_text(metrics.get("text_feedback"))
            if correct
            else _evaluator_feedback(results)
        ),
        diagnostic=diagnostic,
    )


def _combined_score(value: Any, problems: list[str]) -> float:
    if type(value) not in (int, float):
        problems.append(f"has a non-numeric 'combined_score' ({type(value).__name__})")
        return 0.0
    try:
        score = float(value)
    except OverflowError:  # an integer too large for a float
        score = math.inf if value > 0 else -math.inf
    if not math.isfinite(score):
        problems.append(f"has a non-finite 'combined_score' ({score})")
        return 0.0
    return score


def _metric_group(
    metrics: dict[str, Any], name: str, problems: list[str]
) -> dict[str, Any]:
    group = metrics.get(name)
    if group is None:
        return {}
    if not isinstance(group, dict):
        problems.append(f"has a '{name}' that is not a JSON object")
        return {}
    return group


def _evaluator_feedback(results: Mapping[str, Any]) -> str:
    """Return what the evaluator said about a program that is not stored correct."""
    metrics, correctness = results.get("metrics"), results.get("correct")
    feedback = metrics.get("text_feedback") if isinstance(metrics, dict) else None
    if not feedback and isinstance(correctness, dict):
        feedback = correctness.get("error")
    return _as_text(feedback)


def _as_text(feedback: Any) -> str:
    if isinstance(feedback, list):
        return "\n".join(str(item) for item in feedback)
    return "" if feedback is None else str(feedback)
