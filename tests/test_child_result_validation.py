"""Child evaluations get the same result validation as the seed."""

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval_integrity_utils import (
    SEED_ID,
    RecordingSelection,
    StaticScheduler,
    children,
    good_results,
    make_job,
    make_runner,
    open_database,
)
from shinka.core.evaluation_results import (
    failed_evaluation,
    validate_evaluation_results,
)


def _results(correct: Any, metrics: Any, **extra: Any) -> dict[str, Any]:
    return {"correct": correct, "metrics": metrics, "stdout_log": "", "stderr_log": "", **extra}


BAD_RESULTS = {
    "metrics-missing": _results(
        {"correct": True},
        {},
        evaluation_error="Metrics file not found at results/metrics.json",
    ),
    "infinite-score": _results({"correct": True}, {"combined_score": math.inf}),
    "nan-score": _results({"correct": True}, {"combined_score": math.nan}),
    "overflowing-score": _results({"correct": True}, {"combined_score": 10**400}),
    "string-correct": _results({"correct": "false"}, {"combined_score": 1.0}),
    "integer-correct": _results({"correct": 1}, {"combined_score": 1.0}),
    "correct-not-object": _results([True], {"combined_score": 1.0}),
    "metrics-not-object": _results({"correct": True}, [1.0]),
    "string-score": _results({"correct": True}, {"combined_score": "1.0"}),
    "boolean-score": _results({"correct": True}, {"combined_score": True}),
    "public-not-object": _results(
        {"correct": True}, {"combined_score": 1.0, "public": [1, 2]}
    ),
}


@pytest.mark.parametrize("results", BAD_RESULTS.values(), ids=BAD_RESULTS.keys())
def test_bad_child_results_are_stored_incorrect_with_a_diagnostic(
    tmp_path: Path, results: dict[str, Any]
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            selection = RecordingSelection()
            runner = make_runner(
                async_db, StaticScheduler(results), llm_selection=selection
            )

            assert await runner._process_single_job_safely(make_job())

            (child,) = children(sync_db)
            assert child.correct == 0
            assert child.combined_score == 0.0
            assert child.metadata["evaluation_failed"] is True
            assert "evaluation failed" in child.text_feedback.lower()
            assert child.text_feedback in child.metadata["stderr_log"]
            # The row must not count as correct in SQL or become the best program.
            correct_ids = {
                row["id"]
                for row in sync_db.cursor.execute(
                    "SELECT id FROM programs WHERE correct = 1"
                )
            }
            assert correct_ids == {SEED_ID}
            assert sync_db.get_best_program().id == SEED_ID
            # A rejected result never reaches the bandit as a reward.
            assert [update["reward"] for update in selection.updates] == [None]

    asyncio.run(run())


def test_incorrect_child_with_non_finite_score_is_stored_with_zero(
    tmp_path: Path,
) -> None:
    results = _results({"correct": False, "error": "diverged"}, {"combined_score": math.nan})

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(results))

            assert await runner._process_single_job_safely(make_job())

            (child,) = children(sync_db)
            assert child.correct == 0
            assert child.combined_score == 0.0
            assert child.text_feedback == "diverged"
            assert "non-finite" in child.metadata["stderr_log"]
            assert "evaluation_failed" not in child.metadata

    asyncio.run(run())


def test_valid_child_results_are_stored_as_reported(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            selection = RecordingSelection()
            runner = make_runner(
                async_db, StaticScheduler(good_results(2.5)), llm_selection=selection
            )

            assert await runner._process_single_job_safely(make_job())

            (child,) = children(sync_db)
            assert child.correct == 1
            assert child.combined_score == 2.5
            assert child.public_metrics == {"acc": 1.0}
            assert "evaluation_failed" not in child.metadata
            assert sync_db.get_best_program().id == child.id
            assert [update["reward"] for update in selection.updates] == [2.5]

    asyncio.run(run())


def test_correct_result_from_a_failed_local_evaluator_is_rejected(
    tmp_path: Path,
) -> None:
    """Result files left behind by a crashed evaluator are not trusted."""

    class Handle(SimpleNamespace):
        def __str__(self) -> str:
            return f"ProcessWithLogging(PID: {self.pid})"

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(good_results()))
            crashed = make_job(job_id=Handle(pid=10, returncode=-9), generation=1)
            clean = make_job(job_id=Handle(pid=11, returncode=0), generation=2)

            assert await runner._process_single_job_safely(crashed)
            assert await runner._process_single_job_safely(clean)

            by_generation = {child.generation: child for child in children(sync_db)}
            assert by_generation[1].correct == 0
            assert "status -9" in by_generation[1].text_feedback
            assert by_generation[2].correct == 1

    asyncio.run(run())


def test_validator_accepts_integer_scores_and_feedback_lists() -> None:
    outcome = validate_evaluation_results(
        _results({"correct": True}, {"combined_score": 3, "text_feedback": ["a", "b"]})
    )

    assert outcome.correct is True
    assert outcome.combined_score == 3.0
    assert type(outcome.combined_score) is float
    assert outcome.text_feedback == "a\nb"
    assert not outcome.failed and outcome.diagnostic is None


def test_validator_keeps_results_of_an_incorrect_program_that_exited_nonzero() -> None:
    outcome = validate_evaluation_results(
        _results({"correct": False, "error": "assert failed"}, {"combined_score": 0.25}),
        exit_status=1,
    )

    assert outcome.correct is False
    assert outcome.combined_score == 0.25
    assert outcome.text_feedback == "assert failed"


def test_failed_evaluation_carries_no_score_or_metrics() -> None:
    outcome = failed_evaluation("boom")

    assert outcome.failed and not outcome.correct
    assert outcome.combined_score == 0.0
    assert outcome.public_metrics == {} and outcome.private_metrics == {}
    assert outcome.text_feedback == outcome.diagnostic == "boom"
