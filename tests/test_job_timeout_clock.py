from shinka.core.async_runner import evaluation_clock_start


def test_timeout_clock_starts_at_evaluation_not_proposal():
    # A 14-minute proposal followed by an evaluation that has just started:
    # a 5-minute job.time must leave the evaluation its full budget.
    proposal_started_at = 1_000.0
    evaluation_submitted_at = proposal_started_at + 14 * 60
    evaluation_started_at = evaluation_submitted_at + 1.0
    start = evaluation_clock_start(
        proposal_started_at, evaluation_submitted_at, evaluation_started_at
    )
    assert start == evaluation_started_at
    assert (evaluation_started_at + 10.0) - start < 5 * 60


def test_timeout_clock_falls_back_to_submission_then_proposal():
    assert evaluation_clock_start(1.0, 2.0, None) == 2.0
    assert evaluation_clock_start(1.0, None, None) == 1.0
