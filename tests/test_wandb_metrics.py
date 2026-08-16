import pytest

from shinka.database import Program
from shinka.wandb_logging import (
    EVALUATED_COUNT_METRIC,
    INDIVIDUAL_SCORE_METRIC,
    MAX_METRIC_KEY_LENGTH,
    MAX_PUBLIC_METRICS,
    PROGRAM_TABLE_COLUMNS,
    build_population_progress_payload,
    build_program_log_payload,
    build_run_summary,
    program_table_row,
)
from wandb_test_utils import make_db


def test_program_payload_logs_one_compact_event_per_individual(tmp_path):
    _, _, second = make_db(tmp_path)

    payload = build_program_log_payload(second)

    assert payload["generation"] == 1
    assert payload[INDIVIDUAL_SCORE_METRIC] == 0.25
    assert payload["individual/model_name"] == "fallback-model"
    assert payload["individual/is_copy"] is False
    assert "public_metrics/score" not in payload
    assert payload["individual/cost/api"] == pytest.approx(0.20)
    assert not any(key.startswith("headless/") for key in payload)
    assert "cost/api" not in payload
    assert "program/combined_score" not in payload
    assert not any(key.startswith("metadata/") for key in payload)
    assert not any("latest" in key for key in payload)
    assert "print(1)" not in repr(payload)
    assert "[1.0, 2.0]" not in repr(payload)


def test_program_payload_skips_bulky_values_and_duplicate_timing(tmp_path):
    _, first, _ = make_db(tmp_path)

    payload = build_program_log_payload(first)

    assert payload["public_metrics/nested/accuracy"] == 0.5
    assert "private_metrics/hidden" not in payload
    assert "public_metrics/bulky_text" not in payload
    assert payload["individual/timing/pipeline_seconds"] == 2.0
    assert payload["headless/usage_status"] == "reported"
    assert payload["headless/usage_unknown"] is False
    assert payload["headless/pricing_status"] == "missing"
    assert payload["headless/pricing_unknown"] is True
    assert "headless/cost_basis" not in payload
    assert "individual/cost/api" not in payload
    assert "metadata/pipeline_seconds" not in payload
    assert "must-not-leak" not in repr(payload)


@pytest.mark.parametrize("copy_marker", ["_is_island_copy", "_spawned_island"])
def test_population_progress_counts_copies_but_excludes_their_costs(
    tmp_path, copy_marker
):
    db, _, _ = make_db(tmp_path)
    copy = Program(
        id="copy-1",
        code="# Copy\n",
        generation=0,
        correct=True,
        combined_score=1.0,
        island_idx=1,
        metadata={copy_marker: True, "embed_cost": 9.0},
    )
    db.add(copy, defer_maintenance=True)

    payload = build_population_progress_payload(db.get_all_programs())

    assert payload[EVALUATED_COUNT_METRIC] == 2
    assert payload["population/count"] == 3
    assert payload["population/correct_count"] == 2
    assert payload["population/best_score"] == 1.0
    assert payload["cost/embed"] == pytest.approx(0.03)
    assert payload["cost/total"] == pytest.approx(0.35)
    assert payload["island/0/evaluated_count"] == 2
    assert payload["island/1/evaluated_count"] == 0
    assert payload["island/1/count"] == 1
    assert program_table_row(copy)[PROGRAM_TABLE_COLUMNS.index("cost")] == 0.0


def test_actual_headless_unknown_pricing_suppresses_only_api_cost():
    program = Program(
        id="headless",
        code="pass",
        metadata={
            "model_name": "headless/codex",
            "headless_pricing_unknown": True,
            "api_costs": 12.0,
            "embed_cost": 0.2,
        },
    )

    individual = build_program_log_payload(program)
    population = build_population_progress_payload([program])

    assert "individual/cost/api" not in individual
    assert individual["individual/cost/embed"] == pytest.approx(0.2)
    assert population["cost/api"] == 0.0
    assert population["cost/embed"] == pytest.approx(0.2)
    assert population["cost/pricing_unknown_count"] == 1
    assert program_table_row(program)[PROGRAM_TABLE_COLUMNS.index("cost")] == 0.2


def test_stale_headless_prompt_path_does_not_suppress_api_cost():
    program = Program(
        id="headless-path",
        code="pass",
        metadata={
            "model_name": "codex",
            "headless_prompt_path": "/redacted/prompt.md",
            "headless_pricing_unknown": True,
            "api_costs": 12.0,
        },
    )

    payload = build_program_log_payload(program)

    assert payload["individual/cost/api"] == 12.0
    assert "/redacted/prompt.md" not in repr(payload)


def test_program_payload_bounds_public_metrics_and_omits_private_metrics():
    public_metrics = {f"metric_{idx}": idx for idx in range(MAX_PUBLIC_METRICS + 10)}
    public_metrics["x" * (MAX_METRIC_KEY_LENGTH + 1)] = 1.0
    public_metrics["aws_secret_access_key"] = 2.0
    program = Program(
        id="bounded",
        code="pass",
        public_metrics=public_metrics,
        private_metrics={"hidden_score": 99.0},
    )

    payload = build_program_log_payload(program)
    logged_public = {
        key: value
        for key, value in payload.items()
        if key.startswith("public_metrics/")
    }

    assert len(logged_public) == MAX_PUBLIC_METRICS
    assert all(len(key) <= MAX_METRIC_KEY_LENGTH for key in logged_public)
    assert not any(key.startswith("private_metrics/") for key in payload)
    assert "aws_secret_access_key" not in repr(payload)


@pytest.mark.parametrize(
    "sensitive_key",
    ["apiKey", "accessToken", "clientSecret", "AuthorizationHeader"],
)
def test_program_payload_omits_compound_sensitive_metric_keys(sensitive_key):
    program = Program(
        id="sensitive-metric",
        code="pass",
        public_metrics={sensitive_key: 123.0, "safeScore": 0.5},
    )

    payload = build_program_log_payload(program)

    assert sensitive_key not in repr(payload)
    assert payload["public_metrics/safeScore"] == 0.5


def test_run_summary_uses_correct_programs_for_best_score(tmp_path):
    db, _, _ = make_db(tmp_path)

    summary = build_run_summary(
        db.get_all_programs(),
        total_proposals_generated=2,
        total_cost=0.5,
    )

    assert summary["run/program_count"] == 2
    assert summary["run/correct_rate"] == 0.5
    assert summary["run/best_score"] == 1.0
    assert summary["run/max_generation"] == 1
    assert summary["run/total_proposals_generated"] == 2
    assert summary["run/evaluated_count"] == 2
    assert summary["run/total_cost"] == 0.5


def test_run_summary_preserves_legacy_api_cost_and_prefers_explicit_total(tmp_path):
    db, _, _ = make_db(tmp_path)

    explicit = build_run_summary(
        db.get_all_programs(), total_api_cost=0.4, total_cost=0.7
    )
    legacy_only = build_run_summary(db.get_all_programs(), total_api_cost=0.4)

    assert explicit["run/total_api_cost"] == 0.4
    assert explicit["run/total_cost"] == 0.7
    assert legacy_only["run/total_api_cost"] == 0.4
    assert legacy_only["run/total_cost"] == 0.4
