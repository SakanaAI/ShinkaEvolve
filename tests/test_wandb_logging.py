import sys
from types import ModuleType, SimpleNamespace

import pytest

from shinka.core.config import EvolutionConfig
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.wandb_logging import (
    INDIVIDUAL_SCORE_METRIC,
    PROGRAM_TABLE_COLUMNS,
    PROGRAM_TABLE_KEY,
    ShinkaWandbLogger,
    build_program_log_payload,
    build_run_summary,
    ensure_wandb_run_id,
)


def _make_db(tmp_path):
    db = ProgramDatabase(DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")))
    first = Program(
        id="p0",
        code="print(0)",
        generation=0,
        correct=True,
        combined_score=1.0,
        public_metrics={
            "score": 1.0,
            "nested": {"accuracy": 0.5},
            "bulky_text": "not a chart metric",
        },
        private_metrics={"hidden": 2.0},
        metadata={
            "api_costs": 0.10,
            "embed_cost": 0.01,
            "novelty_cost": 0.02,
            "meta_cost": 0.03,
            "patch_type": "init",
            "pipeline_seconds": 2.0,
            "evaluation_seconds": 1.5,
            "model_name": "test-model",
        },
    )
    second = Program(
        id="p1",
        code="print(1)",
        generation=1,
        parent_id="p0",
        correct=False,
        combined_score=0.25,
        public_metrics={"score": 0.25},
        metadata={
            "api_costs": 0.20,
            "embed_cost": 0.02,
            "novelty_cost": 0.03,
            "meta_cost": 0.04,
            "patch_type": "diff",
            "llm_result": {"model": "fallback-model"},
        },
    )
    db.add(first, defer_maintenance=True)
    db.add(second, defer_maintenance=True)
    return db, first, second


def test_wandb_logging_is_opt_in_and_does_not_configure_webui():
    config = EvolutionConfig()

    assert config.enable_wandb_logging is False
    assert config.wandb_run_id is None
    assert config.wandb_resume == "allow"
    assert not hasattr(config, "enable_webui_logging")


def test_wandb_run_id_is_reused_and_can_be_overridden(tmp_path):
    first_run_id = ensure_wandb_run_id(tmp_path)

    assert first_run_id
    assert ensure_wandb_run_id(tmp_path) == first_run_id
    assert ensure_wandb_run_id(tmp_path, "configured-id") == "configured-id"
    assert (tmp_path / ".wandb_run_id").read_text(encoding="utf-8") == (
        "configured-id\n"
    )


def test_program_payload_logs_one_compact_event_per_individual(tmp_path):
    _, _, second = _make_db(tmp_path)

    payload = build_program_log_payload(second)

    assert payload["generation"] == 1
    assert payload[INDIVIDUAL_SCORE_METRIC] == 0.25
    assert payload["individual/model_name"] == "fallback-model"
    assert "public_metrics/score" not in payload
    assert payload["cost/api"] == pytest.approx(0.20)
    assert "program/combined_score" not in payload
    assert not any(key.startswith("metadata/") for key in payload)
    assert not any("latest" in key for key in payload)


def test_program_payload_skips_bulky_values_and_duplicate_timing(tmp_path):
    _, first, _ = _make_db(tmp_path)

    payload = build_program_log_payload(first)

    assert payload["public_metrics/nested/accuracy"] == 0.5
    assert payload["private_metrics/hidden"] == 2.0
    assert "public_metrics/bulky_text" not in payload
    assert payload["timing/pipeline_seconds"] == 2.0
    assert "metadata/pipeline_seconds" not in payload


def test_run_summary_uses_correct_programs_for_best_score(tmp_path):
    db, _, _ = _make_db(tmp_path)

    summary = build_run_summary(
        db.get_all_programs(),
        total_proposals_generated=2,
        total_api_cost=0.5,
    )

    assert summary["run/program_count"] == 2
    assert summary["run/correct_rate"] == 0.5
    assert summary["run/best_score"] == 1.0
    assert summary["run/max_generation"] == 1
    assert summary["run/total_proposals_generated"] == 2
    assert summary["run/total_api_cost"] == 0.5


def test_invalid_wandb_config_is_non_fatal(tmp_path, monkeypatch, caplog):
    fake_wandb = ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: pytest.fail("wandb.init should not be called")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = ShinkaWandbLogger(enabled=True)

    logger.start(
        evo_config=SimpleNamespace(wandb_config="invalid"),
        db_config=SimpleNamespace(),
        job_config=SimpleNamespace(),
        results_dir=tmp_path,
    )

    assert logger.active is False
    assert "Failed to initialize W&B logging" in caplog.text


def test_wandb_logger_dry_run_and_resume_use_fake_wandb(tmp_path, monkeypatch):
    db, first, second = _make_db(tmp_path)
    logged_payloads = []
    init_kwargs = {}

    class FakeRun:
        def __init__(self):
            self.defined = []
            self.summary = {}
            self.finished = False

        def define_metric(self, *args, **kwargs):
            self.defined.append((args, kwargs))

        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            self.finished = True

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    fake_run = FakeRun()
    fake_wandb = ModuleType("wandb")

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    fake_wandb.init = fake_init
    fake_wandb.Table = FakeTable
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    evo_config = EvolutionConfig(
        wandb_project="project",
        wandb_name="name",
        wandb_mode="offline",
    )
    logger = ShinkaWandbLogger(enabled=True)
    logger.start(
        evo_config=evo_config,
        db_config=SimpleNamespace(),
        job_config=SimpleNamespace(),
        results_dir=tmp_path,
    )
    logger.log_program(first)
    logger.log_program(first)
    logger.log_program(second)
    logger.log_final(
        db=db,
        total_proposals_generated=2,
        total_api_cost=0.5,
    )
    logger.finish()

    assert init_kwargs["project"] == "project"
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["id"]
    assert init_kwargs["resume"] == "allow"
    assert (tmp_path / ".wandb_run_id").read_text(encoding="utf-8").strip() == (
        init_kwargs["id"]
    )
    assert init_kwargs["config"]["evolution"]["wandb_run_id"] == init_kwargs["id"]
    assert fake_run.finished is True
    assert [payload[INDIVIDUAL_SCORE_METRIC] for payload in logged_payloads[:2]] == [
        1.0,
        0.25,
    ]
    assert len(logged_payloads) == 3
    table = logged_payloads[-1][PROGRAM_TABLE_KEY]
    assert isinstance(table, FakeTable)
    assert table.columns == PROGRAM_TABLE_COLUMNS
    assert len(table.data) == 2
    assert "code" not in table.columns
    assert "embedding" not in table.columns
    assert fake_run.summary["run/best_score"] == 1.0
    score_definition = next(
        item for item in fake_run.defined if item[0] == (INDIVIDUAL_SCORE_METRIC,)
    )
    assert score_definition[1] == {"step_metric": "generation"}

    first_run_id = init_kwargs["id"]
    resumed_config = EvolutionConfig(
        wandb_project="project",
        wandb_mode="offline",
        wandb_resume="must",
    )
    resumed_logger = ShinkaWandbLogger(enabled=True)
    resumed_logger.start(
        evo_config=resumed_config,
        db_config=SimpleNamespace(),
        job_config=SimpleNamespace(),
        results_dir=tmp_path,
    )

    assert init_kwargs["id"] == first_run_id
    assert init_kwargs["resume"] == "must"
    assert resumed_config.wandb_run_id == first_run_id
    resumed_logger.finish()


@pytest.mark.requires_secrets
def test_wandb_online_logging_with_authenticated_sdk(tmp_path, monkeypatch, caplog):
    import wandb

    db, first, second = _make_db(tmp_path)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    wandb.login(verify=True)

    config = EvolutionConfig(
        enable_wandb_logging=True,
        wandb_project="shinka-evolve-integration",
        wandb_name=f"authenticated-smoke-{tmp_path.name}",
        wandb_mode="online",
        wandb_tags=["integration-test"],
    )
    logger = ShinkaWandbLogger(enabled=True)
    logger.start(
        evo_config=config,
        db_config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")),
        job_config=SimpleNamespace(),
        results_dir=tmp_path,
    )

    try:
        assert logger.active, caplog.text
        logger.log_program(first)
        logger.log_program(second)
        logger.log_final(
            db=db,
            total_proposals_generated=2,
            total_api_cost=0.5,
        )
    finally:
        logger.finish()

    integration_warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "shinka.wandb_logging" and record.levelno >= 30
    ]
    assert integration_warnings == []
