import sys
from types import ModuleType, SimpleNamespace

import pytest

from shinka.core.config import EvolutionConfig
from shinka.database import DatabaseConfig, Program
from shinka.launch.scheduler import LocalJobConfig
from shinka.wandb_logging import (
    EVALUATED_COUNT_METRIC,
    INDIVIDUAL_SCORE_METRIC,
    PROGRAM_TABLE_COLUMNS,
    PROGRAM_TABLE_KEY,
    ShinkaWandbLogger,
    ensure_wandb_run_id,
)
from wandb_test_utils import make_db


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


def test_wandb_history_failure_is_non_fatal_and_retryable(tmp_path, caplog):
    _, first, _ = make_db(tmp_path)

    class FlakyRun:
        def __init__(self):
            self.calls = 0

        def log(self, _payload):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary telemetry failure")

    logger = ShinkaWandbLogger(enabled=True)
    logger._run = FlakyRun()

    logger.log_program(first)
    logger.log_program(first)

    assert logger._run.calls == 2
    assert "Failed to log individual p0 to W&B" in caplog.text


@pytest.mark.parametrize("copy_marker", ["_is_island_copy", "_spawned_island"])
def test_wandb_history_skips_administrative_copies(copy_marker):
    logged_payloads = []
    logger = ShinkaWandbLogger(enabled=True)
    logger._run = SimpleNamespace(log=logged_payloads.append)
    copy = Program(id="copy", code="pass", metadata={copy_marker: True})

    logger.log_program(copy)

    assert logged_payloads == []


def test_final_history_failure_does_not_suppress_summary_or_table(tmp_path, caplog):
    db, _, _ = make_db(tmp_path)
    logged = []

    class Run:
        summary = {}

        def log(self, payload):
            if not logged:
                logged.append("history-failed")
                raise RuntimeError("history")
            logged.append(payload)

    logger = ShinkaWandbLogger(enabled=True)
    logger._run = Run()
    logger._wandb = SimpleNamespace(Table=lambda **kwargs: SimpleNamespace(**kwargs))

    logger.log_final(db=db, total_api_cost=0.4, total_cost=0.5)

    assert logger._run.summary["run/total_api_cost"] == 0.4
    assert PROGRAM_TABLE_KEY in logged[-1]
    assert "final W&B history" in caplog.text


def test_final_summary_failure_does_not_suppress_table(tmp_path, caplog):
    db, _, _ = make_db(tmp_path)
    logged = []

    class BrokenSummary(dict):
        def update(self, values):
            raise RuntimeError("summary")

    logger = ShinkaWandbLogger(enabled=True)
    logger._run = SimpleNamespace(summary=BrokenSummary(), log=logged.append)
    logger._wandb = SimpleNamespace(Table=lambda **kwargs: SimpleNamespace(**kwargs))

    logger.log_final(db=db, total_api_cost=0.4)

    assert PROGRAM_TABLE_KEY in logged[-1]
    assert "final W&B summary" in caplog.text


def test_final_table_construction_failure_keeps_history_and_summary(tmp_path, caplog):
    db, _, _ = make_db(tmp_path)
    logged = []
    summary = {}
    logger = ShinkaWandbLogger(enabled=True)
    logger._run = SimpleNamespace(summary=summary, log=logged.append)
    logger._wandb = SimpleNamespace(
        Table=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("table"))
    )

    logger.log_final(db=db, total_api_cost=0.4)

    assert logged[0]["run/total_api_cost"] == 0.4
    assert summary["run/total_api_cost"] == 0.4
    assert "final W&B table" in caplog.text


def test_final_table_log_failure_is_non_fatal(tmp_path, caplog):
    db, _, _ = make_db(tmp_path)
    summary = {}

    def log(payload):
        if PROGRAM_TABLE_KEY in payload:
            raise RuntimeError("table log")

    logger = ShinkaWandbLogger(enabled=True)
    logger._run = SimpleNamespace(summary=summary, log=log)
    logger._wandb = SimpleNamespace(Table=lambda **kwargs: SimpleNamespace(**kwargs))

    logger.log_final(db=db, total_api_cost=0.4)

    assert summary["run/total_api_cost"] == 0.4
    assert "log final W&B table" in caplog.text


def test_wandb_logger_dry_run_and_resume_use_fake_wandb(tmp_path, monkeypatch):
    db, first, second = make_db(tmp_path)
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

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_run = FakeRun()
    fake_wandb = ModuleType("wandb")

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    fake_wandb.init = fake_init
    fake_wandb.Table = FakeTable
    fake_wandb.Settings = FakeSettings
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    evo_config = EvolutionConfig(
        wandb_project="project",
        wandb_name="name",
        wandb_mode="offline",
        task_sys_msg="private task instructions",
        wandb_config={
            "nested": {
                "api_key": "wandb-extra-secret",
                "apiKey": "camel-api-secret",
                "accessToken": "camel-access-secret",
                "clientSecret": "camel-client-secret",
                "AuthorizationHeader": "camel-authorization-secret",
                "aws_secret_access_key": "aws-secret",
                "Authorization": "Bearer auth-secret",
                "safe_value": 2,
            }
        },
    )
    logger = ShinkaWandbLogger(enabled=True)
    logger.start(
        evo_config=evo_config,
        db_config=SimpleNamespace(),
        job_config=LocalJobConfig(
            extra_cmd_args={"access_token": "job-secret", "workers": 2}
        ),
        results_dir=tmp_path,
    )
    logger.log_program(first)
    logger.log_program(first)
    logger.log_program(second)
    logger.log_population_progress(db=db)
    logger.log_population_progress(db=db)
    logger.log_final(
        db=db,
        total_proposals_generated=2,
        total_cost=0.5,
    )
    logger.finish()

    assert init_kwargs["project"] == "project"
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["save_code"] is False
    assert init_kwargs["settings"].kwargs == {
        "console": "off",
        "disable_git": True,
        "disable_code": True,
        "x_disable_stats": True,
        "x_disable_machine_info": True,
        "x_save_requirements": False,
    }
    assert init_kwargs["id"]
    assert init_kwargs["resume"] == "allow"
    assert (tmp_path / ".wandb_run_id").read_text(encoding="utf-8").strip() == (
        init_kwargs["id"]
    )
    assert "private task instructions" not in repr(init_kwargs["config"])
    assert "wandb-extra-secret" not in repr(init_kwargs["config"])
    assert "camel-api-secret" not in repr(init_kwargs["config"])
    assert "camel-access-secret" not in repr(init_kwargs["config"])
    assert "camel-client-secret" not in repr(init_kwargs["config"])
    assert "camel-authorization-secret" not in repr(init_kwargs["config"])
    assert "aws-secret" not in repr(init_kwargs["config"])
    assert "auth-secret" not in repr(init_kwargs["config"])
    assert "job-secret" not in repr(init_kwargs["config"])
    assert init_kwargs["config"]["nested"]["safe_value"] == 2
    assert "task_sys_msg" not in init_kwargs["config"]["evolution"]
    assert "init_program_path" not in init_kwargs["config"]["evolution"]
    assert "llm_kwargs" not in init_kwargs["config"]["evolution"]
    assert "db_path" not in init_kwargs["config"]["database"]
    assert "extra_cmd_args" not in init_kwargs["config"]["job"]
    assert fake_run.finished is True
    assert [payload[INDIVIDUAL_SCORE_METRIC] for payload in logged_payloads[:2]] == [
        1.0,
        0.25,
    ]
    assert len(logged_payloads) == 5
    assert logged_payloads[2][EVALUATED_COUNT_METRIC] == 2
    assert logged_payloads[3][EVALUATED_COUNT_METRIC] == 2
    assert logged_payloads[3]["cost/embed"] == pytest.approx(0.03)
    assert logged_payloads[3]["cost/total"] == pytest.approx(0.35)
    assert logged_payloads[3]["run/evaluated_count"] == 2
    assert logged_payloads[3]["run/total_cost"] == 0.5
    table = logged_payloads[-1][PROGRAM_TABLE_KEY]
    assert isinstance(table, FakeTable)
    assert table.columns == PROGRAM_TABLE_COLUMNS
    assert len(table.data) == 2
    assert "code" not in table.columns
    assert "embedding" not in table.columns
    assert fake_run.summary["run/best_score"] == 1.0
    assert fake_run.summary["run/evaluated_count"] == 2
    assert fake_run.summary["run/total_cost"] == 0.5
    score_definition = next(
        item for item in fake_run.defined if item[0] == (INDIVIDUAL_SCORE_METRIC,)
    )
    assert score_definition[1] == {}
    population_definition = next(
        item for item in fake_run.defined if item[0] == ("population/count",)
    )
    assert population_definition[1] == {"step_metric": EVALUATED_COUNT_METRIC}

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

    db, first, second = make_db(tmp_path)
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
            total_cost=0.5,
        )
    finally:
        logger.finish()

    integration_warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "shinka.wandb_logging" and record.levelno >= 30
    ]
    assert integration_warnings == []
