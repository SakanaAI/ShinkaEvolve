import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.wandb_logging import (
    PROGRAM_TABLE_COLUMNS,
    ShinkaWandbLogger,
    build_program_log_payload,
    compute_database_stats,
    program_table_row,
    resolve_logging_methods,
)


@dataclass
class LoggingConfig:
    logging_methods: list[str] = field(default_factory=lambda: ["webui"])
    enable_webui_logging: bool | None = None
    enable_wandb_logging: bool | None = None


def _make_db(tmp_path):
    db = ProgramDatabase(DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")))
    first = Program(
        id="p0",
        code="print(0)",
        generation=0,
        correct=True,
        combined_score=1.0,
        public_metrics={"score": 1.0, "nested": {"accuracy": 0.5}},
        private_metrics={"hidden": 2.0},
        metadata={
            "api_costs": 0.10,
            "embed_cost": 0.01,
            "novelty_cost": 0.02,
            "meta_cost": 0.03,
            "patch_type": "init",
            "patch_name": "initial",
            "pipeline_started_at": 10.0,
            "postprocess_finished_at": 12.0,
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
            "pipeline_started_at": 13.0,
            "postprocess_finished_at": 16.0,
            "llm_result": {"model": "fallback-model"},
        },
    )
    db.add(first, defer_maintenance=True)
    db.add(second, defer_maintenance=True)
    return db, first, second


def test_resolve_logging_methods_supports_list_and_bool_overrides():
    assert resolve_logging_methods(LoggingConfig()) == {"webui"}
    assert resolve_logging_methods(
        LoggingConfig(logging_methods=["webui"], enable_wandb_logging=True)
    ) == {"webui", "wandb"}
    assert resolve_logging_methods(
        LoggingConfig(
            logging_methods=["webui", "wandb"],
            enable_webui_logging=False,
        )
    ) == {"wandb"}


def test_resolve_logging_methods_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown logging method"):
        resolve_logging_methods(LoggingConfig(logging_methods=["webui", "tensorboard"]))


def test_compute_database_stats_matches_run_aggregate_fields(tmp_path):
    db, _, _ = _make_db(tmp_path)

    stats = compute_database_stats(db)

    assert stats["program_count"] == 2
    assert stats["generation_count"] == 2
    assert stats["correct_count"] == 1
    assert stats["best_score"] == 1.0
    assert stats["best_generation"] == 0
    assert stats["max_generation"] == 1
    assert stats["gens_since_improvement"] == 1
    assert stats["total_cost"] == pytest.approx(0.45)
    assert stats["total_runtime_seconds"] == pytest.approx(6.0)
    assert stats["patch_type_counts"] == {"init": 1, "diff": 1}


def test_program_payload_logs_public_private_metadata_and_costs(tmp_path):
    db, _, second = _make_db(tmp_path)

    payload = build_program_log_payload(second, db)

    assert payload["generation"] == 1
    assert payload["program/combined_score"] == 0.25
    assert payload["program/model_name"] == "fallback-model"
    assert payload["public_metrics/score"] == 0.25
    assert payload["metadata/patch_type"] == "diff"
    assert payload["cost/total"] == pytest.approx(0.29)
    assert not any(key.startswith("webui/") for key in payload)


def test_program_table_serializes_heterogeneous_nested_metadata():
    first = Program(id="a", generation=1, metadata={"nested": {"a": 1}})
    second = Program(id="b", generation=2, metadata={"nested": [1, 2, 3]})

    first_row = program_table_row(first)
    second_row = program_table_row(second)
    metadata_index = PROGRAM_TABLE_COLUMNS.index("metadata")

    assert isinstance(first_row[metadata_index], str)
    assert isinstance(second_row[metadata_index], str)
    assert '"a": 1' in first_row[metadata_index]


def test_wandb_logger_uses_fake_wandb_without_network(tmp_path, monkeypatch):
    db, first, _ = _make_db(tmp_path)
    plot_dir = tmp_path / "gen_1" / "results" / "plots"
    plot_dir.mkdir(parents=True)
    (plot_dir / "score.png").write_bytes(b"png")
    (tmp_path / "gen_1" / "failure.json").write_text("{}", encoding="utf-8")
    logged_payloads = []
    artifacts = []
    init_kwargs = {}

    class FakeRun:
        name = "fake-run"

        def __init__(self):
            self.defined = []
            self.finished = False

        def define_metric(self, *args, **kwargs):
            self.defined.append((args, kwargs))

        def log(self, payload):
            logged_payloads.append(payload)

        def log_artifact(self, artifact):
            artifacts.append(artifact)

        def finish(self):
            self.finished = True

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    fake_run = FakeRun()
    fake_wandb = ModuleType("wandb")
    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    fake_wandb.init = fake_init
    fake_wandb.Table = FakeTable
    fake_wandb.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = ShinkaWandbLogger(enabled=True)
    logger.start(
        evo_config=SimpleNamespace(
            wandb_project="project",
            wandb_entity=None,
            wandb_group=None,
            wandb_name="name",
            wandb_mode="offline",
            wandb_tags=[],
            wandb_notes=None,
            wandb_dir=None,
            wandb_config={},
            wandb_run_id="stable-run-id",
            wandb_resume="allow",
        ),
        db_config=SimpleNamespace(),
        job_config=SimpleNamespace(),
        results_dir=tmp_path,
    )
    logger.log_program(program=first, db=db)
    logger.log_final(db=db, results_dir=tmp_path, total_proposals_generated=2)
    logger.finish()

    assert fake_run.finished is True
    assert init_kwargs["id"] == "stable-run-id"
    assert init_kwargs["resume"] == "allow"
    assert logged_payloads[0]["program/id"] == "p0"
    assert isinstance(logged_payloads[0]["program/latest"], FakeTable)
    assert logged_payloads[0]["program/latest"].columns == PROGRAM_TABLE_COLUMNS
    final_payload = logged_payloads[1]
    assert final_payload["final/program_count"] == 2
    assert final_payload["final/total_cost"] == pytest.approx(0.45)
    assert not any(key.startswith("webui/") for key in final_payload)
    assert not any(key.startswith("final/webui/") for key in final_payload)
    assert isinstance(final_payload["program/all"], FakeTable)
    assert artifacts
    artifact_names = {name for _, name in artifacts[0].files}
    assert "programs.sqlite" in artifact_names
    assert "gen_1/results/plots/score.png" in artifact_names
    assert "gen_1/failure.json" in artifact_names
