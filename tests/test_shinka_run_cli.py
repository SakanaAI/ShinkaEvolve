from __future__ import annotations

from pathlib import Path

import pytest

import shinka.cli.run as cli_run


def _make_task_dir(tmp_path: Path, *, include_evaluate: bool = True) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    if include_evaluate:
        (task_dir / "evaluate.py").write_text(
            "def main(program_path: str, results_dir: str):\n    pass\n",
            encoding="utf-8",
        )
    (task_dir / "initial.py").write_text(
        "# EVOLVE-BLOCK-START\ndef run():\n    return 0\n# EVOLVE-BLOCK-END\n",
        encoding="utf-8",
    )
    return task_dir


class _DummyRunner:
    last_kwargs = None
    run_calls = 0

    def __init__(self, **kwargs):
        _DummyRunner.last_kwargs = kwargs

    def run(self):
        _DummyRunner.run_calls += 1


def _reset_dummy_runner() -> None:
    _DummyRunner.last_kwargs = None
    _DummyRunner.run_calls = 0


def test_shinka_run_help_is_detailed(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(["--help"])
    assert exc_info.value.code == 0

    help_output = capsys.readouterr().out
    assert "Task directory contract" in help_output
    assert "initial.<ext>" in help_output
    assert "--set NS.FIELD=VALUE" in help_output
    assert "--config-fname" in help_output
    assert "unknown namespace/field: non-zero exit" in help_output
    assert "--results_dir always sets evo.results_dir" in help_output


def test_shinka_run_happy_path_with_authoritative_overrides(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    exit_code = cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "7",
            "--set",
            "evo.results_dir=should_not_win",
            "--set",
            "evo.num_generations=999",
            "--set",
            "db.num_islands=2",
            "--set",
            "job.time=00:03:00",
        ]
    )

    assert exit_code == 0
    assert _DummyRunner.run_calls == 1
    assert _DummyRunner.last_kwargs is not None

    evo_config = _DummyRunner.last_kwargs["evo_config"]
    db_config = _DummyRunner.last_kwargs["db_config"]
    job_config = _DummyRunner.last_kwargs["job_config"]
    init_program_str = _DummyRunner.last_kwargs["init_program_str"]
    evaluate_str = _DummyRunner.last_kwargs["evaluate_str"]

    assert evo_config.results_dir == str(results_dir.resolve())
    assert evo_config.num_generations == 7
    assert evo_config.task_sys_msg is not None
    assert evo_config.patch_types == ["diff", "full", "cross"]
    assert evo_config.patch_type_probs == [0.6, 0.3, 0.1]
    assert evo_config.max_patch_attempts == 1
    assert evo_config.llm_models == [
        "gpt-5-mini",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
    ]
    assert evo_config.llm_dynamic_selection == "ucb"
    assert evo_config.llm_dynamic_selection_kwargs == {"cost_aware_coef": 0.5}
    assert evo_config.llm_kwargs == {
        "temperatures": [0.0, 0.5, 1.0],
        "max_tokens": 16384,
    }
    assert evo_config.meta_rec_interval == 10
    assert evo_config.embedding_model == "text-embedding-3-small"
    assert evo_config.code_embed_sim_threshold == pytest.approx(0.99)
    assert db_config.num_islands == 2
    assert db_config.archive_size == 40
    assert db_config.num_archive_inspirations == 1
    assert db_config.num_top_k_inspirations == 1
    assert db_config.migration_rate == pytest.approx(0.0)
    assert db_config.parent_selection_strategy == "weighted"
    assert job_config.time == "00:03:00"
    assert not hasattr(evo_config, "max_proposal_jobs")
    assert not hasattr(evo_config, "max_db_workers")
    assert "def run" in init_program_str
    assert "def main" in evaluate_str


def test_shinka_run_parses_json_overrides(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results_json"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
            "--set",
            'evo.llm_models=["gpt-5-mini","gpt-5-nano"]',
            "--set",
            'job.extra_cmd_args={"seed":42}',
        ]
    )

    evo_config = _DummyRunner.last_kwargs["evo_config"]
    job_config = _DummyRunner.last_kwargs["job_config"]
    assert evo_config.llm_models == ["gpt-5-mini", "gpt-5-nano"]
    assert job_config.extra_cmd_args == {"seed": 42}


def test_shinka_run_parses_activate_script_override(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results_activate_script"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
            "--set",
            "job.activate_script=.venv/bin/activate",
        ]
    )

    job_config = _DummyRunner.last_kwargs["job_config"]
    assert job_config.activate_script == ".venv/bin/activate"


def test_shinka_run_defaults_to_verbose_logging(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results_default_verbose"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
        ]
    )

    assert _DummyRunner.last_kwargs is not None
    assert _DummyRunner.last_kwargs["verbose"] is True
    assert _DummyRunner.last_kwargs["banner_style"] == "minimal"


def test_shinka_run_allows_disabling_verbose_logging(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results_no_verbose"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
            "--no-verbose",
        ]
    )

    assert _DummyRunner.last_kwargs is not None
    assert _DummyRunner.last_kwargs["verbose"] is False
    assert _DummyRunner.last_kwargs["banner_style"] == "minimal"


def test_shinka_run_loads_optional_config_yaml_with_precedence(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    (task_dir / "shinka.yaml").write_text(
        (
            "max_evaluation_jobs: 8\n"
            "max_proposal_jobs: 7\n"
            "max_db_workers: 6\n"
            "verbose: true\n"
            "debug: true\n"
            "db_config:\n"
            "  num_islands: 4\n"
            "job_config:\n"
            "  time: 00:04:00\n"
            "evo_config:\n"
            "  num_generations: 999\n"
            "  results_dir: from_config\n"
            '  llm_models: ["gpt-5-nano"]\n'
        ),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results_config"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--config-fname",
            "shinka.yaml",
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
            "--max-db-workers",
            "9",
            "--set",
            "db.num_islands=2",
            "--set",
            'evo.llm_models=["gpt-5-mini"]',
        ]
    )

    assert _DummyRunner.last_kwargs is not None
    evo_config = _DummyRunner.last_kwargs["evo_config"]
    db_config = _DummyRunner.last_kwargs["db_config"]
    job_config = _DummyRunner.last_kwargs["job_config"]

    assert evo_config.results_dir == str(results_dir.resolve())
    assert evo_config.num_generations == 3
    assert evo_config.llm_models == ["gpt-5-mini"]
    assert db_config.num_islands == 2
    assert job_config.time == "00:04:00"
    assert _DummyRunner.last_kwargs["max_evaluation_jobs"] == 8
    assert _DummyRunner.last_kwargs["max_proposal_jobs"] == 7
    assert _DummyRunner.last_kwargs["max_db_workers"] == 9
    assert _DummyRunner.last_kwargs["verbose"] is True
    assert _DummyRunner.last_kwargs["debug"] is True


def test_shinka_run_respects_config_verbose_false(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    (task_dir / "shinka.yaml").write_text(
        "verbose: false\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results_config_no_verbose"
    monkeypatch.setattr(cli_run, "ShinkaEvolveRunner", _DummyRunner)

    cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--config-fname",
            "shinka.yaml",
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "3",
        ]
    )

    assert _DummyRunner.last_kwargs is not None
    assert _DummyRunner.last_kwargs["verbose"] is False


def test_shinka_run_invalid_config_field_fails(tmp_path):
    task_dir = _make_task_dir(tmp_path)
    (task_dir / "bad.yaml").write_text(
        "evo_config:\n  unknown_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--config-fname",
                "bad.yaml",
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "5",
            ]
        )
    assert exc_info.value.code == 2


def test_shinka_run_rejects_nested_concurrency_config(tmp_path):
    task_dir = _make_task_dir(tmp_path)
    (task_dir / "bad.yaml").write_text(
        ("evo_config:\n  max_proposal_jobs: 3\n  max_db_workers: 2\n"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--config-fname",
                "bad.yaml",
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "5",
            ]
        )
    assert exc_info.value.code == 2


def test_shinka_run_unknown_override_field_fails(tmp_path):
    task_dir = _make_task_dir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "5",
                "--set",
                "evo.unknown_field=1",
            ]
        )
    assert exc_info.value.code == 2


def test_shinka_run_invalid_override_type_fails(tmp_path):
    task_dir = _make_task_dir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "5",
                "--set",
                "evo.max_patch_attempts=not_an_int",
            ]
        )
    assert exc_info.value.code == 2


def test_shinka_run_requires_evaluate_file(tmp_path):
    task_dir = _make_task_dir(tmp_path, include_evaluate=False)
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "5",
            ]
        )
    assert exc_info.value.code == 2


def test_dataclass_defaults_match_shared_baseline():
    evo_config = cli_run.EvolutionConfig()
    db_config = cli_run.DatabaseConfig()
    job_config = cli_run.LocalJobConfig()

    assert evo_config.task_sys_msg is not None
    assert evo_config.patch_types == ["diff", "full", "cross"]
    assert evo_config.patch_type_probs == [0.6, 0.3, 0.1]
    assert evo_config.num_generations == 50
    assert evo_config.max_patch_attempts == 1
    assert evo_config.llm_models == [
        "gpt-5-mini",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
    ]
    assert evo_config.llm_dynamic_selection == "ucb"
    assert evo_config.llm_dynamic_selection_kwargs == {"cost_aware_coef": 0.5}
    assert evo_config.llm_kwargs == {
        "temperatures": [0.0, 0.5, 1.0],
        "max_tokens": 16384,
    }
    assert evo_config.meta_rec_interval == 10
    assert evo_config.embedding_model == "text-embedding-3-small"
    assert evo_config.code_embed_sim_threshold == pytest.approx(0.99)
    assert evo_config.enable_controlled_oversubscription is False

    assert db_config.num_islands == 2
    assert db_config.archive_size == 40
    assert db_config.num_archive_inspirations == 1
    assert db_config.num_top_k_inspirations == 1
    assert db_config.migration_interval == 10
    assert db_config.migration_rate == pytest.approx(0.0)
    assert db_config.parent_selection_strategy == "weighted"
    assert db_config.parent_selection_lambda == pytest.approx(10.0)
    assert db_config.enable_dynamic_islands is False

    assert job_config.time is None
    assert job_config.conda_env is None
    assert job_config.activate_script is None
