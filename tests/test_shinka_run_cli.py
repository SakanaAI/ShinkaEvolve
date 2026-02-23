from __future__ import annotations

from pathlib import Path

import pytest

import shinka.cli.run as cli_run


def _make_task_dir(tmp_path: Path, *, include_evaluate: bool = True) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    if include_evaluate:
        (task_dir / "evaluate.py").write_text(
            "def main(program_path: str, results_dir: str):\n"
            "    pass\n",
            encoding="utf-8",
        )
    (task_dir / "initial.py").write_text(
        "# EVOLVE-BLOCK-START\n"
        "def run():\n"
        "    return 0\n"
        "# EVOLVE-BLOCK-END\n",
        encoding="utf-8",
    )
    return task_dir


class _DummyRunner:
    last_kwargs = None
    run_calls = 0

    def __init__(self, **kwargs):
        _DummyRunner.last_kwargs = kwargs

    async def run(self):
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
    assert "unknown namespace/field: non-zero exit" in help_output
    assert "--results_dir always sets evo.results_dir" in help_output


def test_shinka_run_happy_path_with_authoritative_overrides(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli_run, "AsyncEvolutionRunner", _DummyRunner)

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
    assert db_config.num_islands == 2
    assert job_config.time == "00:03:00"
    assert "def run" in init_program_str
    assert "def main" in evaluate_str


def test_shinka_run_parses_json_overrides(tmp_path, monkeypatch):
    _reset_dummy_runner()
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results_json"
    monkeypatch.setattr(cli_run, "AsyncEvolutionRunner", _DummyRunner)

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
                "evo.max_parallel_jobs=not_an_int",
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
