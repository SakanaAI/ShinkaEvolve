import shlex
import subprocess
from pathlib import Path

import pytest

from shinka.launch import JobScheduler, SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.launch import slurm
from shinka.launch.slurm_results import render_seed_evaluator

AWKWARD_COMMAND = [
    "python",
    "evaluate.py",
    "--program_path",
    "my runs/gen 1/main.py",
    "--results_dir",
    "my runs/gen 1/results; touch pwned",
    "--note",
    "it's $HOME",
]


def _script_lines(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the batch script handed to sbatch."""
    lines: list[str] = []

    def run(cmd, **kwargs):
        assert cmd[0] == "sbatch"
        lines.extend(Path(cmd[-1]).read_text().splitlines())
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 123\n")

    monkeypatch.setattr(subprocess, "run", run)
    return lines


def test_async_evaluator_command_is_quoted_like_the_seed_command() -> None:
    setup, command, finish = render_seed_evaluator(AWKWARD_COMMAND, None, "job")

    assert (setup, finish) == ("", "exit $?")
    assert shlex.split(command) == AWKWARD_COMMAND


def test_async_conda_script_keeps_arguments_with_spaces_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = _script_lines(monkeypatch)
    config = SlurmCondaJobConfig(
        eval_program_path="my evaluators/evaluate.py",
        conda_env="$HOME/envs/shinka",
        modules=["cuda/12.1"],
        extra_cmd_args={"note": "it's $HOME"},
        numeric_threads_per_job=2,
    )
    scheduler = JobScheduler("slurm_conda", config)
    try:
        job_id = scheduler.submit_async("gen 1/main.py", str(tmp_path / "gen 1"))
    finally:
        scheduler.shutdown()

    assert job_id == "123"
    (command,) = [line for line in lines if line.startswith("python ")]
    assert shlex.split(command) == [
        "python",
        "my evaluators/evaluate.py",
        "--program_path",
        "gen 1/main.py",
        "--results_dir",
        str(tmp_path / "gen 1"),
        "--note",
        "it's $HOME",
    ]
    # An environment given as a path keeps its shell expansion.
    assert "conda activate $HOME/envs/shinka" in lines
    assert "module load cuda/12.1" in lines
    # Plain values are written exactly as before.
    assert "export OMP_NUM_THREADS=2" in lines
    assert "export SHINKA_EVAL_VERBOSE=1" in lines
    assert "export PYTHONIOENCODING=utf-8" in lines


def test_async_docker_script_keeps_arguments_with_spaces_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = _script_lines(monkeypatch)
    config = SlurmDockerJobConfig(
        image_tar_path="/tmp/image.tar", extra_cmd_args={"note": "two words"}
    )
    scheduler = JobScheduler("slurm_docker", config)
    try:
        scheduler.submit_async("main.py", str(tmp_path))
    finally:
        scheduler.shutdown()

    (command,) = [line.strip() for line in lines if "/workspace/evaluate.py" in line]
    assert shlex.split(command)[-2:] == ["--note", "two words"]
    assert 'echo "Launching Docker container..."' in lines
    assert "-e SHINKA_EVAL_VERBOSE=1" in "\n".join(lines)
    assert "-e PYTHONIOENCODING=utf-8" in "\n".join(lines)


def test_env_values_are_quoted_for_the_shell() -> None:
    env = {"PLAIN": "1", "SPACED": "a b", "SUBSTITUTION": "$(id)"}

    assert slurm._render_env_exports(env).splitlines() == [
        "export PLAIN=1",
        "export SPACED='a b'",
        "export SUBSTITUTION='$(id)'",
    ]
    assert slurm._render_env_docker_flags(env) == (
        "-e PLAIN=1 -e SPACED='a b' -e SUBSTITUTION='$(id)'"
    )


def test_local_fallback_command_is_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(
        slurm, "launch_local_subprocess", lambda job_id, cmd, gpus: launched.append(cmd)
    )
    log_dir = tmp_path / "my logs"

    slurm.submit_local_conda(
        log_dir=str(log_dir),
        cmd=AWKWARD_COMMAND,
        time="00:10:00",
        partition="gpu",
        cpus=1,
        gpus=0,
        mem="8G",
        eval_env={"SPACED": "a b"},
    )

    ((shell, flag, script),) = launched
    assert (shell, flag) == ("bash", "-lc")
    segments = [segment.strip() for segment in script.split(";")]
    assert "export SPACED='a b'" in segments
    evaluator = shlex.split(script[script.index("python evaluate.py") :])
    assert evaluator == [
        *AWKWARD_COMMAND,
        ">>",
        f"{log_dir}/job_log.out",
        "2>>",
        f"{log_dir}/job_log.err",
    ]
