from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from shinka.launch import JobScheduler, LocalJobConfig, SlurmCondaJobConfig
from shinka.launch.scheduler import SlurmEnvJobConfig
from shinka.launch.slurm import submit_conda, submit_local_conda


def test_slurm_env_config_rejects_conda_and_activate_script() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SlurmEnvJobConfig(conda_env="shinka", activate_script=".venv/bin/activate")


def test_local_job_config_rejects_conda_and_activate_script() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LocalJobConfig(conda_env="shinka", activate_script=".venv/bin/activate")


def test_submit_conda_sources_activate_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "sbatch"
        script_path = Path(cmd[1])
        captured["script"] = script_path.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Submitted batch job 123\n",
            stderr="",
        )

    monkeypatch.setattr("shinka.launch.slurm.subprocess.run", fake_run)

    job_id = submit_conda(
        log_dir=str(tmp_path / "logs"),
        cmd=["python", "evaluate.py"],
        time="00:10:00",
        partition="gpu",
        cpus=1,
        gpus=0,
        mem="8G",
        activate_script=".venv/bin/activate",
    )

    assert job_id == "123"
    assert 'source ".venv/bin/activate"' in captured["script"]
    assert "conda activate" not in captured["script"]


def test_submit_local_conda_sources_activate_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_cmd: list[str] = []
    captured_gpus = -1

    def fake_launch(job_id: str, cmd: list[str], gpus: int) -> None:
        assert job_id.startswith("local-conda-")
        captured_cmd.extend(cmd)
        nonlocal captured_gpus
        captured_gpus = gpus

    monkeypatch.setattr("shinka.launch.slurm.launch_local_subprocess", fake_launch)

    job_id = submit_local_conda(
        log_dir=str(tmp_path / "logs"),
        cmd=["python", "evaluate.py"],
        time="00:10:00",
        partition="gpu",
        cpus=1,
        gpus=0,
        mem="8G",
        activate_script=".venv/bin/activate",
    )

    assert job_id.startswith("local-conda-")
    assert captured_cmd[0:2] == ["bash", "-lc"]
    assert 'source ".venv/bin/activate"' in captured_cmd[2]
    assert "conda activate" not in captured_cmd[2]
    assert captured_gpus == 0


def test_job_scheduler_accepts_slurm_env_alias() -> None:
    scheduler = JobScheduler(
        job_type="slurm_env",
        config=SlurmEnvJobConfig(
            eval_program_path="evaluate.py",
            activate_script=".venv/bin/activate",
        ),
    )

    assert scheduler.monitor is not None


def test_job_scheduler_builds_local_sourced_command() -> None:
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path="evaluate.py",
            activate_script=".venv/bin/activate",
        ),
    )

    cmd = scheduler._build_command("program.py", "results")

    assert cmd[0:2] == ["bash", "-lc"]
    assert 'source ".venv/bin/activate"' in cmd[2]
    assert "evaluate.py" in cmd[2]


def test_job_scheduler_uses_current_python_without_activation() -> None:
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(eval_program_path="evaluate.py"),
    )

    cmd = scheduler._build_command("program.py", "results")

    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["evaluate.py", "--program_path"]


def test_job_scheduler_builds_local_numeric_thread_env_overrides() -> None:
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path="evaluate.py",
            numeric_threads_per_job=3,
        ),
    )

    env_overrides = scheduler._build_local_env_overrides()

    assert env_overrides is not None
    assert env_overrides["OMP_NUM_THREADS"] == "3"
    assert env_overrides["OPENBLAS_NUM_THREADS"] == "3"
    assert env_overrides["MKL_NUM_THREADS"] == "3"
    assert env_overrides["NUMEXPR_NUM_THREADS"] == "3"


def test_slurm_conda_job_config_allows_activate_script_for_back_compat() -> None:
    config = SlurmCondaJobConfig(activate_script=".venv/bin/activate")

    assert config.activate_script == ".venv/bin/activate"
