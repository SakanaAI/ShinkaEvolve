import asyncio
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.llm import AsyncLLMClient


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
@pytest.mark.parametrize("restart_output", ["empty", "fresh", "removed"])
def test_slurm_requeue_only_uses_current_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    restart_output: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "9")
    query = AsyncMock(side_effect=AssertionError("seed run must stay offline"))
    monkeypatch.setattr(AsyncLLMClient, "query", query)
    _install_batch_stubs(tmp_path, monkeypatch)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        dedent(f"""\
        import json
        import os
        from pathlib import Path
        import sys

        output = Path(sys.argv[sys.argv.index("--results_dir") + 1])
        assert sys.argv[sys.argv.index("--label") + 1] == "two words"
        restarted = int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
        Path("restart-path" if restarted else "first-path").write_text(str(output))
        print("restart stderr" if restarted else "first stderr", file=sys.stderr)
        if restarted and {restart_output!r} == "removed":
            output.rmdir()
        if not restarted or {restart_output!r} == "fresh":
            (output / "correct.json").write_text(json.dumps({{"correct": True}}))
            (output / "metrics.json").write_text(json.dumps({{
                "combined_score": 23 if restarted else 17
            }}))
        raise SystemExit(0 if restarted or {restart_output!r} == "removed" else 23)
        """)
    )
    image = tmp_path / "image.tar"
    image.touch()
    job_config = (
        SlurmDockerJobConfig(
            eval_program_path=str(evaluator),
            image_tar_path=str(image),
            extra_cmd_args={"label": "two words"},
        )
        if backend == "slurm_docker"
        else SlurmCondaJobConfig(
            eval_program_path=str(evaluator), extra_cmd_args={"label": "two words"}
        )
    )
    _mock_requeue(monkeypatch, restart_output)
    runner = ShinkaEvolveRunner(
        EvolutionConfig(
            num_generations=1,
            results_dir="run",
            job_type=backend,
            embedding_model=None,
            llm_models=["gpt-5-mini"],
            llm_dynamic_selection=None,
        ),
        job_config,
        DatabaseConfig(num_islands=1),
        init_program_str="pass\n",
        verbose=False,
    )

    asyncio.run(asyncio.wait_for(runner.run_async(), 12))

    with sqlite3.connect("run/programs.sqlite") as db:
        [(correct, score, metadata)] = db.execute(
            "SELECT correct, combined_score, metadata FROM programs WHERE generation = 0"
        ).fetchall()
        expected = (1, 23.0) if restart_output == "fresh" else (0, 0.0)
        assert (correct, score) == expected, metadata
        assert db.execute("SELECT COUNT(*) FROM attempt_log").fetchone() == (0,)
    assert runner.total_api_cost == 0.0
    query.assert_not_awaited()

    first_path = Path(Path("first-path").read_text())
    restart_path = Path(Path("restart-path").read_text())
    assert restart_path != first_path
    assert json.loads((first_path / "metrics.json").read_text())["combined_score"] == 17
    public = Path("run/gen_0/results")
    assert "restart stderr" in (public / "job_log.err").read_text()
    assert "restart stderr" in json.loads(metadata)["stderr_log"]
    if restart_output == "fresh":
        assert json.loads((public / "metrics.json").read_text())["combined_score"] == 23
    else:
        assert json.loads(metadata)["evaluation_failed"] is True
        assert not (public / "correct.json").exists()
        assert not (public / "metrics.json").exists()


def _install_batch_stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = tmp_path / "module"
    module.write_text("#!/bin/bash\nexit 0\n")
    module.chmod(0o755)
    docker = tmp_path / "docker"
    docker.write_text(
        dedent(f"""\
        #!{sys.executable}
        import os
        import sys
        args = iter(sys.argv[1:])
        action = next(args)
        if action == "load":
            raise SystemExit(0)
        assert action == "run"
        for arg in args:
            if arg == "--rm":
                continue
            if arg == "-e":
                next(args)
                continue
            assert arg == "ubuntu:latest"
            break
        assert next(args) == "python"
        command = [arg.removeprefix("/workspace/") for arg in args]
        os.execv(sys.executable, [sys.executable, *command])
        """)
    )
    docker.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            [str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"]]
        ),
    )


def _mock_requeue(monkeypatch: pytest.MonkeyPatch, restart_output: str) -> None:
    run = subprocess.run
    job_name = ""

    def fake_slurm(command, **kwargs):
        nonlocal job_name
        if command[0] == "sbatch":
            job_name = command[1].split("=", 1)[1]
            batch = Path(command[-1])
            stderr = next(
                Path(line.split("=", 1)[1])
                for line in batch.read_text().splitlines()
                if line.startswith("#SBATCH --error=")
            )
            env = dict(kwargs.get("env") or os.environ, SLURM_JOB_ID="123")
            first = run(["bash", str(batch)], env=env, capture_output=True, timeout=4)
            assert first.returncode == (0 if restart_output == "removed" else 23), (
                first.stderr
            )
            env["SLURM_RESTART_COUNT"] = "1"
            restart = run(["bash", str(batch)], env=env, capture_output=True, timeout=4)
            stderr.write_bytes(restart.stderr)
            assert restart.returncode == 0, restart.stderr
            output = "Submitted batch job 123\n"
        elif command[0] == "squeue":
            output = "COMPLETED|1\n"
        elif command[0] == "scontrol":
            output = f"JobId=123 JobName={job_name} JobState=COMPLETED ExitCode=0:0 Restarts=1"
        else:
            assert command[0] == "sacct"
            status = "FAILED|23:0" if restart_output == "fresh" else "COMPLETED|0:0"
            output = f"123|{job_name}|{status}\n"
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(subprocess, "run", fake_slurm)
    monkeypatch.setattr(
        "shinka.launch.slurm_evaluation.time", SimpleNamespace(sleep=lambda _delay: None)
    )
