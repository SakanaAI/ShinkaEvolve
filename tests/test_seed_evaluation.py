import asyncio
import json
import subprocess
import time
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import (
    AsyncProgramDatabase,
    DatabaseConfig,
    Program,
    ProgramDatabase,
)
from shinka.launch import (
    JobScheduler,
    LocalJobConfig,
    SlurmCondaJobConfig,
    SlurmDockerJobConfig,
    local,
)
from shinka.prompts.prompts_base import construct_individual_program_msg
from shinka.utils import load_results


SEED = "def approximate(x):\n    return x\n"


def _run_seed(
    tmp_path: Path,
    config: LocalJobConfig,
    code: str = SEED,
    *,
    scheduler: JobScheduler | None = None,
) -> tuple[Program, bool, Mock]:
    db_config = DatabaseConfig(db_path=str(tmp_path / "programs.db"), num_islands=1)
    db = ProgramDatabase(db_config)
    runner = ShinkaEvolveRunner.__new__(ShinkaEvolveRunner)
    runner.db_config = db_config
    runner.results_dir = str(tmp_path)
    runner.lang_ext = "py"
    runner.verbose = False
    runner.embedding_client = None
    runner.max_proposal_jobs = 1
    runner.max_evaluation_jobs = 1
    runner.max_db_workers = 1
    runner.total_api_cost = 0.0
    runner.meta_summarizer = None
    runner.llm_selection = Mock()
    runner.stuck_detection_count = 0
    runner.wandb_logger = None
    runner.scheduler = scheduler or JobScheduler("local", config)
    runner.async_db = AsyncProgramDatabase(db)

    async def setup() -> None:
        try:
            await runner._setup_initial_program(code)
        finally:
            await runner.async_db.close_async()

    try:
        asyncio.run(setup())
        (seed,) = db.get_all_programs()
        parent, _, _, needs_fix = db.sample_with_fix_mode()
        assert parent.id == seed.id
        best = db.get_best_program()
        assert (best.id if best else None) == (seed.id if seed.correct else None)
        assert (tmp_path / "gen_0" / "main.py").read_text() == code
        assert (tmp_path / "gen_0" / "results").is_dir()
        assert seed.generation == 0
        assert seed.metadata["patch_type"] == "init"
        return seed, needs_fix, runner.llm_selection.set_baseline_score
    finally:
        runner.scheduler.shutdown()
        db.close()


def _write_evaluator(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "evaluate.py"
    path.write_text(
        "import sys\nimport time\nfrom pathlib import Path\n"
        "results = Path(sys.argv[sys.argv.index('--results_dir') + 1])\n"
        "print('evaluator stdout')\nprint('evaluator stderr', file=sys.stderr)\n"
        + dedent(body)
    )
    return path


def test_seed_timeout_kills_local_evaluator_with_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _write_evaluator(
        tmp_path,
        """
        (results / 'correct.json').write_text('{"correct": true}')
        (results / 'metrics.json').write_text('{"combined_score": 42}')
        (results / 'ready').touch()
        time.sleep(5)
        (results / 'finished').touch()
        """,
    )
    results_dir = tmp_path / "gen_0" / "results"

    def monitor_after_partial_results(process, directory, **kwargs):
        ready_deadline = time.monotonic() + 5.0
        while (
            not (results_dir / "ready").exists() and time.monotonic() < ready_deadline
        ):
            time.sleep(0.01)
        return local.monitor(process, directory, **kwargs)

    monkeypatch.setattr(
        "shinka.launch.scheduler.monitor_local", monitor_after_partial_results
    )
    seed, needs_fix, baseline = _run_seed(
        tmp_path, LocalJobConfig(eval_program_path=str(evaluator), time="00:00:01")
    )

    assert (results_dir / "ready").exists()
    assert not (results_dir / "finished").exists()
    assert seed.correct == 0
    assert needs_fix
    assert seed.combined_score == 0.0
    assert seed.metadata["evaluation_failed"] is True
    assert "00:00:01" in seed.metadata["stderr_log"]
    assert "evaluator stdout" in seed.metadata["stdout_log"]
    assert "evaluator stderr" in seed.metadata["stderr_log"]
    baseline.assert_called_once_with(0.0)


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
@pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "COMPLETED"])
def test_slurm_seed_requires_successful_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, state: str
) -> None:
    def submit(results_dir, *args, **kwargs):
        path = Path(results_dir)
        (path / "correct.json").write_text('{"correct": true}')
        (path / "metrics.json").write_text('{"combined_score": 42}')
        (path / "job_log.err").write_text("evaluator stderr")
        return "123"

    def run(cmd, **kwargs):
        output = "" if cmd[0] == "squeue" else f"123|{state}|0:0"
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    monkeypatch.setattr(f"shinka.launch.scheduler.submit_{backend}", submit)
    monkeypatch.setattr("shinka.launch.slurm.subprocess.run", run)
    config = (
        SlurmCondaJobConfig() if backend == "slurm_conda" else SlurmDockerJobConfig()
    )
    seed, needs_fix, _ = _run_seed(
        tmp_path, LocalJobConfig(), scheduler=JobScheduler(backend, config)
    )

    assert bool(seed.correct) is (state == "COMPLETED")
    assert needs_fix is (state != "COMPLETED")
    assert seed.combined_score == (42.0 if state == "COMPLETED" else 0.0)
    if state != "COMPLETED":
        assert seed.metadata["evaluation_failed"]
        assert state in seed.metadata["stderr_log"]
        assert "evaluator stderr" in seed.metadata["stderr_log"]


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
@pytest.mark.parametrize(
    ("failed_command", "failure", "expected_attempts"),
    [
        ("sbatch", "timeout", 1),
        ("squeue", "error", 3),
        ("squeue", "timeout", 3),
        ("sacct", "timeout", 6),
    ],
)
def test_slurm_control_failures_enter_seed_fix_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    failed_command: str,
    failure: str,
    expected_attempts: int,
) -> None:
    failed_attempts = 0

    def run(cmd, **kwargs):
        nonlocal failed_attempts
        path = tmp_path / "gen_0" / "results"
        (path / "correct.json").write_text('{"correct": true}')
        (path / "metrics.json").write_text('{"combined_score": 42}')
        (path / "job_log.err").write_text("partial evaluator log")
        if cmd[0] == failed_command:
            failed_attempts += 1
            assert failed_attempts <= expected_attempts, "Unbounded retries"
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            raise subprocess.CalledProcessError(1, cmd, stderr="Controller unavailable")
        output = "Submitted batch job 123" if cmd[0] == "sbatch" else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    monkeypatch.setattr("shinka.launch.slurm.subprocess.run", run)
    monkeypatch.setattr("shinka.launch.slurm.time.sleep", Mock())
    config = (
        SlurmCondaJobConfig()
        if backend == "slurm_conda"
        else SlurmDockerJobConfig(image_tar_path="/tmp/image.tar")
    )
    seed, needs_fix, baseline = _run_seed(
        tmp_path, LocalJobConfig(), scheduler=JobScheduler(backend, config)
    )

    assert not seed.correct and needs_fix
    assert seed.combined_score == 0.0
    assert seed.metadata["evaluation_failed"]
    assert failed_command in seed.metadata["stderr_log"]
    assert "partial evaluator log" in seed.metadata["stderr_log"]
    assert failed_attempts == expected_attempts
    baseline.assert_called_once_with(0.0)


def test_seed_launch_exception_is_saved_for_fix_mode(tmp_path: Path) -> None:
    seed, needs_fix, _ = _run_seed(
        tmp_path, LocalJobConfig(python_executable=str(tmp_path / "missing-python"))
    )

    assert seed.correct == 0
    assert needs_fix
    assert seed.metadata["evaluation_failed"] is True
    assert "FileNotFoundError" in seed.metadata["stderr_log"]
    assert "missing-python" in seed.metadata["stderr_log"]


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_slurm_seed_without_accounting_uses_controller_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, state: str
) -> None:
    def submit(results_dir, *args, **kwargs):
        path = Path(results_dir)
        (path / "correct.json").write_text('{"correct": true}')
        (path / "metrics.json").write_text('{"combined_score": 42}')
        (path / "job_log.err").write_text("partial evaluator log")
        return "123"

    def run(cmd, **kwargs):
        output = (
            f"JobId=123 JobState={state} ExitCode=0:0" if cmd[0] == "scontrol" else ""
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    monkeypatch.setattr(f"shinka.launch.scheduler.submit_{backend}", submit)
    monkeypatch.setattr("shinka.launch.slurm.subprocess.run", run)
    config = (
        SlurmCondaJobConfig() if backend == "slurm_conda" else SlurmDockerJobConfig()
    )
    seed, needs_fix, _ = _run_seed(
        tmp_path, LocalJobConfig(), scheduler=JobScheduler(backend, config)
    )

    assert bool(seed.correct) is (state == "COMPLETED")
    assert needs_fix is (state != "COMPLETED")
    assert seed.combined_score == (42.0 if state == "COMPLETED" else 0.0)
    if state != "COMPLETED":
        assert seed.metadata["evaluation_failed"]
        assert state in seed.metadata["stderr_log"]
        assert "partial evaluator log" in seed.metadata["stderr_log"]


@pytest.mark.parametrize(
    ("correct", "metrics", "exit_code", "diagnostic"),
    [
        pytest.param("null", "{}", 0, "correct.json", id="null-correctness"),
        pytest.param(
            '{"correct": "false"}', "{}", 0, "boolean", id="string-correctness"
        ),
        pytest.param("{", "{}", 0, "JSONDecodeError", id="malformed-correctness"),
        pytest.param("{}", "{}", 0, "boolean", id="missing-correctness-field"),
        pytest.param(None, "{}", 0, "correct.json", id="missing-correctness-file"),
        pytest.param(
            '{"correct": true}', None, 0, "Metrics file not found", id="missing-metrics"
        ),
        pytest.param('{"correct": true}', "[]", 0, "metrics.json", id="list-metrics"),
        pytest.param(
            '{"correct": true}', "{", 0, "metrics.json", id="malformed-metrics"
        ),
        pytest.param(
            '{"correct": true}',
            '{"combined_score": []}',
            0,
            "combined_score",
            id="invalid-score",
        ),
        pytest.param('{"correct": true}', "{}", 4, "status 4", id="nonzero-exit"),
    ],
)
def test_seed_invalid_result_is_saved_for_fix_mode(
    tmp_path: Path,
    correct: str | None,
    metrics: str | None,
    exit_code: int,
    diagnostic: str,
) -> None:
    body = ""
    if metrics is not None:
        body += f"(results / 'metrics.json').write_text({metrics!r})\n"
    if correct is not None:
        body += f"(results / 'correct.json').write_text({correct!r})\n"
    evaluator = _write_evaluator(tmp_path, body + f"sys.exit({exit_code})\n")
    seed, needs_fix, _ = _run_seed(
        tmp_path, LocalJobConfig(eval_program_path=str(evaluator))
    )

    assert seed.correct == 0
    assert needs_fix
    assert seed.metadata["evaluation_failed"] is True
    assert diagnostic in seed.metadata["stderr_log"]
    assert "evaluator stdout" in seed.metadata["stdout_log"]
    assert "evaluator stderr" in seed.metadata["stderr_log"]


def test_result_loader_supplies_empty_metrics_if_the_file_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "correct.json").write_text('{"correct": true}')

    results = load_results(str(tmp_path))

    assert results["metrics"] == {}
    assert results["correct"] == {"correct": True}


@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("score", [0.0, -2.5])
def test_seed_preserves_reported_correctness_and_finite_score(
    tmp_path: Path, correct: bool, score: float
) -> None:
    correctness = '{"correct": true}' if correct else '{"correct": false}'
    metrics = json.dumps(
        {
            "combined_score": score,
            "public": {"measured": 0},
            "text_feedback": "feedback",
        }
    )
    evaluator = _write_evaluator(
        tmp_path,
        f"(results / 'correct.json').write_text({correctness!r})\n"
        f"(results / 'metrics.json').write_text({metrics!r})\n",
    )
    seed, needs_fix, baseline = _run_seed(
        tmp_path, LocalJobConfig(eval_program_path=str(evaluator))
    )

    assert seed.correct == correct
    assert needs_fix is not correct
    assert seed.combined_score == score
    assert seed.public_metrics == {"measured": 0}
    assert seed.text_feedback == "feedback"
    assert "evaluation_failed" not in seed.metadata
    baseline.assert_called_once_with(score if correct else 0.0)


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "negative-inf"],
)
def test_incorrect_seed_with_nonfinite_score_can_be_formatted(
    tmp_path: Path, score: float
) -> None:
    correctness = json.dumps({"correct": False, "error": "invalid output"})
    metrics = json.dumps({"combined_score": score, "public": {"measured": 0}})
    evaluator = _write_evaluator(
        tmp_path,
        f"(results / 'correct.json').write_text({correctness!r})\n"
        f"(results / 'metrics.json').write_text({metrics!r})\n",
    )
    seed, needs_fix, _ = _run_seed(
        tmp_path, LocalJobConfig(eval_program_path=str(evaluator))
    )

    prompt = construct_individual_program_msg(seed, include_text_feedback=True)
    assert "Combined score to maximize: 0.00" in prompt
    assert "invalid output" in prompt
    assert needs_fix
    assert seed.combined_score == 0.0
    assert seed.public_metrics == {"measured": 0}
    assert "non-finite" in seed.metadata["stderr_log"]
    assert "evaluator stderr" in seed.metadata["stderr_log"]
