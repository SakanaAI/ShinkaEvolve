import asyncio
import logging
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Optional

import pytest

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.launch import JobScheduler, LocalJobConfig, ProcessWithLogging


def _runner(scheduler, time_limit: Optional[str] = None) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.scheduler = scheduler
    runner.job_config = SimpleNamespace(time=time_limit)
    return runner


def _job(submitted_seconds_ago: float) -> AsyncRunningJob:
    submitted_at = time.time() - submitted_seconds_ago
    return AsyncRunningJob(
        job_id="123",
        exec_fname="main.py",
        results_dir="results",
        start_time=submitted_at,
        proposal_started_at=submitted_at - 60.0,
        evaluation_submitted_at=submitted_at,
        evaluation_started_at=submitted_at,
        generation=1,
    )


@pytest.mark.parametrize("job_type", ["slurm_conda", "slurm_docker"])
def test_queue_wait_never_counts_against_a_slurm_job(job_type: str) -> None:
    runner = _runner(SimpleNamespace(job_type=job_type), time_limit="01:00:00")

    # 45 minutes pending plus 20 minutes running, and a wait longer than `time`.
    assert runner._get_evaluation_runtime_limit_seconds() is None
    assert runner._is_job_hung(_job(submitted_seconds_ago=65 * 60)) is False
    assert runner._is_job_hung(_job(submitted_seconds_ago=5 * 3600)) is False


def test_local_time_limit_accepts_slurm_formats() -> None:
    runner = _runner(SimpleNamespace(job_type="local"), time_limit="1-00:00:00")

    assert runner._get_evaluation_runtime_limit_seconds() == 86400.0
    assert runner._is_job_hung(_job(submitted_seconds_ago=3600)) is False
    assert runner._is_job_hung(_job(submitted_seconds_ago=86401)) is True


def test_unparseable_time_cannot_take_the_job_monitor_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _runner(SimpleNamespace(job_type="local"), time_limit="one hour")

    with caplog.at_level(logging.ERROR):
        assert runner._is_job_hung(_job(submitted_seconds_ago=10)) is False
        assert runner._is_job_hung(_job(submitted_seconds_ago=10)) is False

    assert caplog.text.count("Hung-job detection disabled") == 1


def _write_evaluator(tmp_path: Path) -> Path:
    """Report the program's text as correct unless the program is 'crash'."""
    path = tmp_path / "evaluate.py"
    path.write_text(
        dedent("""
            import json
            import sys
            from pathlib import Path

            argument = lambda name: sys.argv[sys.argv.index(name) + 1]
            results = Path(argument("--results_dir"))
            results.mkdir(parents=True, exist_ok=True)
            correct = Path(argument("--program_path")).read_text() != "crash"
            (results / "correct.json").write_text(json.dumps({"correct": correct}))
            (results / "metrics.json").write_text(json.dumps({"combined_score": 1.0}))
            """)
    )
    return path


@pytest.fixture
def scheduler(tmp_path: Path):
    config = LocalJobConfig(eval_program_path=str(_write_evaluator(tmp_path)))
    instance = JobScheduler("local", config)
    yield instance
    instance.shutdown()


def _evaluate(
    scheduler: JobScheduler,
    tmp_path: Path,
    name: str,
    program: str,
    *,
    took_seconds: float,
    killed: bool = False,
) -> None:
    """Run one asynchronous evaluation and report it as having taken this long."""
    program_path = tmp_path / f"{name}.py"
    program_path.write_text(program)
    results_dir = tmp_path / name
    process = scheduler.submit_async(str(program_path), str(results_dir))
    assert isinstance(process, ProcessWithLogging)
    assert process.wait(timeout=10.0) == 0
    process.launched_at = time.monotonic() - took_seconds
    process.was_killed = killed
    job = SimpleNamespace(job_id=process, start_time=time.time(), generation=1)
    assert scheduler.check_job_status(job) is False
    results = scheduler.get_job_results(process, str(results_dir))
    assert results is not None
    assert results["correct"]["correct"] is (program != "crash")


def test_early_crash_does_not_shrink_the_adaptive_limit(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    runner = _runner(scheduler)
    assert runner._get_evaluation_runtime_limit_seconds() is None

    _evaluate(scheduler, tmp_path, "crash_first", "crash", took_seconds=1.0)
    assert scheduler.longest_successful_runtime is None

    _evaluate(scheduler, tmp_path, "healthy", "ok", took_seconds=300.0)
    _evaluate(scheduler, tmp_path, "crash_later", "crash", took_seconds=1.0)
    _evaluate(scheduler, tmp_path, "quick", "ok", took_seconds=2.0)

    assert scheduler.longest_successful_runtime == pytest.approx(300.0, abs=5.0)
    limit = runner._get_evaluation_runtime_limit_seconds()
    assert limit is not None
    assert limit == pytest.approx(300.0 * 5.0 + 60.0, abs=25.0)
    assert runner._is_job_hung(_job(submitted_seconds_ago=310.0)) is False
    assert runner._is_job_hung(_job(submitted_seconds_ago=limit + 30.0)) is True


def test_killed_evaluation_does_not_extend_the_adaptive_limit(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    _evaluate(scheduler, tmp_path, "healthy", "ok", took_seconds=20.0)
    _evaluate(scheduler, tmp_path, "hung", "ok", took_seconds=900.0, killed=True)

    assert scheduler.longest_successful_runtime == pytest.approx(20.0, abs=5.0)


def test_hang_limit_exists_before_any_evaluation_has_succeeded(
    scheduler: JobScheduler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A candidate that loops forever must not hold its slot for the rest of the run.
    runner = _runner(scheduler)

    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        _evaluate(scheduler, tmp_path, "crash", "crash", took_seconds=1.0)
        limit = runner._get_evaluation_runtime_limit_seconds()
        runner._get_evaluation_runtime_limit_seconds()

    assert limit == pytest.approx(1.0 * 5.0 + 60.0, abs=25.0)
    assert runner._is_job_hung(_job(submitted_seconds_ago=10 * 3600)) is True
    assert caplog.text.count("Hang limit for local evaluations") == 1
    assert "no evaluation has succeeded yet" in caplog.text


def test_evaluation_killed_before_any_success_raises_the_provisional_limit(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    # A limit learned from an early crash may be too low for a healthy
    # candidate. It has to grow, or every slower candidate would be killed.
    runner = _runner(scheduler)
    _evaluate(scheduler, tmp_path, "crash", "crash", took_seconds=1.0)

    _evaluate(scheduler, tmp_path, "slow", "ok", took_seconds=70.0, killed=True)

    assert scheduler.longest_successful_runtime is None
    limit = runner._get_evaluation_runtime_limit_seconds()
    assert limit == pytest.approx(70.0 * 5.0 + 60.0, abs=25.0)


def test_first_success_replaces_the_provisional_limit(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    runner = _runner(scheduler)
    _evaluate(scheduler, tmp_path, "slow_crash", "crash", took_seconds=600.0)

    _evaluate(scheduler, tmp_path, "healthy", "ok", took_seconds=20.0)

    limit = runner._get_evaluation_runtime_limit_seconds()
    assert limit == pytest.approx(20.0 * 5.0 + 60.0, abs=25.0)


def test_seed_evaluation_sets_the_adaptive_limit(
    scheduler: JobScheduler, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(scheduler)
    program_path = tmp_path / "seed.py"
    program_path.write_text("ok")
    clock = iter([1000.0, 1300.0])
    monkeypatch.setattr(
        "shinka.launch.scheduler.time", SimpleNamespace(time=lambda: next(clock))
    )

    results, runtime = scheduler.run(str(program_path), str(tmp_path / "seed"))

    assert results["correct"]["correct"] is True
    assert runtime == 300.0
    assert scheduler.longest_successful_runtime == 300.0
    assert runner._get_evaluation_runtime_limit_seconds() == 1560.0


def test_incorrect_seed_sets_only_a_provisional_limit(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    runner = _runner(scheduler)
    program_path = tmp_path / "seed.py"
    program_path.write_text("crash")

    results, _runtime = scheduler.run(str(program_path), str(tmp_path / "seed"))

    assert results["correct"]["correct"] is False
    assert scheduler.longest_successful_runtime is None
    assert runner._get_evaluation_runtime_limit_seconds() == pytest.approx(60.0, abs=25.0)


def test_seed_whose_evaluator_fails_sets_a_provisional_limit(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("raise SystemExit(3)")
    scheduler = JobScheduler("local", LocalJobConfig(eval_program_path=str(evaluator)))
    try:
        with pytest.raises(RuntimeError, match="exited with status 3"):
            scheduler.run(str(tmp_path / "seed.py"), str(tmp_path / "seed"))

        limit = _runner(scheduler)._get_evaluation_runtime_limit_seconds()
    finally:
        scheduler.shutdown()

    assert limit == pytest.approx(60.0, abs=25.0)


def _history(best_program) -> SimpleNamespace:
    async def get_best_program_async():
        if isinstance(best_program, BaseException):
            raise best_program
        return best_program

    return SimpleNamespace(get_best_program_async=get_best_program_async)


def _program(correct: bool, evaluation_seconds) -> SimpleNamespace:
    return SimpleNamespace(
        correct=correct, metadata={"evaluation_seconds": evaluation_seconds}
    )


def test_resumed_run_takes_its_hang_limit_from_the_best_earlier_evaluation(
    scheduler: JobScheduler,
) -> None:
    runner = _runner(scheduler)
    runner.async_db = _history(_program(True, 120.0))

    asyncio.run(runner._restore_hang_limit_from_history())

    assert runner._get_evaluation_runtime_limit_seconds() == 660.0


@pytest.mark.parametrize(
    "best_program",
    [None, _program(False, 120.0), _program(True, None), OSError("database locked")],
    ids=["empty", "incorrect", "untimed", "unreadable"],
)
def test_history_without_a_timed_success_sets_no_hang_limit(
    scheduler: JobScheduler, best_program
) -> None:
    runner = _runner(scheduler)
    runner.async_db = _history(best_program)

    asyncio.run(runner._restore_hang_limit_from_history())

    assert runner._get_evaluation_runtime_limit_seconds() is None


def test_configured_time_does_not_consult_the_history(
    scheduler: JobScheduler,
) -> None:
    runner = _runner(scheduler, time_limit="00:05:00")
    runner.async_db = _history(AssertionError("history must not be read"))

    asyncio.run(runner._restore_hang_limit_from_history())

    assert scheduler.longest_successful_runtime is None
