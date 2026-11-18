import asyncio
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_lifecycle_test_utils import (
    is_alive,
    kill_leftovers,
    wait_until,
    worker_pid,
    write_evaluator_with_worker,
)
from shinka.launch import (
    JobScheduler,
    LocalJobConfig,
    ProcessWithLogging,
    SlurmCondaJobConfig,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="requires POSIX process groups"
)


@pytest.fixture
def scheduler(tmp_path: Path, request: pytest.FixtureRequest):
    time_limit = getattr(request, "param", None)
    config = LocalJobConfig(
        eval_program_path=str(write_evaluator_with_worker(tmp_path)), time=time_limit
    )
    instance = JobScheduler("local", config)
    yield instance
    instance.shutdown()


def _submit(scheduler: JobScheduler, tmp_path: Path) -> tuple[ProcessWithLogging, int]:
    results_dir = tmp_path / "results"
    process = scheduler.submit_async("main.py", str(results_dir))
    assert isinstance(process, ProcessWithLogging)
    return process, worker_pid(results_dir)


def test_cancel_kills_an_async_evaluator_and_its_workers(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    process, worker = _submit(scheduler, tmp_path)
    try:
        assert asyncio.run(scheduler.cancel_job_async(process)) is True

        assert process.returncode == -signal.SIGKILL
        assert wait_until(lambda: not is_alive(worker))
        assert all(log_file.closed for log_file in process.log_files)
    finally:
        kill_leftovers(process.pid, worker)


@pytest.mark.parametrize("scheduler", ["00:00:01"], indirect=True)
def test_timeout_kills_an_async_evaluator_and_its_workers(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    process, worker = _submit(scheduler, tmp_path)
    try:
        fresh = SimpleNamespace(job_id=process, start_time=time.time(), generation=1)
        assert scheduler.check_job_status(fresh) is True
        expired = SimpleNamespace(
            job_id=process, start_time=time.time() - 5.0, generation=1
        )
        assert scheduler.check_job_status(expired) is False

        assert process.returncode == -signal.SIGKILL
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(process.pid, worker)


def test_shutdown_releases_a_seed_evaluation_that_is_still_running(
    scheduler: JobScheduler, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results"
    outcome: list[BaseException] = []

    def evaluate_seed() -> None:
        try:
            scheduler.run("main.py", str(results_dir))
        except BaseException as error:
            outcome.append(error)

    seed = threading.Thread(target=evaluate_seed, daemon=True)
    seed.start()
    worker = worker_pid(results_dir)
    try:
        scheduler.shutdown()
        seed.join(timeout=5.0)

        assert not seed.is_alive()
        assert len(outcome) == 1 and "exited with status -9" in str(outcome[0])
        assert wait_until(lambda: not is_alive(worker))
    finally:
        kill_leftovers(worker)


def test_local_scheduler_arms_the_fatal_signal_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed: list[str] = []
    monkeypatch.setattr(
        "shinka.launch.scheduler.kill_evaluators_on_fatal_signals",
        lambda: armed.append("armed"),
    )

    JobScheduler("slurm_conda", SlurmCondaJobConfig()).shutdown()
    assert armed == []
    JobScheduler("local", LocalJobConfig()).shutdown()
    assert armed == ["armed"]
