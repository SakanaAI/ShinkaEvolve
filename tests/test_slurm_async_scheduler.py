import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from job_lifecycle_test_utils import (
    PURGED_JOB_ERROR,
    FakeClock,
    fake_squeue,
    squeue_failure,
)
from shinka.launch import JobScheduler, SlurmCondaJobConfig
from shinka.launch import scheduler as scheduler_module
from shinka.launch import slurm, slurm_status


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    fake.install(monkeypatch, slurm, slurm_status)
    return fake


@pytest.fixture
def scheduler():
    instance = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    yield instance
    instance.shutdown()


def _jobs(*job_ids: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(job_id=job_id) for job_id in job_ids]


def test_purged_job_is_no_longer_tracked_as_running(
    scheduler: JobScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_squeue(monkeypatch, lambda job_ids: squeue_failure(PURGED_JOB_ERROR))

    assert scheduler.check_job_status(SimpleNamespace(job_id="123")) is False


def test_monitor_cycle_asks_the_controller_once_for_all_jobs(
    scheduler: JobScheduler, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: ["101", "103"])

    async def poll_three_cycles() -> list[list[bool]]:
        jobs = _jobs("101", "102", "103")
        return [await scheduler.batch_check_status_async(jobs) for _ in range(3)]

    assert asyncio.run(poll_three_cycles()) == [[True, False, True]] * 3
    assert [call["job_ids"] for call in calls] == [["101", "102", "103"]]


def test_status_failure_is_reported_per_job_instead_of_raised(
    scheduler: JobScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(job_ids):
        raise ValueError("unexpected")

    monkeypatch.setattr(scheduler._slurm_queue, "active_jobs", broken)

    outcomes = asyncio.run(scheduler.batch_check_status_async(_jobs("101", "102")))

    assert [str(outcome) for outcome in outcomes] == ["unexpected", "unexpected"]


def test_result_retrieval_never_waits_past_its_deadline(
    scheduler: JobScheduler,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    tmp_path: Path,
) -> None:
    (tmp_path / "correct.json").write_text('{"correct": true}')
    (tmp_path / "metrics.json").write_text('{"combined_score": 7}')
    fake_squeue(monkeypatch, lambda job_ids: job_ids)
    started = clock.now

    results = scheduler.get_job_results("123", str(tmp_path))

    assert results is not None and results["metrics"]["combined_score"] == 7
    waited = clock.now - started
    assert waited == pytest.approx(scheduler_module.SLURM_RESULTS_SETTLE_SECONDS)
    # The runner abandons a retrieval that takes longer than 30 seconds.
    assert waited < 30


@pytest.mark.parametrize(
    ("returncode", "stderr", "cancelled"),
    [
        (0, "", True),
        (
            1,
            "scancel: error: Kill job error on job id 123: Invalid job id specified\n",
            True,
        ),
        (1, "scancel: error: Kill job error on job id 123: Access denied\n", False),
        (1, None, False),
    ],
)
def test_cancelling_a_purged_job_reports_it_gone(
    scheduler: JobScheduler,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: Optional[str],
    cancelled: bool,
) -> None:
    def run(cmd, **kwargs):
        assert cmd == ["scancel", "123"]
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)

    assert asyncio.run(scheduler.cancel_job_async("123")) is cancelled
