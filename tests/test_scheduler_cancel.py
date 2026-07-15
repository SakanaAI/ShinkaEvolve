import asyncio
import subprocess

from shinka.launch import scheduler as scheduler_module
from shinka.launch.scheduler import JobScheduler, SlurmCondaJobConfig


def test_slurm_cancel_waits_for_job_to_leave_queue(monkeypatch) -> None:
    statuses = iter(["RUNNING", "COMPLETING", ""])
    status_checks: list[str] = []
    sleeps: list[float] = []

    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    def get_job_status(job_id: str) -> str:
        status_checks.append(job_id)
        return next(statuses)

    monkeypatch.setattr(scheduler_module, "get_job_status", get_job_status)
    monkeypatch.setattr(scheduler_module.time, "sleep", sleeps.append)

    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        cancelled = asyncio.run(scheduler.cancel_job_async("12345"))
    finally:
        scheduler.shutdown()

    assert cancelled is True
    assert status_checks == ["12345", "12345", "12345"]
    assert len(sleeps) == 2


def test_slurm_cancel_fails_when_job_remains_queued(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(
        scheduler_module, "get_job_status", lambda _job_id: "COMPLETING"
    )
    monkeypatch.setattr(scheduler_module, "_SLURM_CANCEL_TIMEOUT_SECONDS", 0.0)

    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    try:
        cancelled = asyncio.run(scheduler.cancel_job_async("12345"))
    finally:
        scheduler.shutdown()

    assert cancelled is False
