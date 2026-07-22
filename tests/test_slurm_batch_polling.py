import asyncio
from types import SimpleNamespace

from shinka.launch import slurm
from shinka.launch.scheduler import JobScheduler, SlurmCondaJobConfig


def test_get_active_job_ids_batches_squeue(monkeypatch):
    calls = []

    class Result:
        stdout = "101\n103\n"
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.get_active_job_ids(["101", "102", "103", "local-x"]) == {
        "101",
        "103",
    }
    assert len(calls) == 1


def test_get_active_job_ids_empty_and_failure(monkeypatch):
    assert slurm.get_active_job_ids([]) == set()
    assert slurm.get_active_job_ids(["local-only"]) == set()

    class Result:
        stdout = ""
        returncode = 1

    monkeypatch.setattr(slurm.subprocess, "run", lambda *args, **kwargs: Result())

    assert slurm.get_active_job_ids(["999"]) is None


def test_batch_status_keeps_local_job_running(monkeypatch):
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    running = SimpleNamespace(poll=lambda: None)
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-running", {"popen": running})

    try:
        statuses = asyncio.run(
            scheduler.batch_check_status_async(
                [SimpleNamespace(job_id="local-running")]
            )
        )
    finally:
        scheduler.executor.shutdown(wait=True)

    assert statuses == [True]


def test_batch_status_combines_local_and_slurm_jobs(monkeypatch):
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    running = SimpleNamespace(poll=lambda: None)
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-running", {"popen": running})
    monkeypatch.setattr(slurm, "get_active_job_ids", lambda job_ids: {"101"})
    jobs = [
        SimpleNamespace(job_id="local-running"),
        SimpleNamespace(job_id="101"),
        SimpleNamespace(job_id="102"),
    ]

    try:
        statuses = asyncio.run(scheduler.batch_check_status_async(jobs))
    finally:
        scheduler.executor.shutdown(wait=True)

    assert statuses == [True, True, False]


def test_batch_status_preserves_raising_local_result_in_mixed_job_order(monkeypatch):
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    jobs = [
        SimpleNamespace(job_id="101"),
        SimpleNamespace(job_id="local-raising"),
        SimpleNamespace(job_id="102"),
    ]
    monkeypatch.setattr(slurm, "get_active_job_ids", lambda job_ids: {"101"})

    async def check(job):
        raise RuntimeError(f"status failed for {job.job_id}")

    monkeypatch.setattr(scheduler, "check_job_status_async", check)

    try:
        statuses = asyncio.run(scheduler.batch_check_status_async(jobs))
    finally:
        scheduler.executor.shutdown(wait=True)

    assert statuses[0] is True
    assert isinstance(statuses[1], RuntimeError)
    assert str(statuses[1]) == "status failed for local-raising"
    assert statuses[2] is False


def test_batch_status_falls_back_in_original_order_when_squeue_fails(monkeypatch):
    scheduler = JobScheduler("slurm_conda", SlurmCondaJobConfig())
    jobs = [SimpleNamespace(job_id="101"), SimpleNamespace(job_id="102")]
    monkeypatch.setattr(slurm, "get_active_job_ids", lambda job_ids: None)

    async def check(job):
        return job.job_id == "102"

    monkeypatch.setattr(scheduler, "check_job_status_async", check)

    try:
        statuses = asyncio.run(scheduler.batch_check_status_async(jobs))
    finally:
        scheduler.executor.shutdown(wait=True)

    assert statuses == [False, True]
