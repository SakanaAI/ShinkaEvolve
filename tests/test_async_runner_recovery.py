import asyncio
import time
from types import SimpleNamespace

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner


class _FakeAsyncDB:
    def __init__(self, total_programs: int):
        self.total_programs = total_programs

    async def get_total_program_count_async(self):
        return self.total_programs


class _FakeSlotPool:
    def __init__(self):
        self.released = []

    async def release(self, worker_id):
        self.released.append(worker_id)


def _build_runner(**overrides):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.async_db = overrides.get("async_db", _FakeAsyncDB(0))
    runner.db = overrides.get("db", SimpleNamespace(last_iteration=0))
    runner.db_config = overrides.get("db_config", SimpleNamespace(num_islands=1))
    runner.job_config = overrides.get("job_config", SimpleNamespace(time=None))
    runner.scheduler = overrides.get("scheduler", SimpleNamespace(job_type="local"))
    runner.evo_config = overrides.get("evo_config", SimpleNamespace(num_generations=10))
    runner.completed_generations = overrides.get("completed_generations", 0)
    runner.next_generation_to_submit = overrides.get("next_generation_to_submit", 1)
    runner.running_jobs = overrides.get("running_jobs", [])
    runner.active_proposal_tasks = overrides.get("active_proposal_tasks", {})
    runner.failed_jobs_for_retry = overrides.get("failed_jobs_for_retry", {})
    runner.assigned_generations = overrides.get("assigned_generations", set())
    runner.evaluation_slot_pool = overrides.get("evaluation_slot_pool", _FakeSlotPool())
    runner.sampling_slot_pool = overrides.get("sampling_slot_pool", _FakeSlotPool())
    runner._evaluation_seconds_ewma = overrides.get("evaluation_ewma")
    return runner


def test_restore_resume_progress_uses_actual_program_count():
    async def _run():
        runner = _build_runner(
            async_db=_FakeAsyncDB(total_programs=7),
            db=SimpleNamespace(last_iteration=8),
            db_config=SimpleNamespace(num_islands=2),
            evo_config=SimpleNamespace(num_generations=10),
        )

        await runner._restore_resume_progress()

        assert runner.completed_generations == 6
        assert runner.next_generation_to_submit == 9

    asyncio.run(_run())


def test_get_remaining_completed_work_accounts_for_inflight_jobs():
    runner = _build_runner(
        evo_config=SimpleNamespace(num_generations=5),
        completed_generations=2,
        running_jobs=[object()],
        active_proposal_tasks={"proposal-1": object()},
        failed_jobs_for_retry={},
        next_generation_to_submit=99,
    )

    assert runner._get_remaining_completed_work() == 1


def test_cleanup_proposal_task_state_releases_generation_and_slot():
    async def _run():
        slot_pool = _FakeSlotPool()
        runner = _build_runner(
            assigned_generations={7},
            active_proposal_tasks={"task-1": object()},
            sampling_slot_pool=slot_pool,
        )

        await runner._cleanup_proposal_task_state(
            generation=7,
            task_id="task-1",
            sampling_worker_id=3,
        )

        assert runner.assigned_generations == set()
        assert runner.active_proposal_tasks == {}
        assert slot_pool.released == [3]

    asyncio.run(_run())


def test_get_evaluation_runtime_limit_uses_ewma_when_timeout_not_configured():
    runner = _build_runner(
        job_config=SimpleNamespace(time=None),
        scheduler=SimpleNamespace(job_type="local"),
        evaluation_ewma=30.0,
    )

    assert runner._get_evaluation_runtime_limit_seconds() == 210.0


def test_is_job_hung_when_runtime_exceeds_limit():
    runner = _build_runner(
        job_config=SimpleNamespace(time=None),
        scheduler=SimpleNamespace(job_type="local"),
        evaluation_ewma=30.0,
    )
    job = AsyncRunningJob(
        job_id="job-1",
        exec_fname="program.py",
        results_dir="results",
        start_time=time.time() - 400.0,
        proposal_started_at=time.time() - 400.0,
        evaluation_submitted_at=time.time() - 390.0,
        evaluation_started_at=time.time() - 380.0,
        generation=4,
    )

    assert runner._is_job_hung(job) is True
