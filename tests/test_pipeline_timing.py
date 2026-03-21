import asyncio
import time
from types import SimpleNamespace

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.core.pipeline_timing import with_pipeline_timing
from shinka.core.runtime_slots import LogicalSlotPool


def test_with_pipeline_timing_adds_boundaries_and_durations():
    metadata = with_pipeline_timing(
        {"patch_name": "demo"},
        pipeline_started_at=10.0,
        sampling_started_at=10.0,
        sampling_finished_at=15.0,
        evaluation_started_at=15.0,
        evaluation_finished_at=27.5,
        postprocess_started_at=27.5,
        postprocess_finished_at=31.0,
    )

    assert metadata["patch_name"] == "demo"
    assert metadata["sampling_seconds"] == 5.0
    assert metadata["evaluation_seconds"] == 12.5
    assert metadata["postprocess_seconds"] == 3.5
    assert metadata["pipeline_seconds"] == 21.0
    assert metadata["compute_time"] == 12.5
    assert metadata["pipeline_started_at"] == 10.0
    assert metadata["postprocess_finished_at"] == 31.0


def test_with_pipeline_timing_clamps_negative_stage_durations():
    metadata = with_pipeline_timing(
        {},
        pipeline_started_at=10.0,
        sampling_started_at=10.0,
        sampling_finished_at=9.0,
        evaluation_started_at=9.0,
        evaluation_finished_at=8.0,
        postprocess_started_at=8.0,
        postprocess_finished_at=7.0,
    )

    assert metadata["sampling_seconds"] == 0.0
    assert metadata["evaluation_seconds"] == 0.0
    assert metadata["postprocess_seconds"] == 0.0
    assert metadata["pipeline_seconds"] == 0.0


class _FakeScheduler:
    async def get_job_results_async(self, job_id, results_dir):
        return {
            "correct": {"correct": True},
            "metrics": {
                "combined_score": 2.5,
                "public": {"acc": 1.0},
                "private": {"loss": 0.1},
                "text_feedback": "ok",
            },
            "stdout_log": "stdout",
            "stderr_log": "",
        }


class _FakeAsyncDB:
    def __init__(self):
        self.programs = []
        self.seen_source_job_ids = set()

    async def add_program_async(self, program, **kwargs):
        self.programs.append(program)
        source_job_id = (program.metadata or {}).get("source_job_id")
        if source_job_id is not None:
            self.seen_source_job_ids.add(source_job_id)

    async def has_program_with_source_job_id_async(self, source_job_id):
        return source_job_id in self.seen_source_job_ids


def test_logical_slot_pool_reuses_slots():
    async def _run():
        pool = LogicalSlotPool(2, "test")
        first = await pool.acquire()
        second = await pool.acquire()

        assert {first, second} == {1, 2}
        assert pool.in_use == 2
        assert pool.peak_in_use == 2

        await pool.release(first)
        third = await pool.acquire()

        assert third == first
        assert pool.in_use == 2

    asyncio.run(_run())


def test_process_single_job_safely_persists_timing_metadata():
    async def _run():
        now = time.time()
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = _FakeAsyncDB()
        runner.evo_config = SimpleNamespace(evolve_prompts=False, meta_rec_interval=None)
        runner.meta_summarizer = None
        runner.llm_selection = None
        runner.MAX_DB_RETRY_ATTEMPTS = 3
        runner.failed_jobs_for_retry = {}
        runner.total_api_cost = 0.0
        runner.verbose = False
        runner.console = None
        runner.max_proposal_jobs = 2
        runner.max_evaluation_jobs = 2
        runner.max_db_workers = 2
        runner._sampling_seconds_ewma = None
        runner._evaluation_seconds_ewma = None
        runner._proposal_timing_samples = 0
        runner._last_proposal_target_log = None
        runner.evaluation_slot_pool = LogicalSlotPool(2, "evaluation")
        runner.postprocess_slot_pool = LogicalSlotPool(2, "postprocess")
        await runner.evaluation_slot_pool.acquire()
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)

        persisted_metadata = {}

        async def persist_program_metadata(program):
            persisted_metadata.update(program.metadata or {})

        runner._persist_program_metadata_async = persist_program_metadata

        job = AsyncRunningJob(
            job_id="job-1",
            exec_fname="program.py",
            results_dir="results",
            start_time=now - 6.0,
            proposal_started_at=now - 6.0,
            evaluation_submitted_at=now - 2.0,
            evaluation_started_at=now - 1.0,
            generation=3,
            sampling_worker_id=2,
            evaluation_worker_id=1,
            active_proposals_at_start=2,
            running_eval_jobs_at_submit=1,
            parent_id="parent-1",
            meta_patch_data={"patch_name": "timed_patch", "patch_type": "full"},
            embed_cost=0.2,
            novelty_cost=0.1,
        )

        success = await runner._process_single_job_safely(job)

        assert success
        assert len(runner.async_db.programs) == 1
        program = runner.async_db.programs[0]
        assert program.metadata["patch_name"] == "timed_patch"
        assert program.metadata["sampling_seconds"] >= 4.5
        assert program.metadata["sampling_seconds"] <= 5.5
        assert program.metadata["evaluation_seconds"] >= 0.5
        assert program.metadata["evaluation_seconds"] <= 1.5
        assert program.metadata["postprocess_seconds"] >= 0.0
        assert program.metadata["pipeline_seconds"] >= program.metadata["evaluation_seconds"]
        assert program.metadata["compute_time"] == program.metadata["evaluation_seconds"]
        assert program.metadata["timeline_lane_mode"] == "pool_slots"
        assert program.metadata["sampling_worker_id"] == 2
        assert program.metadata["evaluation_worker_id"] == 1
        assert program.metadata["postprocess_worker_id"] == 1
        assert program.metadata["sampling_worker_capacity"] == 2
        assert program.metadata["evaluation_worker_capacity"] == 2
        assert program.metadata["postprocess_worker_capacity"] == 2
        assert persisted_metadata["pipeline_started_at"] == job.proposal_started_at
        assert persisted_metadata["sampling_finished_at"] == job.evaluation_started_at
        assert persisted_metadata["evaluation_started_at"] == job.evaluation_started_at
        assert persisted_metadata["evaluation_started_at"] > job.evaluation_submitted_at
        assert persisted_metadata["postprocess_finished_at"] >= persisted_metadata["postprocess_started_at"]

    asyncio.run(_run())


def test_process_single_job_safely_skips_duplicate_source_job():
    async def _run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = _FakeAsyncDB()
        runner.evo_config = SimpleNamespace(evolve_prompts=False, meta_rec_interval=None)
        runner.meta_summarizer = None
        runner.llm_selection = None
        runner.MAX_DB_RETRY_ATTEMPTS = 3
        runner.failed_jobs_for_retry = {}
        runner.total_api_cost = 0.0
        runner.verbose = False
        runner.console = None
        runner.max_proposal_jobs = 2
        runner.max_evaluation_jobs = 2
        runner.max_db_workers = 2
        runner._sampling_seconds_ewma = None
        runner._evaluation_seconds_ewma = None
        runner._proposal_timing_samples = 0
        runner._last_proposal_target_log = None
        runner.evaluation_slot_pool = LogicalSlotPool(2, "evaluation")
        runner.postprocess_slot_pool = LogicalSlotPool(2, "postprocess")
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(0, result=None)

        job = AsyncRunningJob(
            job_id="job-dup",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time() - 4.0,
            proposal_started_at=time.time() - 4.0,
            evaluation_submitted_at=time.time() - 1.5,
            evaluation_started_at=time.time() - 1.0,
            generation=4,
            sampling_worker_id=1,
            evaluation_worker_id=1,
            active_proposals_at_start=1,
            running_eval_jobs_at_submit=1,
            meta_patch_data={"patch_name": "dup_patch"},
        )

        ok_first = await runner._process_single_job_safely(job)
        ok_second = await runner._process_single_job_safely(job)

        assert ok_first is True
        assert ok_second is True
        assert len(runner.async_db.programs) == 1

    asyncio.run(_run())
