import asyncio
import time
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.core.pipeline_timing import (
    summarize_timing_metadata,
    with_pipeline_timing,
    with_side_effect_timing,
)
from shinka.core.runtime_slots import LogicalSlotPool
from shinka.database import Program
from shinka.launch import LocalJobConfig


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
    assert metadata["post_eval_queue_wait_seconds"] == 0.0
    assert metadata["postprocess_seconds"] == 3.5
    assert metadata["pipeline_accounted_seconds"] == 21.0
    assert metadata["pipeline_seconds"] == 21.0
    assert metadata["pipeline_unaccounted_seconds"] == 0.0
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
    assert metadata["post_eval_queue_wait_seconds"] == 0.0
    assert metadata["postprocess_seconds"] == 0.0
    assert metadata["pipeline_accounted_seconds"] == 0.0
    assert metadata["pipeline_seconds"] == 0.0
    assert metadata["pipeline_unaccounted_seconds"] == 0.0


def test_configure_local_job_runtime_sets_numeric_thread_cap_from_eval_concurrency():
    runner = object.__new__(ShinkaEvolveRunner)
    runner.job_config = LocalJobConfig()
    runner.max_evaluation_jobs = 10
    runner.verbose = False

    runner._configure_local_job_runtime(cpu_count=32)

    assert runner.job_config.numeric_threads_per_job == 3


def test_with_side_effect_timing_adds_wait_and_end_to_end_fields():
    metadata = with_side_effect_timing(
        {
            "pipeline_started_at": 10.0,
            "postprocess_finished_at": 31.0,
            "pipeline_accounted_seconds": 21.0,
        },
        apply_started_at=36.0,
        apply_finished_at=40.0,
    )

    assert metadata["postprocess_apply_wait_seconds"] == 5.0
    assert metadata["postprocess_apply_seconds"] == 4.0
    assert metadata["end_to_end_with_side_effects_seconds"] == 30.0
    assert metadata["end_to_end_accounted_seconds"] == 30.0
    assert metadata["end_to_end_unaccounted_seconds"] == 0.0


def test_summarize_timing_metadata_reports_basic_stats():
    summary = summarize_timing_metadata(
        [
            {"evaluation_seconds": 1.0, "post_eval_queue_wait_seconds": 2.0},
            {"evaluation_seconds": 3.0, "post_eval_queue_wait_seconds": 4.0},
            {"evaluation_seconds": 5.0, "post_eval_queue_wait_seconds": 6.0},
        ],
        ["evaluation_seconds", "post_eval_queue_wait_seconds"],
    )

    assert summary["evaluation_seconds"]["mean"] == pytest.approx(3.0)
    assert summary["evaluation_seconds"]["median"] == pytest.approx(3.0)
    assert summary["evaluation_seconds"]["p90"] == pytest.approx(5.0)
    assert summary["evaluation_seconds"]["max"] == pytest.approx(5.0)
    assert summary["post_eval_queue_wait_seconds"]["mean"] == pytest.approx(4.0)


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


class _SlowFakeScheduler(_FakeScheduler):
    async def get_job_results_async(self, job_id, results_dir):
        await asyncio.sleep(0.02)
        return await super().get_job_results_async(job_id, results_dir)


class _FakeAsyncDB:
    def __init__(self):
        self.programs = []
        self.seen_source_job_ids = set()

    async def add_program_async(self, program, **kwargs):
        source_job_id = (program.metadata or {}).get("source_job_id")
        if source_job_id is not None and source_job_id in self.seen_source_job_ids:
            return False
        self.programs.append(program)
        if source_job_id is not None:
            self.seen_source_job_ids.add(source_job_id)
        return True

    async def has_program_with_source_job_id_async(self, source_job_id):
        return source_job_id in self.seen_source_job_ids

    async def get_program_by_source_job_id_async(self, source_job_id):
        for program in self.programs:
            if (program.metadata or {}).get("source_job_id") == source_job_id:
                return program
        return None


class _ConcurrentRecordingAsyncDB(_FakeAsyncDB):
    def __init__(self):
        super().__init__()
        self.active_adds = 0
        self.peak_adds = 0

    async def add_program_async(self, program, **kwargs):
        self.active_adds += 1
        self.peak_adds = max(self.peak_adds, self.active_adds)
        try:
            await asyncio.sleep(0.05)
            await super().add_program_async(program, **kwargs)
        finally:
            self.active_adds -= 1


class _FailOnceAsyncDB(_FakeAsyncDB):
    def __init__(self):
        super().__init__()
        self.add_attempts = 0

    async def add_program_async(self, program, **kwargs):
        self.add_attempts += 1
        if self.add_attempts == 1:
            raise asyncio.TimeoutError()
        await super().add_program_async(program, **kwargs)


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
        runner.submitted_jobs = {}
        await runner.evaluation_slot_pool.acquire()
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)

        persist_call_count = 0

        async def persist_program_metadata(program):
            nonlocal persist_call_count
            persist_call_count += 1

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
        assert program.metadata["postprocess_worker_id"] == 1
        assert program.metadata["sampling_worker_capacity"] == 2
        assert program.metadata["evaluation_worker_capacity"] == 2
        assert program.metadata["postprocess_worker_capacity"] == 2
        assert persist_call_count == 1

    asyncio.run(_run())


def test_process_single_job_uses_completion_detection_time_for_eval_finish():
    async def _run():
        now = time.time()
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _SlowFakeScheduler()
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
        runner.submitted_jobs = {}
        runner._completed_job_batch_tasks = set()
        runner._completed_jobs_pending = 0
        await runner.evaluation_slot_pool.acquire()
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(
            0, result=None
        )
        runner._record_oversubscription_timing_sample = lambda metadata: None

        completion_detected_at = now - 0.25
        job = AsyncRunningJob(
            job_id="job-detected-finish",
            exec_fname="program.py",
            results_dir="results",
            start_time=now - 2.0,
            proposal_started_at=now - 2.0,
            evaluation_submitted_at=now - 1.5,
            evaluation_started_at=now - 1.0,
            completion_detected_at=completion_detected_at,
            generation=9,
            evaluation_worker_id=1,
        )

        success = await runner._process_single_job_safely(job)

        assert success is True
        assert len(runner.async_db.programs) == 1
        program = runner.async_db.programs[0]
        assert job.results_retrieved_at is not None
        assert job.results_retrieved_at > completion_detected_at
        assert program.metadata["evaluation_finished_at"] == pytest.approx(
            completion_detected_at
        )

    asyncio.run(_run())


def test_process_single_job_safely_flushes_metadata_once_after_side_effects():
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
        runner.submitted_jobs = {}
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._record_oversubscription_timing_sample = lambda metadata: None
        runner._record_progress = lambda: None

        persist_call_count = 0

        async def persist_program_metadata(program):
            nonlocal persist_call_count
            persist_call_count += 1

        runner._persist_program_metadata_async = persist_program_metadata

        job = AsyncRunningJob(
            job_id="job-flush-once",
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
            meta_patch_data={"patch_name": "flush_once"},
        )

        ok = await runner._process_single_job_safely(job)

        assert ok is True
        assert persist_call_count == 1

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
        runner.submitted_jobs = {}
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


def test_process_single_job_safely_reuses_existing_row_when_duplicate_matches():
    class _FakeMetaSummarizer:
        def __init__(self):
            self.programs = []

        def add_evaluated_program(self, program):
            self.programs.append(program.id)

        def should_update_meta(self, _interval):
            return False

    async def _run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = _FakeAsyncDB()
        runner.evo_config = SimpleNamespace(evolve_prompts=False, meta_rec_interval=5)
        runner.meta_summarizer = _FakeMetaSummarizer()
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
        runner.submitted_jobs = {}
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(
            0, result=None
        )
        runner._record_oversubscription_timing_sample = lambda metadata: None

        from shinka.database import Program

        existing_program = Program(
            id="persisted-dup",
            code="print('hi')\n",
            language="python",
            generation=4,
            correct=True,
            combined_score=2.5,
            public_metrics={"acc": 1.0},
            private_metrics={"loss": 0.1},
            text_feedback="ok",
            metadata={"source_job_id": "job-dup"},
        )
        runner.async_db.programs.append(existing_program)
        runner.async_db.seen_source_job_ids.add("job-dup")

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

        ok = await runner._process_single_job_safely(job)

        assert ok is True
        assert len(runner.async_db.programs) == 1
        assert len(runner.meta_summarizer.programs) == 1

    asyncio.run(_run())


def test_process_single_job_safely_ignores_duplicate_marker_on_existing_row():
    class _FakeMetaSummarizer:
        def __init__(self):
            self.programs = []

        def add_evaluated_program(self, program):
            self.programs.append(program.id)

        def should_update_meta(self, _interval):
            return False

    async def _run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = _FakeAsyncDB()
        runner.evo_config = SimpleNamespace(evolve_prompts=False, meta_rec_interval=5)
        runner.meta_summarizer = _FakeMetaSummarizer()
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
        runner.submitted_jobs = {}
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(
            0, result=None
        )
        runner._record_oversubscription_timing_sample = lambda metadata: None

        from shinka.database import Program

        existing_program = Program(
            id="persisted-dup",
            code="print('hi')\n",
            language="python",
            generation=4,
            correct=True,
            combined_score=2.5,
            public_metrics={"acc": 1.0},
            private_metrics={"loss": 0.1},
            text_feedback="ok",
            metadata={
                "source_job_id": "job-dup",
                "postprocess_side_effects_applied": True,
            },
        )
        runner.async_db.programs.append(existing_program)
        runner.async_db.seen_source_job_ids.add("job-dup")

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

        ok = await runner._process_single_job_safely(job)

        assert ok is True
        assert len(runner.async_db.programs) == 1
        assert len(runner.meta_summarizer.programs) == 0

    asyncio.run(_run())


def test_process_single_job_safely_reuses_initial_eval_finish_time_on_retry():
    async def _run():
        now = time.time()
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = _FailOnceAsyncDB()
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
        runner.submitted_jobs = {}
        await runner.evaluation_slot_pool.acquire()
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(
            0, result=None
        )
        runner._record_oversubscription_timing_sample = lambda metadata: None

        job = AsyncRunningJob(
            job_id="job-retry",
            exec_fname="program.py",
            results_dir="results",
            start_time=now - 6.0,
            proposal_started_at=now - 6.0,
            evaluation_submitted_at=now - 2.0,
            evaluation_started_at=now - 1.0,
            generation=5,
            sampling_worker_id=1,
            evaluation_worker_id=1,
            active_proposals_at_start=1,
            running_eval_jobs_at_submit=1,
            meta_patch_data={"patch_name": "retry_patch"},
        )

        first_ok = await runner._process_single_job_safely(job)
        first_finished_at = job.results_retrieved_at

        assert first_ok is False
        assert first_finished_at is not None
        assert runner.evaluation_slot_pool.in_use == 0
        assert "job-retry" in runner.failed_jobs_for_retry

        await asyncio.sleep(0.02)
        second_ok = await runner._process_single_job_safely(job)

        assert second_ok is True
        assert len(runner.async_db.programs) == 1
        assert runner.async_db.add_attempts == 2

        program = runner.async_db.programs[0]
        assert program.metadata["evaluation_finished_at"] == pytest.approx(
            first_finished_at
        )
        assert program.metadata["evaluation_started_at"] == pytest.approx(
            job.evaluation_started_at
        )
        assert program.metadata["postprocess_started_at"] >= first_finished_at

    asyncio.run(_run())


def test_process_completed_jobs_safely_persists_completed_jobs_concurrently():
    async def _run():
        now = time.time()
        async_db = _ConcurrentRecordingAsyncDB()
        runner = object.__new__(ShinkaEvolveRunner)
        runner.scheduler = _FakeScheduler()
        runner.async_db = async_db
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
        runner.running_jobs = []
        runner.submitted_jobs = {}
        runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
        runner._persist_program_metadata_async = lambda program: asyncio.sleep(
            0, result=None
        )
        runner._record_oversubscription_timing_sample = lambda metadata: None
        runner._update_completed_generations = lambda: asyncio.sleep(0, result=None)

        applied_program_ids = []

        async def apply_side_effects(event):
            applied_program_ids.append(event.program.id)
            await asyncio.sleep(0)

        runner._apply_persisted_program_side_effects = apply_side_effects

        jobs = [
            AsyncRunningJob(
                job_id="job-a",
                exec_fname="program_a.py",
                results_dir="results_a",
                start_time=now - 6.0,
                proposal_started_at=now - 6.0,
                evaluation_submitted_at=now - 2.0,
                evaluation_started_at=now - 1.0,
                generation=3,
                sampling_worker_id=1,
                evaluation_worker_id=1,
                meta_patch_data={"patch_name": "patch_a"},
            ),
            AsyncRunningJob(
                job_id="job-b",
                exec_fname="program_b.py",
                results_dir="results_b",
                start_time=now - 5.5,
                proposal_started_at=now - 5.5,
                evaluation_submitted_at=now - 1.8,
                evaluation_started_at=now - 0.9,
                generation=4,
                sampling_worker_id=2,
                evaluation_worker_id=2,
                meta_patch_data={"patch_name": "patch_b"},
            ),
        ]

        await runner._process_completed_jobs_safely(jobs)

        assert len(async_db.programs) == 2
        assert len(applied_program_ids) == 2

    asyncio.run(_run())


def test_process_completed_jobs_safely_waits_for_slow_side_effects():
    async def _run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.running_jobs = []
        runner.active_proposal_tasks = {}
        runner.failed_jobs_for_retry = {}
        runner.processing_lock = asyncio.Lock()
        runner.submitted_jobs = {"job-1": object()}
        runner.should_stop = asyncio.Event()
        runner.slot_available = asyncio.Event()
        runner.completed_generations = 3
        runner.last_progress_time = time.time()
        runner.stuck_detection_count = 0
        runner.cost_limit_reached = False
        runner.evo_config = SimpleNamespace(num_generations=10)
        runner.verbose = False
        runner._update_completed_generations = lambda: asyncio.sleep(0, result=None)

        job = AsyncRunningJob(
            job_id="job-1",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time() - 3.0,
            proposal_started_at=time.time() - 3.0,
            evaluation_submitted_at=time.time() - 1.0,
            generation=4,
        )
        program = Program(
            id="program-1",
            code="print('hi')\n",
            language="python",
            generation=4,
            metadata={},
        )
        persisted_event = runner._make_persisted_event(
            job=job,
            program=program,
            evaluation_finished_at=time.time() - 0.5,
            postprocess_started_at=time.time() - 0.25,
            postprocess_finished_at=time.time() - 0.1,
        )

        async def persist_completed_job(_job):
            return SimpleNamespace(
                success=True,
                persisted_event=persisted_event,
            )

        runner._persist_completed_job = persist_completed_job

        side_effect_started = asyncio.Event()
        allow_side_effect_finish = asyncio.Event()
        applied_program_ids = []

        async def apply_side_effects(event):
            side_effect_started.set()
            await allow_side_effect_finish.wait()
            applied_program_ids.append(event.program.id)

        runner._apply_persisted_program_side_effects = apply_side_effects

        task = asyncio.create_task(runner._process_completed_jobs_safely([job]))
        await asyncio.wait_for(side_effect_started.wait(), timeout=0.2)
        assert task.done() is False
        assert applied_program_ids == []

        allow_side_effect_finish.set()
        await asyncio.wait_for(task, timeout=0.2)

        assert applied_program_ids == ["program-1"]

    asyncio.run(_run())
