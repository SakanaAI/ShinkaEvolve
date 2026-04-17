import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import (
    AsyncRunningJob,
    CompletedJobPersistResult,
    PersistedProgramEvent,
    ShinkaEvolveRunner,
)
from shinka.core.runtime_slots import LogicalSlotPool


class _FakeAsyncDB:
    def __init__(self, total_programs: int):
        self.total_programs = total_programs

    async def get_total_program_count_async(self):
        return self.total_programs


class _RecordingAsyncDB(_FakeAsyncDB):
    def __init__(self):
        super().__init__(total_programs=0)
        self.programs = []
        self.maintenance_calls = 0
        self.attempt_events = []

    async def add_program_async(self, program, **kwargs):
        self.programs.append((program, kwargs))
        self.total_programs += 1

    async def run_program_maintenance_async(self, program, verbose=False):
        self.maintenance_calls += 1

    async def record_attempt_event_async(
        self, generation, stage, status, details=None
    ):
        self.attempt_events.append(
            {
                "generation": generation,
                "stage": stage,
                "status": status,
                "details": details or {},
            }
        )


class _FakeSlotPool:
    def __init__(self):
        self.released = []
        self.in_use = 0

    async def release(self, worker_id):
        self.released.append(worker_id)
        if worker_id is not None and self.in_use > 0:
            self.in_use -= 1

    async def acquire(self):
        self.in_use += 1
        return 0


class _FakeScheduler:
    def __init__(self, cancelled_job_ids=None):
        self.cancelled_job_ids = []
        self._cancelled_job_ids = set(cancelled_job_ids or [])

    async def cancel_job_async(self, job_id):
        self.cancelled_job_ids.append(job_id)
        return job_id in self._cancelled_job_ids

    async def submit_async_nonblocking(self, exec_fname, results_dir):
        return f"job-for-{exec_fname}"


class _FakeAsyncDBWithGuard(_FakeAsyncDB):
    def __init__(self):
        super().__init__(total_programs=0)
        self.add_calls = 0
        self.source_job_checks = []

    async def has_program_with_source_job_id_async(self, source_job_id: str):
        self.source_job_checks.append(source_job_id)
        return False

    async def add_program_async(self, *args, **kwargs):
        self.add_calls += 1
        raise AssertionError("discarded jobs must not be persisted")


class _FakeEvent:
    def __init__(self):
        self._is_set = False

    def set(self):
        self._is_set = True

    def clear(self):
        self._is_set = False

    def is_set(self):
        return self._is_set


class _FakeLock:
    def __init__(self):
        self._locked = False

    async def acquire(self):
        self._locked = True

    def release(self):
        self._locked = False

    def locked(self):
        return self._locked


class _TrackedSlotPool:
    def __init__(self, events, label):
        self.events = events
        self.label = label
        self.in_use = 0

    async def acquire(self):
        self.events.append(f"{self.label}.acquire")
        self.in_use += 1
        return 7

    async def release(self, worker_id):
        self.events.append(f"{self.label}.release:{worker_id}")
        if worker_id is not None and self.in_use > 0:
            self.in_use -= 1


class _TrackedScheduler:
    def __init__(self, events):
        self.events = events

    async def submit_async_nonblocking(self, exec_fname, results_dir):
        self.events.append(f"submit:{exec_fname}:{results_dir}")
        return "job-123"


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
    runner.postprocess_slot_pool = overrides.get("postprocess_slot_pool", _FakeSlotPool())
    runner.sampling_slot_pool = overrides.get("sampling_slot_pool", _FakeSlotPool())
    runner.scheduler = overrides.get("scheduler", _FakeScheduler())
    runner.submitted_jobs = overrides.get("submitted_jobs", {})
    runner.slot_available = overrides.get("slot_available", _FakeEvent())
    runner.should_stop = overrides.get("should_stop", _FakeEvent())
    runner.finalization_complete = overrides.get("finalization_complete", _FakeEvent())
    runner.max_evaluation_jobs = overrides.get("max_evaluation_jobs", 2)
    runner.max_proposal_jobs = overrides.get("max_proposal_jobs", 1)
    runner.max_db_workers = overrides.get("max_db_workers", 1)
    runner.total_api_cost = overrides.get("total_api_cost", 0.0)
    runner.completed_proposal_costs = overrides.get("completed_proposal_costs", [])
    runner.avg_proposal_cost = overrides.get("avg_proposal_cost", 0.0)
    runner.processing_lock = overrides.get("processing_lock", _FakeLock())
    runner._evaluation_seconds_ewma = overrides.get("evaluation_ewma")
    runner.verbose = overrides.get("verbose", False)
    runner.cost_limit_reached = overrides.get("cost_limit_reached", False)
    runner.lang_ext = overrides.get("lang_ext", "py")
    runner.llm_selection = overrides.get("llm_selection")
    runner.meta_summarizer = overrides.get("meta_summarizer")
    runner.results_dir = overrides.get("results_dir", ".")
    runner.prompt_db = overrides.get("prompt_db")
    runner._prompt_percentile_recompute_task = overrides.get(
        "_prompt_percentile_recompute_task"
    )
    runner._prompt_percentile_recompute_pending = overrides.get(
        "_prompt_percentile_recompute_pending", False
    )
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


def test_get_in_flight_work_count_includes_completed_job_processing_lock():
    async def _run():
        runner = _build_runner(
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
        )

        assert runner._get_in_flight_work_count() == 0

        await runner.processing_lock.acquire()
        try:
            assert runner._get_in_flight_work_count() == 1
        finally:
            runner.processing_lock.release()

    asyncio.run(_run())


def test_job_monitor_stops_when_target_reached_with_no_running_jobs():
    async def _run():
        runner = _build_runner(
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
            completed_generations=50,
            evo_config=SimpleNamespace(num_generations=50, max_api_costs=None),
            verbose=True,
        )
        runner._has_persistence_work_in_progress = lambda: False
        runner._cancel_surplus_inflight_work = lambda: asyncio.sleep(0, result=None)
        runner._retry_failed_db_jobs = lambda: asyncio.sleep(0, result=None)

        await runner._job_monitor_task()

        assert runner.should_stop.is_set() is True
        assert runner.finalization_complete.is_set() is True

    asyncio.run(_run())


def test_is_system_stuck_ignores_inflight_persistence_work():
    async def _run():
        runner = _build_runner(
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
            completed_generations=4,
            evo_config=SimpleNamespace(num_generations=10),
        )

        await runner.processing_lock.acquire()
        try:
            assert runner._is_system_stuck() is False
        finally:
            runner.processing_lock.release()

    asyncio.run(_run())


def test_is_system_stuck_ignores_retry_queue_work():
    runner = _build_runner(
        running_jobs=[],
        active_proposal_tasks={},
        failed_jobs_for_retry={"retry-1": object()},
        completed_generations=4,
        evo_config=SimpleNamespace(num_generations=10),
    )

    assert runner._is_system_stuck() is False


def test_get_remaining_generation_slots_uses_hard_generation_budget():
    runner = _build_runner(
        evo_config=SimpleNamespace(num_generations=5),
        next_generation_to_submit=4,
    )

    assert runner._get_remaining_generation_slots() == 1


def test_persist_failed_generation_stores_incorrect_program(tmp_path):
    async def _run():
        async_db = _RecordingAsyncDB()
        slot_event = _FakeEvent()
        runner = _build_runner(
            async_db=async_db,
            slot_available=slot_event,
            postprocess_slot_pool=_FakeSlotPool(),
            evo_config=SimpleNamespace(num_generations=5),
        )
        runner._record_progress = lambda: None

        async def _update_completed_generations():
            runner.completed_generations = await runner._count_completed_generations_from_db()

        async def _persist_program_metadata_async(_program):
            return None

        runner._update_completed_generations = _update_completed_generations
        runner._persist_program_metadata_async = _persist_program_metadata_async

        gen_dir = tmp_path / "gen_3"
        gen_dir.mkdir()
        exec_path = gen_dir / "main.py"
        exec_path.write_text("print('failed candidate')\n")

        program = await runner._persist_failed_generation(
            generation=3,
            exec_fname=str(exec_path),
            proposal_started_at=time.time(),
            sampling_worker_id=5,
            active_proposals_at_start=2,
            parent_program=SimpleNamespace(id="parent-1"),
            archive_programs=[SimpleNamespace(id="archive-1")],
            top_k_programs=[SimpleNamespace(id="topk-1")],
            code_diff="diff",
            meta_patch_data={"api_costs": 0.25, "system_prompt_id": "prompt-1"},
            code_embedding=[0.1, 0.2],
            embed_cost=0.01,
            novelty_cost=0.02,
            api_costs=0.25,
            failure_stage="proposal_failed",
            failure_reason="LLM failed to generate a valid proposal",
        )

        assert program is not None
        assert program.correct is False
        assert program.combined_score == 0.0
        assert program.text_feedback == "LLM failed to generate a valid proposal"
        assert program.code == "print('failed candidate')\n"
        assert program.metadata["node_kind"] == "failed_proposal"
        assert program.metadata["failure_stage"] == "proposal_failed"
        assert program.metadata["failure_class"] == "proposal_generation_failed"
        assert program.metadata["failure_persisted"] is True
        assert program.metadata["downstream_eval_submitted"] is False
        assert program.metadata["failure_json_path"] == str(gen_dir / "failure.json")
        assert runner.completed_generations == 1
        assert slot_event.is_set() is True
        assert len(async_db.programs) == 1

        failure_payload = json.loads((gen_dir / "failure.json").read_text())
        assert failure_payload["generation"] == 3
        assert failure_payload["node_kind"] == "failed_proposal"
        assert failure_payload["language"] == "python"
        assert failure_payload["failure_stage"] == "proposal_failed"
        assert failure_payload["failure_class"] == "proposal_generation_failed"
        assert failure_payload["failure_reason"] == "LLM failed to generate a valid proposal"
        assert failure_payload["artifacts"]["generated_code_path"] == str(exec_path)
        assert failure_payload["downstream_eval_submitted"] is False

    asyncio.run(_run())


def test_generate_evolved_proposal_records_failed_node_attempt_after_pre_eval_failure(
    tmp_path,
):
    async def _run():
        async_db = _RecordingAsyncDB()
        runner = _build_runner(
            async_db=async_db,
            postprocess_slot_pool=_FakeSlotPool(),
            evo_config=SimpleNamespace(
                max_novelty_attempts=1,
                max_patch_resamples=1,
                num_generations=5,
            ),
        )
        runner._record_progress = lambda: None
        runner.novelty_judge = None

        async def _update_completed_generations():
            runner.completed_generations = await runner._count_completed_generations_from_db()

        async def _persist_program_metadata_async(_program):
            return None

        async def _sample_with_fix_mode_async(**_kwargs):
            return (
                SimpleNamespace(id="parent-1", generation=1, island_idx=None),
                [SimpleNamespace(id="archive-1")],
                [SimpleNamespace(id="topk-1")],
                False,
            )

        async def _run_patch_async(*_args, **_kwargs):
            return (
                None,
                {
                    "api_costs": 0.1,
                    "patch_type": "diff",
                    "patch_name": "broken_patch",
                    "patch_description": "fails before evaluation",
                    "patch_attempt": 1,
                    "resample_attempt": 1,
                    "novelty_attempt": 1,
                    "last_error_msg": "No changes applied",
                    "error_attempt": "Max attempts reached without successful patch",
                },
                False,
            )

        runner._update_completed_generations = _update_completed_generations
        runner._persist_program_metadata_async = _persist_program_metadata_async
        runner.async_db.sample_with_fix_mode_async = _sample_with_fix_mode_async
        runner._run_patch_async = _run_patch_async

        gen_dir = tmp_path / "gen_4"
        gen_dir.mkdir()
        exec_path = gen_dir / "main.py"

        result = await runner._generate_evolved_proposal(
            generation=4,
            task_id="proposal-4",
            exec_fname=str(exec_path),
            results_dir=str(gen_dir / "results"),
            meta_recs=None,
            meta_summary=None,
            meta_scratch=None,
            proposal_started_at=time.time(),
            sampling_worker_id=None,
            active_proposals_at_start=1,
        )

        assert result is None
        assert runner.total_api_cost == 0.1
        assert runner.completed_generations == 0
        assert len(async_db.programs) == 0
        assert len(async_db.attempt_events) == 1
        event = async_db.attempt_events[0]
        assert event["generation"] == 4
        assert event["stage"] == "proposal"
        assert event["status"] == "failed"
        assert event["details"]["node_kind"] == "failed_proposal"
        assert event["details"]["failure_stage"] == "proposal"
        assert event["details"]["failure_class"] == "patch_apply_failed"
        assert event["details"]["failure_reason"] == "No changes applied"
        assert event["details"]["pipeline_started_at"] is not None
        assert event["details"]["sampling_started_at"] is not None
        assert event["details"]["sampling_finished_at"] is not None
        assert event["details"]["evaluation_started_at"] is not None
        assert event["details"]["evaluation_finished_at"] is not None
        assert event["details"]["postprocess_started_at"] is not None
        assert event["details"]["postprocess_finished_at"] is not None

        failure_payload = json.loads((gen_dir / "failure.json").read_text())
        assert failure_payload["language"] == "python"
        assert failure_payload["failure_stage"] == "proposal"
        assert failure_payload["failure_class"] == "patch_apply_failed"
        assert failure_payload["failure_reason"] == "No changes applied"

    asyncio.run(_run())


def test_persist_failed_generation_skips_maintenance_and_handles_missing_code(tmp_path):
    async def _run():
        async_db = _RecordingAsyncDB()
        runner = _build_runner(
            async_db=async_db,
            postprocess_slot_pool=_FakeSlotPool(),
            evo_config=SimpleNamespace(num_generations=5),
        )
        runner._record_progress = lambda: None

        async def _update_completed_generations():
            runner.completed_generations = await runner._count_completed_generations_from_db()

        async def _persist_program_metadata_async(_program):
            raise AssertionError("failed proposal persistence should not perform metadata rewrite")

        runner._update_completed_generations = _update_completed_generations
        runner._persist_program_metadata_async = _persist_program_metadata_async

        gen_dir = tmp_path / "gen_4"
        gen_dir.mkdir()
        exec_path = gen_dir / "main.py"

        program = await runner._persist_failed_generation(
            generation=4,
            exec_fname=str(exec_path),
            proposal_started_at=time.time(),
            sampling_worker_id=None,
            active_proposals_at_start=1,
            parent_program=SimpleNamespace(id="parent-1"),
            archive_programs=[],
            top_k_programs=[],
            code_diff=None,
            meta_patch_data={
                "api_costs": 0.05,
                "patch_type": "full",
                "patch_name": "broken",
                "error_attempt": "Max attempts reached without successful patch",
                "last_error_msg": "Could not extract code from patch string",
            },
            code_embedding=None,
            embed_cost=0.0,
            novelty_cost=0.0,
            api_costs=0.05,
            failure_stage="proposal",
            failure_reason="Could not extract code from patch string",
        )

        assert program is not None
        assert program.code == ""
        assert async_db.maintenance_calls == 0
        assert program.metadata["postprocess_worker_id"] is None
        assert program.metadata["failure_json_path"] == str(gen_dir / "failure.json")

    asyncio.run(_run())


def test_record_terminal_failed_proposal_updates_total_api_cost_once(tmp_path):
    async def _run():
        async_db = _RecordingAsyncDB()
        runner = _build_runner(async_db=async_db, total_api_cost=1.25)

        gen_dir = tmp_path / "gen_9"
        gen_dir.mkdir()
        exec_path = gen_dir / "main.py"

        await runner._record_terminal_failed_proposal(
            generation=9,
            exec_fname=str(exec_path),
            proposal_started_at=time.time(),
            sampling_worker_id=None,
            active_proposals_at_start=2,
            parent_program=SimpleNamespace(id="parent-9"),
            archive_programs=[],
            top_k_programs=[],
            code_diff=None,
            meta_patch_data={
                "api_costs": 0.06,
                "novelty_attempt": 1,
                "resample_attempt": 1,
                "patch_attempt": 1,
            },
            code_embedding=None,
            embed_cost=0.01,
            novelty_cost=0.02,
            api_costs=0.06,
            failure_stage="proposal",
            failure_reason="Could not extract code from patch string",
        )

        assert runner.total_api_cost == 1.34
        assert len(async_db.attempt_events) == 1

    asyncio.run(_run())


def test_record_terminal_failed_proposal_updates_avg_proposal_cost(tmp_path):
    async def _run():
        async_db = _RecordingAsyncDB()
        runner = _build_runner(
            async_db=async_db,
            total_api_cost=0.5,
            completed_proposal_costs=[0.3],
            avg_proposal_cost=0.3,
        )

        gen_dir = tmp_path / "gen_10"
        gen_dir.mkdir()
        exec_path = gen_dir / "main.py"

        await runner._record_terminal_failed_proposal(
            generation=10,
            exec_fname=str(exec_path),
            proposal_started_at=time.time(),
            sampling_worker_id=None,
            active_proposals_at_start=1,
            parent_program=SimpleNamespace(id="parent-10"),
            archive_programs=[],
            top_k_programs=[],
            code_diff=None,
            meta_patch_data={
                "api_costs": 0.06,
                "novelty_attempt": 1,
                "resample_attempt": 1,
                "patch_attempt": 1,
            },
            code_embedding=None,
            embed_cost=0.01,
            novelty_cost=0.02,
            api_costs=0.06,
            failure_stage="proposal",
            failure_reason="Could not extract code from patch string",
        )

        assert runner.total_api_cost == 0.59
        assert runner.completed_proposal_costs == [0.3, 0.09]
        assert runner.avg_proposal_cost == pytest.approx(0.195)

    asyncio.run(_run())


def test_maybe_evolve_prompt_updates_total_api_cost():
    async def _run():
        runner = _build_runner(
            total_api_cost=1.0,
            evo_config=SimpleNamespace(
                evolve_prompts=True,
                prompt_evolution_interval=1,
                prompt_evo_top_k_programs=3,
                language="python",
                use_text_feedback=False,
            ),
            prompt_db=SimpleNamespace(last_generation=2, add=lambda *args, **kwargs: None),
        )
        runner.prompt_evolution_counter = 0
        runner.prompt_sampler_evo = SimpleNamespace(
            sample=lambda: SimpleNamespace(id="prompt-parent")
        )
        runner.prompt_evolver = SimpleNamespace(
            evolve=lambda **kwargs: asyncio.sleep(
                0, result=(SimpleNamespace(id="prompt-new", generation=3), "diff", 0.25)
            )
        )
        runner.async_db.get_top_programs_async = lambda n: asyncio.sleep(0, result=[])
        runner.meta_summarizer = None
        runner.prompt_api_cost = 0.0
        runner.verbose = False

        await runner._maybe_evolve_prompt()

        assert runner.prompt_api_cost == 0.25
        assert runner.total_api_cost == 1.25

    asyncio.run(_run())


def test_process_single_job_safely_applies_side_effects_inline():
    async def _run():
        runner = _build_runner()
        job = AsyncRunningJob(
            job_id="job-side-effects",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=9,
        )
        event = PersistedProgramEvent(
            job=job,
            program=SimpleNamespace(id="program-1"),
            evaluation_finished_at=1.0,
            postprocess_started_at=2.0,
            postprocess_finished_at=3.0,
        )
        applied = []

        async def persist_completed_job(_job):
            return CompletedJobPersistResult(job=_job, success=True, persisted_event=event)

        async def enqueue_background_side_effects(_events):
            raise AssertionError("side effects should not be queued in inline mode")

        async def apply_persisted_program_side_effects(_persisted_event):
            applied.append(_persisted_event)

        runner._persist_completed_job = persist_completed_job
        runner._enqueue_background_side_effects = enqueue_background_side_effects
        runner._apply_persisted_program_side_effects = apply_persisted_program_side_effects

        success = await runner._process_single_job_safely(job)

        assert success is True
        assert applied == [event]

    asyncio.run(_run())


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
        assert slot_pool.released == []

    asyncio.run(_run())


def test_start_proposals_does_not_assign_generation_past_target():
    async def _run():
        runner = _build_runner(
            evo_config=SimpleNamespace(num_generations=5, max_api_costs=None),
            next_generation_to_submit=4,
            max_proposal_jobs=4,
        )

        async def _fake_generate(_generation, _task_id):
            await asyncio.sleep(0)
            return None

        runner._generate_proposal_async = _fake_generate

        await runner._start_proposals(3)

        assert runner.next_generation_to_submit == 5
        assert runner.assigned_generations == {4}
        assert len(runner.active_proposal_tasks) == 1

        await asyncio.gather(*runner.active_proposal_tasks.values(), return_exceptions=True)
        await runner._cleanup_completed_proposal_tasks()

    asyncio.run(_run())


def test_generate_evolved_proposal_returns_none_when_all_attempts_fail(tmp_path):
    async def _run():
        runner = _build_runner(
            async_db=SimpleNamespace(
                sample_with_fix_mode_async=lambda **kwargs: asyncio.sleep(
                    0,
                    result=(
                        SimpleNamespace(id="parent-1", generation=0, combined_score=1.0),
                        [],
                        [],
                        False,
                    ),
                )
            ),
            db=SimpleNamespace(island_manager=None),
            evo_config=SimpleNamespace(
                num_generations=5,
                max_api_costs=None,
                max_novelty_attempts=1,
                max_patch_resamples=1,
            ),
        )
        runner.llm_selection = None
        runner.novelty_judge = None
        async def _run_patch_async(*args, **kwargs):
            return None

        runner._run_patch_async = _run_patch_async

        exec_path = tmp_path / "proposal_failed.py"
        exec_path.write_text("print('candidate')\n")

        result = await runner._generate_evolved_proposal(
            generation=4,
            task_id="task-1",
            exec_fname=str(exec_path),
            results_dir=str(tmp_path / "results"),
            meta_recs=None,
            meta_summary=None,
            meta_scratch=None,
            proposal_started_at=time.time(),
            sampling_worker_id=1,
            active_proposals_at_start=1,
        )

        assert result is None

    asyncio.run(_run())


def test_generate_evolved_proposal_returns_none_when_submit_fails(tmp_path):
    async def _run():
        class _FailingSubmitScheduler:
            async def submit_async_nonblocking(self, exec_fname, results_dir):
                raise RuntimeError("submit boom")

        runner = _build_runner(
            async_db=SimpleNamespace(
                sample_with_fix_mode_async=lambda **kwargs: asyncio.sleep(
                    0,
                    result=(
                        SimpleNamespace(id="parent-1", generation=0, combined_score=1.0),
                        [],
                        [],
                        False,
                    ),
                )
            ),
            db=SimpleNamespace(island_manager=None),
            evo_config=SimpleNamespace(
                num_generations=5,
                max_api_costs=None,
                max_novelty_attempts=1,
                max_patch_resamples=1,
            ),
            scheduler=_FailingSubmitScheduler(),
        )
        runner.llm_selection = None
        runner.novelty_judge = None
        runner._get_code_embedding_async = lambda _path: asyncio.sleep(
            0, result=([0.1], 0.01)
        )

        async def _run_patch_async(*args, **kwargs):
            return ("diff", {"api_costs": 0.2, "system_prompt_id": "prompt-1"}, True)

        runner._run_patch_async = _run_patch_async

        exec_path = tmp_path / "submit_failed.py"
        exec_path.write_text("print('candidate')\n")

        result = await runner._generate_evolved_proposal(
            generation=4,
            task_id="task-1",
            exec_fname=str(exec_path),
            results_dir=str(tmp_path / "results"),
            meta_recs=None,
            meta_summary=None,
            meta_scratch=None,
            proposal_started_at=time.time(),
            sampling_worker_id=1,
            active_proposals_at_start=1,
        )

        assert result is None

    asyncio.run(_run())


def test_generate_evolved_proposal_assigns_worker_ids_on_submit(tmp_path):
    async def _run():
        events = []
        runner = _build_runner(
            async_db=SimpleNamespace(
                sample_with_fix_mode_async=lambda **kwargs: asyncio.sleep(
                    0,
                    result=(
                        SimpleNamespace(id="parent-1", generation=0, combined_score=1.0),
                        [],
                        [],
                        False,
                    ),
                )
            ),
            db=SimpleNamespace(island_manager=None),
            evo_config=SimpleNamespace(
                num_generations=5,
                max_api_costs=None,
                max_novelty_attempts=1,
                max_patch_resamples=1,
            ),
            scheduler=_TrackedScheduler(events),
        )
        runner.llm_selection = None
        runner.novelty_judge = None
        runner._get_code_embedding_async = lambda _path: asyncio.sleep(
            0, result=([0.1], 0.01)
        )

        async def _run_patch_async(*args, **kwargs):
            return ("diff", {"api_costs": 0.2, "system_prompt_id": "prompt-1"}, True)

        runner._run_patch_async = _run_patch_async

        exec_path = tmp_path / "submit_success.py"
        exec_path.write_text("print('candidate')\n")
        results_dir = str(tmp_path / "results")

        result = await runner._generate_evolved_proposal(
            generation=4,
            task_id="task-1",
            exec_fname=str(exec_path),
            results_dir=results_dir,
            meta_recs=None,
            meta_summary=None,
            meta_scratch=None,
            proposal_started_at=time.time(),
            sampling_worker_id=3,
            active_proposals_at_start=1,
        )

        assert result is not None
        assert result.job_id == "job-123"
        assert result.sampling_worker_id == 3
        assert result.evaluation_worker_id == 0
        assert result.running_eval_jobs_at_submit == 1
        assert events == [f"submit:{exec_path}:{results_dir}"]

    asyncio.run(_run())


def test_generation_budget_exhaustion_waits_for_completed_job_processing_to_finish():
    async def _run():
        runner = _build_runner(
            evo_config=SimpleNamespace(num_generations=100, max_api_costs=None),
            completed_generations=92,
            next_generation_to_submit=100,
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
        )
        runner.cost_limit_reached = False
        recorded_events = []

        async def _record_generation_event(**kwargs):
            recorded_events.append(kwargs)

        async def _wait_for_slot_or_stop(timeout):
            runner.should_stop.set()

        runner._record_generation_event = _record_generation_event
        runner._wait_for_slot_or_stop = _wait_for_slot_or_stop
        runner._cleanup_completed_proposal_tasks = lambda: asyncio.sleep(
            0, result=None
        )
        runner._is_system_stuck = lambda: False
        runner._handle_stuck_system = lambda: asyncio.sleep(0, result=True)
        runner._compute_proposal_pipeline_target = lambda: 0
        runner._log_proposal_target_decision = lambda target: None
        runner._get_committed_cost = lambda: 0.0

        await runner.processing_lock.acquire()
        try:
            await runner._proposal_coordinator_task()
        finally:
            runner.processing_lock.release()

        assert recorded_events == []
        assert runner.finalization_complete.is_set() is False

    asyncio.run(_run())


def test_retry_failed_db_jobs_refreshes_completion_progress():
    async def _run():
        slot_event = _FakeEvent()
        runner = _build_runner(
            slot_available=slot_event,
            failed_jobs_for_retry={},
            submitted_jobs={},
        )
        runner.MAX_DB_RETRY_ATTEMPTS = 3

        update_calls = 0
        progress_calls = 0
        async def _process_single_job_safely(_job):
            return True

        async def _update_completed_generations():
            nonlocal update_calls
            update_calls += 1

        def _record_progress():
            nonlocal progress_calls
            progress_calls += 1

        runner._process_single_job_safely = _process_single_job_safely
        runner._update_completed_generations = _update_completed_generations
        runner._record_progress = _record_progress
        job = AsyncRunningJob(
            job_id="job-retry",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=7,
        )
        runner.failed_jobs_for_retry = {"job-retry": job}
        runner.submitted_jobs = {"job-retry": job}

        await runner._retry_failed_db_jobs()

        assert update_calls == 1
        assert progress_calls == 1
        assert slot_event.is_set() is True
        assert runner.failed_jobs_for_retry == {}
        assert runner.submitted_jobs == {}

    asyncio.run(_run())


def test_update_prompt_fitness_only_recomputes_after_correct_programs():
    class _FakePromptDB:
        def __init__(self):
            self.update_calls = []
            self.recompute_calls = []

        def update_fitness(self, **kwargs):
            self.update_calls.append(kwargs)

        def recompute_all_percentiles(self, all_correct_scores, program_id_to_score):
            self.recompute_calls.append((list(all_correct_scores), dict(program_id_to_score)))

    class _FakeAsyncDB:
        async def compute_percentile_async(self, program_score, correct_only=True):
            assert correct_only is True
            return 0.75

    async def _run():
        prompt_db = _FakePromptDB()
        runner = _build_runner(
            async_db=_FakeAsyncDB(),
            db=SimpleNamespace(
                get_all_programs=lambda: [
                    SimpleNamespace(id="p0", correct=True, combined_score=1.0),
                    SimpleNamespace(id="p1", correct=True, combined_score=2.0),
                ]
            ),
            evo_config=SimpleNamespace(
                num_generations=10,
                prompt_percentile_recompute_interval=2,
            ),
        )
        runner.prompt_db = prompt_db
        runner.prompt_percentile_recompute_counter = 0

        await runner._update_prompt_fitness(
            prompt_id="prompt-1",
            program_id="prog-incorrect",
            program_score=0.0,
            improvement=0.0,
            correct=False,
        )
        await runner._update_prompt_fitness(
            prompt_id="prompt-1",
            program_id="prog-correct-1",
            program_score=2.5,
            improvement=0.1,
            correct=True,
        )
        await runner._update_prompt_fitness(
            prompt_id="prompt-1",
            program_id="prog-correct-2",
            program_score=2.8,
            improvement=0.2,
            correct=True,
        )
        if runner._prompt_percentile_recompute_task is not None:
            await runner._prompt_percentile_recompute_task

        assert len(prompt_db.update_calls) == 3
        assert len(prompt_db.recompute_calls) == 1
        assert runner.prompt_percentile_recompute_counter == 0

    asyncio.run(_run())


def test_get_missing_persisted_generations_reports_budget_gap():
    class _FakeAsyncDB:
        async def get_persisted_generation_ids_async(self):
            return {0, 1, 3, 4}

    async def _run():
        runner = _build_runner(
            async_db=_FakeAsyncDB(),
            evo_config=SimpleNamespace(num_generations=5),
        )

        missing = await runner._get_missing_persisted_generations()

        assert missing == [2]

    asyncio.run(_run())


def test_get_generations_without_program_due_to_proposal_failure_uses_attempt_log():
    class _FakeCursor:
        def __init__(self):
            self.executed = None

        def execute(self, query, params):
            self.executed = (query, params)

        def fetchall(self):
            return [(2,), (5,), (8,)]

    runner = _build_runner(
        db=SimpleNamespace(cursor=_FakeCursor()),
        evo_config=SimpleNamespace(num_generations=100),
    )

    assert runner._get_generations_without_program_due_to_proposal_failure() == [
        2,
        5,
        8,
    ]


def test_submit_evaluation_job_acquires_slot_before_submitting():
    async def _run():
        events = []
        eval_pool = _TrackedSlotPool(events, "eval")
        sampling_pool = _TrackedSlotPool(events, "sampling")
        runner = _build_runner(
            scheduler=_TrackedScheduler(events),
            evaluation_slot_pool=eval_pool,
            sampling_slot_pool=sampling_pool,
        )

        (
            job_id,
            evaluation_worker_id,
            _evaluation_submitted_at,
            _evaluation_started_at,
            running_eval_jobs_at_submit,
        ) = await runner._submit_evaluation_job_with_slot(
            exec_fname="candidate.py",
            results_dir="results-dir",
            sampling_worker_id=3,
        )

        assert events == [
            "sampling.release:3",
            "eval.acquire",
            "submit:candidate.py:results-dir",
        ]
        assert job_id == "job-123"
        assert evaluation_worker_id == 7
        assert running_eval_jobs_at_submit == 1

    asyncio.run(_run())


def test_generate_evolved_proposal_uses_slot_reservation_helper():
    async def _run():
        helper_calls = []

        async def sample_with_fix_mode_async(**kwargs):
            return SimpleNamespace(id="parent", generation=0), [], [], False

        async def submit_with_slot(exec_fname, results_dir, sampling_worker_id):
            helper_calls.append((exec_fname, results_dir, sampling_worker_id))
            return "job-123", 9, 10.0, 10.0, 1

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("should reserve the slot via helper")

        runner = _build_runner(
            async_db=SimpleNamespace(
                sample_with_fix_mode_async=sample_with_fix_mode_async
            ),
            scheduler=SimpleNamespace(submit_async_nonblocking=fail_if_called),
            active_proposal_tasks={"task-1": object()},
            evo_config=SimpleNamespace(
                max_novelty_attempts=1,
                max_patch_resamples=1,
            ),
            db_config=SimpleNamespace(
                num_islands=1,
                parent_selection_strategy="",
            ),
        )
        runner.novelty_judge = None
        runner.llm_selection = None
        runner.total_api_cost = 0.0
        runner._update_avg_proposal_cost = lambda cost: None
        runner.slot_available = _FakeEvent()
        runner.submitted_jobs = {}
        runner.running_jobs = []
        runner._submit_evaluation_job_with_slot = submit_with_slot

        async def run_patch_async(*args, **kwargs):
            return "diff", {"api_costs": 0.0}, True

        async def get_code_embedding_async(exec_fname):
            return [0.1], 0.0

        runner._run_patch_async = run_patch_async
        runner._get_code_embedding_async = get_code_embedding_async

        job = await runner._generate_evolved_proposal(
            generation=4,
            task_id="task-1",
            exec_fname="program.py",
            results_dir="results",
            meta_recs=None,
            meta_summary=None,
            meta_scratch=None,
            proposal_started_at=time.time() - 1.0,
            sampling_worker_id=None,
            active_proposals_at_start=1,
        )

        assert helper_calls == [("program.py", "results", None)]
        assert job is not None
        assert job.evaluation_worker_id == 9
        assert job.running_eval_jobs_at_submit == 1
        assert runner.running_jobs == [job]
        assert runner.submitted_jobs == {"job-123": job}

    asyncio.run(_run())


def test_release_evaluation_slot_once_does_not_free_reassigned_slot():
    async def _run():
        runner = _build_runner(evaluation_slot_pool=LogicalSlotPool(1, "evaluation"))
        old_job = AsyncRunningJob(
            job_id="job-old",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            evaluation_started_at=time.time(),
            generation=1,
            evaluation_worker_id=1,
        )

        first_slot = await runner.evaluation_slot_pool.acquire()
        assert first_slot == 1

        await runner._release_evaluation_slot_once(old_job)
        reassigned_slot = await runner.evaluation_slot_pool.acquire()
        assert reassigned_slot == 1

        await runner._release_evaluation_slot_once(old_job)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runner.evaluation_slot_pool.acquire(), timeout=0.01)

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


def test_cancel_surplus_inflight_work_cancels_backlog_once_target_hit():
    async def _run():
        scheduler = _FakeScheduler(cancelled_job_ids={"job-1", "job-2"})
        eval_pool = _FakeSlotPool()

        proposal_task = asyncio.create_task(asyncio.sleep(60))
        job_1 = AsyncRunningJob(
            job_id="job-1",
            exec_fname="program_1.py",
            results_dir="results_1",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            evaluation_started_at=time.time(),
            generation=51,
            evaluation_worker_id=3,
        )
        job_2 = AsyncRunningJob(
            job_id="job-2",
            exec_fname="program_2.py",
            results_dir="results_2",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            evaluation_started_at=time.time(),
            generation=52,
            evaluation_worker_id=4,
        )
        runner = _build_runner(
            scheduler=scheduler,
            evaluation_slot_pool=eval_pool,
            running_jobs=[job_1, job_2],
            active_proposal_tasks={"proposal-1": proposal_task},
            assigned_generations={51, 52, 53},
            submitted_jobs={"job-1": job_1, "job-2": job_2},
        )

        await runner._cancel_surplus_inflight_work()

        assert proposal_task.cancelled()
        assert runner.running_jobs == []
        assert runner.active_proposal_tasks == {}
        assert runner.submitted_jobs == {}
        assert runner.assigned_generations == set()
        assert scheduler.cancelled_job_ids == ["job-1", "job-2"]
        assert eval_pool.released == [3, 4]

    asyncio.run(_run())


def test_process_single_job_skips_persistence_for_discarded_surplus_job():
    async def _run():
        async_db = _FakeAsyncDBWithGuard()
        eval_pool = _FakeSlotPool()
        postprocess_pool = _FakeSlotPool()
        runner = _build_runner(
            async_db=async_db,
            evaluation_slot_pool=eval_pool,
            postprocess_slot_pool=postprocess_pool,
        )
        job = AsyncRunningJob(
            job_id="job-surplus",
            exec_fname="program.py",
            results_dir="results",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            evaluation_started_at=time.time(),
            generation=53,
            evaluation_worker_id=9,
            discard_if_completed=True,
        )

        success = await runner._process_single_job_safely(job)

        assert success is True
        assert async_db.source_job_checks == []
        assert async_db.add_calls == 0
        assert eval_pool.released == [9]
        assert postprocess_pool.released == [None]

    asyncio.run(_run())


def test_mark_surplus_completed_jobs_discards_batch_overflow_after_target():
    runner = _build_runner(
        evo_config=SimpleNamespace(num_generations=100),
        completed_generations=98,
    )
    completed_jobs = [
        AsyncRunningJob(
            job_id="job-101",
            exec_fname="program_101.py",
            results_dir="results_101",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=101,
        ),
        AsyncRunningJob(
            job_id="job-99",
            exec_fname="program_99.py",
            results_dir="results_99",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=99,
        ),
        AsyncRunningJob(
            job_id="job-100",
            exec_fname="program_100.py",
            results_dir="results_100",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=100,
        ),
        AsyncRunningJob(
            job_id="job-102",
            exec_fname="program_102.py",
            results_dir="results_102",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=102,
        ),
    ]

    runner._mark_surplus_completed_jobs_for_discard(completed_jobs)

    kept_generations = [
        job.generation for job in completed_jobs if not job.discard_if_completed
    ]
    discarded_generations = [
        job.generation for job in completed_jobs if job.discard_if_completed
    ]

    assert kept_generations == [99, 100]
    assert discarded_generations == [101, 102]


def test_mark_surplus_completed_jobs_discards_all_when_target_already_reached():
    runner = _build_runner(
        evo_config=SimpleNamespace(num_generations=100),
        completed_generations=100,
    )
    completed_jobs = [
        AsyncRunningJob(
            job_id="job-101",
            exec_fname="program_101.py",
            results_dir="results_101",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=101,
        ),
        AsyncRunningJob(
            job_id="job-102",
            exec_fname="program_102.py",
            results_dir="results_102",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=102,
        ),
    ]

    runner._mark_surplus_completed_jobs_for_discard(completed_jobs)

    assert all(job.discard_if_completed for job in completed_jobs)


def test_job_monitor_preserves_jobs_added_during_status_poll():
    async def _run():
        runner = _build_runner(
            evo_config=SimpleNamespace(num_generations=10, max_api_costs=None),
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
        )
        runner.cost_limit_reached = False
        runner._has_persistence_work_in_progress = lambda: False
        runner._is_job_hung = lambda job: False
        runner._cancel_surplus_inflight_work = lambda: asyncio.sleep(0, result=None)
        runner._retry_failed_db_jobs = lambda: asyncio.sleep(0, result=None)
        runner._record_progress = lambda: None

        first_job = AsyncRunningJob(
            job_id="job-1",
            exec_fname="program_1.py",
            results_dir="results_1",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=1,
        )
        second_job = AsyncRunningJob(
            job_id="job-2",
            exec_fname="program_2.py",
            results_dir="results_2",
            start_time=time.time(),
            proposal_started_at=time.time(),
            evaluation_submitted_at=time.time(),
            generation=2,
        )
        runner.running_jobs = [first_job]

        class _RaceScheduler:
            def __init__(self):
                self.calls = 0

            async def batch_check_status_async(self, jobs):
                self.calls += 1
                original_job_count = len(jobs)
                if self.calls == 1:
                    runner.running_jobs.append(second_job)
                    runner.should_stop.set()
                return [True for _ in range(original_job_count)]

        runner.scheduler = _RaceScheduler()

        await runner._job_monitor_task()

        assert runner.running_jobs == [first_job, second_job]

    asyncio.run(_run())


def test_job_monitor_processes_completed_jobs_inline():
    async def _run():
        runner = _build_runner(
            evo_config=SimpleNamespace(num_generations=10, max_api_costs=None),
            running_jobs=[],
            active_proposal_tasks={},
            failed_jobs_for_retry={},
            evaluation_slot_pool=LogicalSlotPool(1, "evaluation"),
            verbose=False,
        )
        runner.processing_lock = asyncio.Lock()
        runner._cancel_surplus_inflight_work = lambda: asyncio.sleep(0, result=None)
        runner._retry_failed_db_jobs = lambda: asyncio.sleep(0, result=None)
        runner._record_progress = lambda: None
        runner._mark_surplus_completed_jobs_for_discard = lambda jobs: None

        batch_started = asyncio.Event()
        allow_finish = asyncio.Event()

        async def slow_process(completed_jobs):
            batch_started.set()
            await allow_finish.wait()

        runner._process_completed_jobs_safely = slow_process

        now = time.time()
        job = AsyncRunningJob(
            job_id="job-complete",
            exec_fname="program.py",
            results_dir="results",
            start_time=now - 3.0,
            proposal_started_at=now - 3.0,
            evaluation_submitted_at=now - 1.0,
            evaluation_started_at=now - 1.0,
            evaluation_worker_id=1,
            generation=4,
        )
        await runner.evaluation_slot_pool.acquire()
        runner.running_jobs = [job]

        class _CompletedScheduler:
            async def batch_check_status_async(self, jobs):
                return [False for _ in jobs]

        runner.scheduler = _CompletedScheduler()

        monitor_task = asyncio.create_task(runner._job_monitor_task())
        await asyncio.wait_for(batch_started.wait(), timeout=0.2)

        assert runner.running_jobs == []
        assert runner.evaluation_slot_pool.in_use == 1

        runner.should_stop.set()
        allow_finish.set()
        await asyncio.wait_for(monitor_task, timeout=0.2)

    asyncio.run(_run())
