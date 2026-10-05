import asyncio
import copy
import pickle
import random
import threading
from pathlib import Path

import numpy as np
import pytest

import shinka.core.async_runner as async_runner_module
from shinka.core import checkpointing
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.checkpointing import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    UncleanCheckpointError,
    capture_rng_states,
    configuration_hashes,
    load_checkpoint,
    restore_rng_states,
    source_tree_identity,
    validate_checkpoint_identity,
    write_checkpoint,
)
from shinka.core.config import EvolutionConfig
from shinka.core.runtime_slots import LogicalSlotPool
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)
from shinka.launch import LocalJobConfig
from shinka.llm import FixedSampler


def _payload(marker: str = "current") -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": marker,
        "clean": True,
    }


def test_rng_round_trip_matches_uninterrupted_draws():
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    try:
        random.seed(1234)
        np.random.seed(1234)
        generator = np.random.default_rng(1234)
        state = capture_rng_states({"component": generator})

        expected = (
            [random.random() for _ in range(5)],
            np.random.random(5),
            generator.random(5),
        )
        random.random()
        np.random.random()
        generator.random()

        restore_rng_states(state, {"component": generator})

        assert [random.random() for _ in range(5)] == expected[0]
        assert np.array_equal(np.random.random(5), expected[1])
        assert np.array_equal(generator.random(5), expected[2])
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)


def test_rng_restore_rejects_bit_generator_change_before_mutating_globals():
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    try:
        generator = np.random.default_rng(9)
        state = capture_rng_states({"component": generator})
        state["named_generators"]["component"]["bit_generator"] = "wrong.Type"
        before_python = random.getstate()
        before_numpy = np.random.get_state()

        with pytest.raises(CheckpointCompatibilityError, match="BitGenerator"):
            restore_rng_states(state, {"component": generator})

        assert random.getstate() == before_python
        assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)


def test_atomic_writer_retains_previous_checkpoint(tmp_path):
    write_checkpoint(tmp_path, _payload("first"))
    write_checkpoint(tmp_path, _payload("second"))

    assert load_checkpoint(tmp_path).payload["checkpoint_id"] == "second"
    (tmp_path / CHECKPOINT_FILENAME).write_bytes(b"corrupt")
    loaded = load_checkpoint(tmp_path)
    assert loaded.path.name == checkpointing.PREVIOUS_CHECKPOINT_FILENAME
    assert loaded.payload["checkpoint_id"] == "first"


def test_interrupted_publish_leaves_current_checkpoint_valid(tmp_path, monkeypatch):
    write_checkpoint(tmp_path, _payload("first"))
    real_replace = checkpointing.os.replace

    def fail_final_replace(source, destination):
        if Path(destination).name == CHECKPOINT_FILENAME:
            raise OSError("simulated publication interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="interruption"):
        write_checkpoint(tmp_path, _payload("second"))

    assert load_checkpoint(tmp_path).payload["checkpoint_id"] == "first"


def test_checksum_mismatch_is_rejected(tmp_path):
    checkpoint_path = write_checkpoint(tmp_path, _payload())
    with checkpoint_path.open("rb") as handle:
        envelope = pickle.load(handle)
    envelope["payload_sha256"] = "0" * 64
    with checkpoint_path.open("wb") as handle:
        pickle.dump(envelope, handle)

    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_checkpoint(tmp_path)


def test_missing_checkpoint_is_rejected(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="No checkpoint.pkl"):
        load_checkpoint(tmp_path)


def test_identity_validation_rejects_schema_clean_source_config_and_runtime():
    source = {"tree": "source"}
    configs = {"evolution": "a", "database": "b", "job": "c"}
    runtime = {"python": "version"}
    base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "clean": True,
        "identity": {
            "source": source,
            "configuration_hashes": configs,
            "runtime": runtime,
        },
    }
    validate_checkpoint_identity(
        base,
        expected_source=source,
        expected_config_hashes=configs,
        expected_runtime=runtime,
    )

    cases = [
        ("schema_version", 999, CheckpointCompatibilityError),
        ("clean", False, UncleanCheckpointError),
    ]
    for key, value, error in cases:
        changed = copy.deepcopy(base)
        changed[key] = value
        with pytest.raises(error):
            validate_checkpoint_identity(
                changed,
                expected_source=source,
                expected_config_hashes=configs,
                expected_runtime=runtime,
            )

    for identity_key in ("source", "configuration_hashes", "runtime"):
        changed = copy.deepcopy(base)
        changed["identity"][identity_key] = {"changed": True}
        with pytest.raises(CheckpointCompatibilityError):
            validate_checkpoint_identity(
                changed,
                expected_source=source,
                expected_config_hashes=configs,
                expected_runtime=runtime,
            )


def test_configuration_hashes_ignore_location_observability_and_resume_policy(
    tmp_path,
):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    evo = EvolutionConfig(
        results_dir=str(tmp_path / "one"),
        init_program_path=None,
        random_seed=7,
        checkpoint_resume_mode="strict",
        wandb_project="first",
    )
    db = DatabaseConfig(db_path=str(tmp_path / "one" / "programs.sqlite"))
    job = LocalJobConfig(eval_program_path=str(evaluator))
    expected = configuration_hashes(evo, db, job)

    evo.results_dir = str(tmp_path / "two")
    evo.checkpoint_resume_mode = "reseed"
    evo.wandb_project = "second"
    db.db_path = str(tmp_path / "two" / "programs.sqlite")
    assert configuration_hashes(evo, db, job) == expected

    evo.random_seed = 8
    assert configuration_hashes(evo, db, job)["evolution"] != expected["evolution"]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"checkpoint_resume_mode": ""}, ValueError),
        ({"checkpoint_resume_mode": "best_effort"}, ValueError),
        ({"checkpoint_resume_mode": "STRICT"}, ValueError),
        ({"random_seed": -1}, ValueError),
        ({"random_seed": 2**32}, ValueError),
        ({"random_seed": True}, TypeError),
        ({"random_seed": 1.5}, TypeError),
        ({"random_seed": "7"}, TypeError),
    ],
)
def test_evolution_config_rejects_invalid_checkpoint_settings(kwargs, error):
    with pytest.raises(error, match="checkpoint_resume_mode|random_seed"):
        EvolutionConfig(**kwargs)


def test_source_tree_identity_changes_with_python_source(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    source_file = package / "module.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_identity(package)
    assert source_tree_identity(package) == first

    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_identity(package) != first


def _add_programs(db: ProgramDatabase) -> None:
    for generation in range(3):
        db.add(
            Program(
                id=f"program-{generation}",
                code=f"value = {generation}\n",
                generation=generation,
                correct=True,
                combined_score=float(generation),
                metadata={"api_costs": 0.1},
            ),
            defer_maintenance=True,
        )


def _build_checkpoint_runner(
    tmp_path: Path,
    *,
    initialize_database: bool,
) -> ShinkaEvolveRunner:
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    db_config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=1,
    )
    db = ProgramDatabase(db_config, embedding_model=None)
    if initialize_database:
        _add_programs(db)

    runner = object.__new__(ShinkaEvolveRunner)
    runner.results_dir = tmp_path
    runner.evo_config = EvolutionConfig(
        num_generations=8,
        results_dir=str(tmp_path),
        init_program_path=None,
        embedding_model=None,
        llm_models=["a", "b"],
        llm_dynamic_selection="fixed",
        llm_dynamic_selection_kwargs={"prior_probs": [0.4, 0.6]},
        random_seed=42,
        checkpoint_resume_mode="strict",
    )
    runner.db_config = db_config
    runner.job_config = LocalJobConfig(eval_program_path=str(evaluator))
    runner.db = db
    runner.async_db = AsyncProgramDatabase(db, max_workers=2)
    runner.prompt_db = None
    runner.llm_selection = FixedSampler(
        arm_names=["a", "b"], prior_probs=np.array([0.4, 0.6]), seed=42
    )
    runner._llm_selection_seed = 42
    runner.meta_summarizer = None
    runner.completed_generations = 3
    runner.next_generation_to_submit = 3
    runner.assigned_generations = set()
    runner.best_program_id = None
    runner.prompt_evolution_counter = 0
    runner.prompt_percentile_recompute_counter = 0
    runner.current_prompt_id = None
    runner.prompt_api_cost = 0.0
    runner.total_api_cost = 0.3
    runner.completed_proposal_costs = [0.1, 0.1]
    runner.avg_proposal_cost = 0.1
    runner.total_proposals_generated = 2
    runner._sampling_seconds_ewma = 1.5
    runner._evaluation_seconds_ewma = 2.5
    runner._proposal_timing_samples = 2
    runner.cost_limit_reached = False
    runner.running_jobs = []
    runner.active_proposal_tasks = {}
    runner.failed_jobs_for_retry = {}
    runner.submitted_jobs = {}
    runner.processing_lock = asyncio.Lock()
    runner._meta_side_effect_lock = asyncio.Lock()
    runner._prompt_side_effect_lock = asyncio.Lock()
    runner._best_solution_lock = asyncio.Lock()
    runner.sampling_slot_pool = LogicalSlotPool(2, "sampling")
    runner.evaluation_slot_pool = LogicalSlotPool(2, "evaluation")
    runner.postprocess_slot_pool = LogicalSlotPool(2, "postprocess")
    runner._completed_job_batch_tasks = set()
    runner._completed_jobs_pending = 0
    runner._background_side_effect_tasks = set()
    runner._background_side_effects_pending = 0
    runner._background_side_effects_busy = False
    runner._background_side_effects_busy_count = 0
    runner._prompt_percentile_recompute_task = None
    runner._prompt_percentile_recompute_pending = False
    runner.proposal_queue = asyncio.Queue()
    runner.side_effect_event_queue = asyncio.Queue()
    runner.pause_new_proposals = asyncio.Event()
    runner.pause_new_proposals.set()
    runner.checkpoint_complete = asyncio.Event()
    runner._checkpoint_published = False
    return runner


def _random_trace(runner: ShinkaEvolveRunner, count: int) -> list[tuple[int, int, int]]:
    trace = []
    for _ in range(count):
        model_one_hot, _ = runner.llm_selection.select_llm()
        trace.append(
            (
                random.randrange(1000),
                int(np.random.randint(1000)),
                int(np.argmax(model_one_hot)),
            )
        )
    return trace


def test_clean_checkpoint_resume_matches_uninterrupted_random_trace(tmp_path):
    async def run_test():
        original_python = random.getstate()
        original_numpy = np.random.get_state()
        source = _build_checkpoint_runner(tmp_path, initialize_database=True)
        target = None
        try:
            source._seed_random_streams()
            _random_trace(source, 5)
            payload = await source._publish_clean_checkpoint()
            expected = _random_trace(source, 12)
            await source.async_db.close_async()
            source.db.close()

            target = _build_checkpoint_runner(tmp_path, initialize_database=False)
            loaded = load_checkpoint(tmp_path)
            target._validate_checkpoint_payload(loaded.payload)
            target._restore_checkpoint_without_rng(loaded.payload)
            restore_rng_states(
                loaded.payload["random_streams"], target._owned_rng_generators()
            )

            assert loaded.payload["checkpoint_id"] == payload["checkpoint_id"]
            assert _random_trace(target, 12) == expected
            assert target.completed_generations == 3
            assert target.next_generation_to_submit == 3
        finally:
            if target is not None:
                await target.async_db.close_async()
                target.db.close()
            elif source.db.conn is not None:
                await source.async_db.close_async()
                source.db.close()
            random.setstate(original_python)
            np.random.set_state(original_numpy)

    asyncio.run(run_test())


def test_best_effort_global_reseed_preserves_loaded_bandit_position(tmp_path):
    runner = _build_checkpoint_runner(tmp_path, initialize_database=True)
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    try:
        runner.llm_selection.select_llm()
        expected_state = copy.deepcopy(runner.llm_selection.get_rng_state())

        runner._seed_random_streams(include_named=False)

        assert runner.llm_selection.get_rng_state() == expected_state
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        asyncio.run(runner.async_db.close_async())
        runner.db.close()


@pytest.mark.parametrize("kind", ["ahead", "behind", "metadata", "prompt"])
def test_strict_validation_rejects_database_watermark_mismatch(tmp_path, kind):
    async def run_test():
        runner = _build_checkpoint_runner(tmp_path, initialize_database=True)
        if kind == "prompt":
            runner.prompt_db = SystemPromptDatabase(
                SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
            )
            runner.prompt_db.add(
                create_system_prompt("initial prompt", generation=0, patch_type="init")
            )
        try:
            await runner._publish_clean_checkpoint()
            if kind == "ahead":
                runner.db.add(
                    Program(
                        id="advanced",
                        code="value = 3\n",
                        generation=3,
                        correct=True,
                    ),
                    defer_maintenance=True,
                )
            elif kind == "behind":
                runner.db.cursor.execute(
                    "DELETE FROM programs WHERE id = ?", ("program-2",)
                )
                runner.db.conn.commit()
            elif kind == "metadata":
                runner.db._update_metadata_in_db("best_program_id", "program-1")
            else:
                runner.prompt_db._update_metadata_in_db("best_prompt_id", "changed")
                runner.prompt_db.add(
                    create_system_prompt(
                        "changed prompt", generation=1, patch_type="full"
                    )
                )
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            with pytest.raises(
                CheckpointCompatibilityError, match="Database watermark"
            ):
                runner._validate_checkpoint_payload(load_checkpoint(tmp_path).payload)
            assert random.getstate() == before_python
            assert np.array_equal(np.random.get_state()[1], before_numpy[1])
        finally:
            if runner.prompt_db is not None:
                runner.prompt_db.close()
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run_test())


def test_clean_checkpoint_rejects_active_proposal(tmp_path):
    async def run_test():
        runner = _build_checkpoint_runner(tmp_path, initialize_database=True)
        runner.active_proposal_tasks["active"] = object()
        try:
            with pytest.raises(UncleanCheckpointError, match="active_proposals"):
                await runner._publish_clean_checkpoint()
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run_test())


def test_clean_checkpoint_drains_embedding_maintenance(tmp_path):
    async def run_test():
        runner = _build_checkpoint_runner(tmp_path, initialize_database=True)
        maintenance_finished = asyncio.Event()

        async def maintenance():
            await asyncio.sleep(0)
            maintenance_finished.set()

        runner.async_db._embedding_recompute_task = asyncio.create_task(maintenance())
        try:
            await runner._publish_clean_checkpoint()
            assert maintenance_finished.is_set()
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run_test())


def test_checkpoint_cleanup_bounds_wandb_shutdown(monkeypatch):
    async def run_test():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.active_proposal_tasks = {}
        runner._checkpoint_published = True
        runner.prompt_db = None
        runner.db = None
        cancelled = asyncio.Event()
        database_closed = asyncio.Event()
        scheduler_stopped = asyncio.Event()

        async def blocked_wandb_finish():
            await asyncio.Event().wait()

        async def cancel_wandb():
            cancelled.set()

        async def close_database():
            database_closed.set()

        runner._finish_wandb_logging = blocked_wandb_finish
        runner._cancel_wandb_tasks_without_waiting = cancel_wandb
        runner.async_db = type(
            "AsyncDB",
            (),
            {"close_async": staticmethod(close_database)},
        )()
        runner.scheduler = type(
            "Scheduler",
            (),
            {"shutdown": lambda _self: scheduler_stopped.set()},
        )()
        monkeypatch.setattr(
            async_runner_module,
            "_CHECKPOINT_WANDB_FINISH_TIMEOUT_SECONDS",
            0.01,
        )

        await runner._cleanup_async()

        assert cancelled.is_set()
        assert database_closed.is_set()
        assert scheduler_stopped.is_set()

    asyncio.run(run_test())


def test_async_database_flush_waits_for_all_workers(tmp_path):
    async def run_test():
        db = ProgramDatabase(
            DatabaseConfig(db_path=str(tmp_path / "flush.sqlite")),
            embedding_model=None,
        )
        async_db = AsyncProgramDatabase(db, max_workers=2)
        gate = threading.Event()
        completed = []
        loop = asyncio.get_running_loop()

        def queued_work(label):
            gate.wait(timeout=5)
            completed.append(label)

        try:
            pending = [
                loop.run_in_executor(async_db.write_executor, queued_work, f"write-{i}")
                for i in range(4)
            ] + [
                loop.run_in_executor(async_db.executor, queued_work, f"read-{i}")
                for i in range(4)
            ]
            flush_task = asyncio.create_task(async_db.flush_async())
            await asyncio.sleep(0)
            assert not flush_task.done()

            gate.set()
            await asyncio.wait_for(flush_task, timeout=5)
            await asyncio.gather(*pending)

            assert sorted(completed) == [
                "read-0",
                "read-1",
                "read-2",
                "read-3",
                "write-0",
                "write-1",
                "write-2",
                "write-3",
            ]
        finally:
            gate.set()
            await async_db.close_async()
            db.close()

    asyncio.run(run_test())


def test_prompt_setup_does_not_duplicate_generation_zero_on_resume(tmp_path):
    async def run_test():
        def build_runner():
            runner = object.__new__(ShinkaEvolveRunner)
            runner.results_dir = tmp_path
            runner.evo_config = EvolutionConfig(
                results_dir=str(tmp_path),
                evolve_prompts=True,
                task_sys_msg="initial prompt",
            )
            runner.prompt_llm = None
            runner.verbose = False
            return runner

        first = build_runner()
        await first._setup_prompt_evolution()
        assert first.prompt_db._count_prompts_in_db() == 1
        first.prompt_db.close()

        resumed = build_runner()
        await resumed._setup_prompt_evolution()
        assert resumed.prompt_db._count_prompts_in_db() == 1
        resumed.prompt_db.close()

    asyncio.run(run_test())
