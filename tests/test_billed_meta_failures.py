"""Failed meta and prompt calls reach both live and restored budgets."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import (
    AsyncRunningJob,
    PersistedProgramEvent,
    ShinkaEvolveRunner,
)
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)
from shinka.llm import AsyncLLMClient
from shinka.llm.providers import QueryResult
from shinka.llm.providers.errors import BilledCancellationError, LLMEndpointError


def test_meta_batch_keeps_billed_sibling_when_request_fails(monkeypatch):
    async def provider(msg, **kwargs):
        if "first" in msg:
            raise LLMEndpointError(401, "unauthorized")
        return QueryResult("summary", msg, "sys", [], "gpt-5-mini", {}, 1, 1, cost=0.8)

    monkeypatch.setattr(llm_module, "query_async", provider)
    sync = MetaSummarizer()
    for generation, label in enumerate(["first", "second"]):
        sync.add_evaluated_program(
            Program(
                id=label,
                code=label,
                generation=generation,
                metadata={"patch_name": label},
            )
        )
    summarizer = AsyncMetaSummarizer(
        sync, AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
    )

    assert asyncio.run(summarizer.update_meta_memory_async()) == (None, 0.8)


def test_final_meta_persists_billed_sibling_on_batch_cancellation(
    tmp_path, monkeypatch
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )

    async def run():
        started = asyncio.Event()

        async def provider(msg, system_msg, model_name, **kwargs):
            if "second" in msg:
                started.set()
                await asyncio.Event().wait()
            return QueryResult(
                "summary", msg, system_msg, [], model_name, {}, 1, 1, cost=0.8
            )

        monkeypatch.setattr(llm_module, "query_async", provider)
        sync = MetaSummarizer()
        best = Program(
            id="first", generation=1, code="first", metadata={"patch_name": "first"}
        )
        db.add(best)
        sync.add_evaluated_program(best)
        sync.add_evaluated_program(
            Program(
                id="second",
                generation=2,
                code="second",
                metadata={"patch_name": "second"},
            )
        )
        summarizer = AsyncMetaSummarizer(
            sync, AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
        )
        task = asyncio.create_task(
            summarizer.perform_final_summary_async(str(tmp_path), best, db.config)
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(BilledCancellationError) as cancelled:
            await task
        assert cancelled.value.cost == pytest.approx(0.8)
        assert db.get(best.id).metadata["meta_cost"] == pytest.approx(0.8)

    try:
        asyncio.run(run())
    finally:
        db.close()


@pytest.mark.parametrize(
    ("initial_spend", "model", "calls", "new_cost"),
    [
        (0.0, "gpt-5-mini", 1, 1.0),
        (1.0, "gpt-5-mini", 0, 0.0),
        (1.0, "local/test@http://127.0.0.1:1234/v1", 3, 0.0),
    ],
)
def test_final_meta_admits_each_paid_stage(
    tmp_path, monkeypatch, initial_spend, model, calls, new_cost
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    requests = []

    async def provider(msg, system_msg, model_name, **kwargs):
        requests.append(msg)
        cost = 0.0 if model_name.startswith("local/") else 1.0
        return QueryResult(
            "summary", msg, system_msg, [], model_name, {}, 1, 1, cost=cost
        )

    monkeypatch.setattr(llm_module, "query_async", provider)
    try:
        program = Program(
            id="best",
            code="pass",
            generation=1,
            metadata={"patch_name": "best", "api_costs": initial_spend},
        )
        db.add(program)
        sync = MetaSummarizer()
        sync.add_evaluated_program(program)
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.prompt_db = None
        runner.evo_config = EvolutionConfig(max_api_costs=1.0)
        runner.total_api_cost = initial_spend
        runner.results_dir = tmp_path
        runner.async_db = SimpleNamespace()
        runner.meta_summarizer = AsyncMetaSummarizer(
            sync, AsyncLLMClient(model_names=model, verbose=False)
        )

        assert asyncio.run(runner._perform_final_meta_summary(program)) == (
            calls == 3,
            new_cost,
        )
        assert len(requests) == calls
        assert runner.total_api_cost == pytest.approx(initial_spend + new_cost)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(
            initial_spend + new_cost
        )
    finally:
        db.close()


@pytest.mark.parametrize("commit_before_error", [False, True])
def test_meta_metadata_write_failure_keeps_bill_once(tmp_path, commit_before_error):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.prompt_db = None
        runner.total_api_cost = 0.0
        program = Program(id="best", code="pass", generation=1)
        db.add(program)

        async def update(program_id, metadata):
            if commit_before_error:
                db.conn.execute(
                    "UPDATE programs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), program_id),
                )
                db.conn.commit()
            raise OSError("metadata write failed")

        async def record(**kwargs):
            db.record_attempt_event(**kwargs)

        runner.async_db = SimpleNamespace(
            update_program_metadata_async=update, record_attempt_event_async=record
        )
        with pytest.raises(OSError, match="metadata write failed"):
            asyncio.run(runner._record_meta_cost(program, 0.80))

        assert runner.total_api_cost == pytest.approx(0.80)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(0.80)
        assert (program.metadata or {}).get("meta_cost", 0.0) == (
            0.80 if commit_before_error else 0.0
        )
    finally:
        db.close()


@pytest.mark.parametrize("recommendations", [None, "recommendation"])
def test_post_evaluation_meta_cost_survives_missing_output_or_write_error(
    tmp_path, recommendations
):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.total_api_cost = 0.0
    runner.verbose = False
    runner.llm_selection = None
    runner.results_dir = tmp_path
    runner.evo_config = EvolutionConfig()
    saved = []

    async def best():
        return None

    async def update(program_id, metadata):
        saved.append(metadata)

    async def meta(best_program):
        return recommendations, 3.3

    async def write(path):
        raise OSError("output unavailable")

    async def noop():
        pass

    runner.async_db = SimpleNamespace(
        get_best_program_async=best, update_program_metadata_async=update
    )
    runner.meta_summarizer = SimpleNamespace(
        add_evaluated_program=lambda program: None,
        should_update_meta=lambda interval: True,
        update_meta_memory_async=meta,
        write_meta_output_async=write,
    )
    runner._update_best_solution_async = noop
    runner._log_program_to_wandb = lambda program: None
    job = AsyncRunningJob("job", "unused", "unused", 0, 0, 0, 1)
    program = Program(
        id="program", code="pass", generation=1, metadata={"meta_cost": 0.2}
    )
    asyncio.run(
        runner._apply_persisted_program_side_effects(
            PersistedProgramEvent(job, program, 0, 0, 0)
        )
    )
    assert runner.total_api_cost == pytest.approx(3.3)
    assert saved[-1]["meta_cost"] == pytest.approx(3.5)


@pytest.mark.parametrize("recommendations", [None, "recommendation"])
def test_final_meta_cost_survives_missing_output_or_write_error(
    tmp_path, recommendations
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:
        program = Program(
            id="program", code="pass", generation=1, metadata={"meta_cost": 0.2}
        )
        db.add(program)
        sync = MetaSummarizer()
        sync.add_evaluated_program(program)
        summarizer = AsyncMetaSummarizer(sync, object())

        async def meta(best_program):
            return recommendations, 3.3

        async def write(path):
            raise OSError("output unavailable")

        summarizer.update_meta_memory_async = meta
        summarizer.write_meta_output_async = write
        result = asyncio.run(
            summarizer.perform_final_summary_async(str(tmp_path), program, db.config)
        )
        assert result == (False, 3.3)
        assert db.get("program").metadata["meta_cost"] == pytest.approx(3.5)
    finally:
        db.close()


def test_failed_prompt_evolution_is_durable(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.total_api_cost = 0.0
        runner.prompt_api_cost = 0.0
        runner.prompt_evolution_counter = 0
        runner.completed_generations = 1
        runner.evo_config = EvolutionConfig(
            evolve_prompts=True, prompt_evolution_interval=1
        )
        runner.meta_summarizer = None
        runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: object())
        runner.prompt_db = SimpleNamespace(
            last_generation=0, get_total_evolution_costs=lambda: 0.0
        )

        async def top(count):
            return []

        async def evolve(**kwargs):
            return None, "full", 3.3

        async def record(**kwargs):
            db.record_attempt_event(**kwargs)

        runner.async_db = SimpleNamespace(
            get_top_programs_async=top, record_attempt_event_async=record
        )
        runner.prompt_evolver = SimpleNamespace(evolve=evolve)
        asyncio.run(runner._maybe_evolve_prompt())
        assert runner.total_api_cost == pytest.approx(3.3)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
    finally:
        db.close()


@pytest.mark.parametrize("recommendations", [None, "recommendation"])
def test_initial_meta_cost_is_persisted_before_optional_output(
    tmp_path, recommendations
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.db_config = db.config
        runner.prompt_db = None
        runner.total_api_cost = 0.0
        runner.llm_selection = None
        runner.verbose = False
        runner.results_dir = tmp_path
        runner.lang_ext = "py"
        runner.max_proposal_jobs = runner.max_evaluation_jobs = (
            runner.max_db_workers
        ) = 1
        runner.evo_config = EvolutionConfig(meta_rec_interval=1)

        async def add(program, **kwargs):
            db.add(program)

        async def best():
            return db.get_best_program()

        async def persist(program_id, metadata):
            db.cursor.execute(
                "UPDATE programs SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), program_id),
            )
            db.conn.commit()

        async def embedding(*args):
            return None, 0.0

        async def meta(best_program):
            return recommendations, 3.3

        async def write(path):
            raise OSError("output unavailable")

        runner.async_db = SimpleNamespace(
            add_program_async=add,
            get_best_program_async=best,
            update_program_metadata_async=persist,
        )
        runner.scheduler = SimpleNamespace(
            run=lambda *args: (
                {"correct": {"correct": True}, "metrics": {"combined_score": 1.0}},
                0.0,
            )
        )
        runner._get_code_embedding_async = embedding
        runner._record_progress = lambda: None
        runner._log_program_to_wandb = lambda program: None
        evaluated = []
        runner.meta_summarizer = SimpleNamespace(
            add_evaluated_program=evaluated.append,
            evaluated_since_last_meta=evaluated,
            should_update_meta=lambda interval: True,
            update_meta_memory_async=meta,
            write_meta_output_async=write,
        )

        def run():
            asyncio.run(
                runner._setup_initial_program_with_metadata("pass", "seed", "seed", 0.0)
            )

        if recommendations:
            with pytest.raises(OSError):
                run()
        else:
            run()
        assert runner.total_api_cost == pytest.approx(3.3)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
    finally:
        db.close()


@pytest.mark.parametrize("persist_before_error", [False, True])
def test_prompt_insert_failure_charges_exactly_once(
    tmp_path, monkeypatch, persist_before_error
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    prompts = SystemPromptDatabase(
        SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.prompt_db = prompts
        runner.total_api_cost = runner.prompt_api_cost = 0.0
        runner.prompt_evolution_counter = 0
        runner.completed_generations = 1
        runner.verbose = False
        runner.evo_config = EvolutionConfig(
            evolve_prompts=True, prompt_evolution_interval=1
        )
        runner.meta_summarizer = None
        runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: object())
        prompt = create_system_prompt(
            prompt_text="task",
            generation=1,
            patch_type="full",
            metadata={"llm": {"cost": 3.3}},
        )

        async def top(count):
            return []

        async def evolve(**kwargs):
            return prompt, "full", 3.3

        async def record(**kwargs):
            db.record_attempt_event(**kwargs)

        add = prompts.add

        def fail(prompt, **kwargs):
            if persist_before_error:
                add(prompt, **kwargs)
            raise OSError("prompt insert failed")

        monkeypatch.setattr(prompts, "add", fail)
        runner.async_db = SimpleNamespace(
            get_top_programs_async=top, record_attempt_event_async=record
        )
        runner.prompt_evolver = SimpleNamespace(evolve=evolve)
        asyncio.run(runner._maybe_evolve_prompt())
        assert runner.total_api_cost == pytest.approx(3.3)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
    finally:
        prompts.close()
        db.close()
