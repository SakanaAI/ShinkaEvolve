"""Ambiguous metadata writes remain recoverable after reconciliation read errors."""

import asyncio
import json
import sys
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from shinka.api_budget import APIBudgetExceeded, run_request
from shinka.core import EvolutionConfig
from shinka.core.async_runner import AsyncRunningJob, PersistedProgramEvent, ShinkaEvolveRunner
from shinka.core.billing import BillingPersistenceError
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)


@pytest.fixture
def runner(tmp_path):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    runner.prompt_db = SystemPromptDatabase(
        SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
    )
    runner.evo_config = EvolutionConfig(
        max_api_costs=None, evolve_prompts=True, prompt_evolution_interval=1
    )
    runner.total_api_cost = runner.prompt_api_cost = 0.0
    runner.should_stop = asyncio.Event()
    runner.finalization_complete = asyncio.Event()
    runner.verbose = False
    runner.meta_summarizer = None
    runner.prompt_evolution_counter = 0
    runner.completed_generations = 1
    runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: object())

    async def record(**kwargs):
        runner.db.record_attempt_event(**kwargs)

    runner.async_db = SimpleNamespace(
        record_attempt_event_async=record, get_top_programs_async=AsyncMock(return_value=[]),
        close_async=AsyncMock(),
    )
    runner.active_proposal_tasks = {}
    runner.scheduler = SimpleNamespace(shutdown=Mock())
    runner._finish_wandb_logging = AsyncMock()
    yield runner
    runner.prompt_db.close()
    runner.db.close()


async def fail_billed_write(runner, monkeypatch, stage, outcome, record_id):
    def unreadable(*args):
        raise OSError("reconciliation read unavailable")

    if stage == "meta":
        program = Program(id=record_id, code="pass", generation=1)
        runner.db.add(program)

        async def write_meta(program):
            if outcome == "committed":
                runner.db.conn.execute(
                    "UPDATE programs SET metadata = ? WHERE id = ?",
                    (json.dumps(program.metadata), program.id),
                )
                runner.db.conn.commit()
            raise OSError("write acknowledgement unavailable")

        monkeypatch.setattr(runner, "_persist_program_metadata_async", write_meta)
        monkeypatch.setattr(runner.db, "get", unreadable)

        async def summarize(best):
            charge = await run_request(AsyncMock(return_value=1.0), float, "gpt-5-mini")
            return None, charge

        runner.meta_summarizer = SimpleNamespace(update_meta_memory_async=summarize)
        with pytest.raises(BillingPersistenceError):
            await runner._update_meta_memory(program, program)
    else:
        prompt = create_system_prompt(
            prompt_text="task", generation=1, patch_type="full",
            metadata={"llm": {"cost": 1.0}},
        )
        prompt.id = record_id
        add = runner.prompt_db.add

        def write_prompt(prompt, **kwargs):
            if outcome == "committed":
                add(prompt, **kwargs)
            raise OSError("write acknowledgement unavailable")

        async def evolve(**kwargs):
            charge = await run_request(AsyncMock(return_value=1.0), float, "gpt-5-mini")
            return prompt, "full", charge

        runner.meta_summarizer = None
        runner.prompt_evolver = SimpleNamespace(evolve=evolve)
        monkeypatch.setattr(runner.prompt_db, "add", write_prompt)
        monkeypatch.setattr(runner.prompt_db, "get", unreadable)
        await runner._maybe_evolve_prompt()


@pytest.mark.parametrize("stage", ["meta", "prompt"])
@pytest.mark.parametrize("outcome", ["committed", "not_committed"])
def test_shutdown_reconciles_ambiguous_charge_once(runner, monkeypatch, stage, outcome):
    async def run():
        with monkeypatch.context() as outage:
            await fail_billed_write(runner, outage, stage, outcome, "target")
            with pytest.raises(BillingPersistenceError):
                await runner._flush_unwritten_charges()

        await runner._cleanup_async()
        await runner._flush_unwritten_charges()
        assert await runner._get_total_api_costs() == pytest.approx(1.0)
        assert runner.total_api_cost == pytest.approx(1.0)
        provider = AsyncMock()
        scope, token = runner._begin_api_cost_scope()
        try:
            with pytest.raises(APIBudgetExceeded):
                await run_request(provider, float, "gpt-5-mini")
        finally:
            await runner._finish_api_cost_scope(scope, token, stage, 1)
        provider.assert_not_awaited()
        assert runner.should_stop.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["meta", "prompt"])
def test_reconciliation_transfers_to_retryable_journal(runner, monkeypatch, stage):
    async def run():
        with monkeypatch.context() as outage:
            await fail_billed_write(runner, outage, stage, "not_committed", "target")
        record = runner.async_db.record_attempt_event_async

        async def ambiguous_write(**kwargs):
            await record(**kwargs)
            raise OSError("journal acknowledgement unavailable")

        with monkeypatch.context() as outage:
            outage.setattr(runner.async_db, "record_attempt_event_async", ambiguous_write)
            with pytest.raises(BillingPersistenceError):
                await runner._cleanup_async()
        await runner._cleanup_async()
        await runner._flush_unwritten_charges()
        assert await runner._get_total_api_costs() == pytest.approx(1.0)
        assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1

    asyncio.run(run())


def test_same_generation_reconciliations_keep_distinct_targets(runner, monkeypatch):
    async def run():
        # These represent already-started stages reporting bills during shutdown.
        for stage, record_id in [("meta", "a"), ("meta", "b"), ("prompt", "a")]:
            with monkeypatch.context() as outage:
                # Admission is already closed; inject the received bill directly.
                async def received(operation, get_cost, model):
                    return await operation()

                outage.setattr(sys.modules[__name__], "run_request", received)
                await fail_billed_write(runner, outage, stage, "not_committed", record_id)
        await runner._flush_unwritten_charges()
        assert await runner._get_total_api_costs() == pytest.approx(3.0)
        assert runner.total_api_cost == pytest.approx(3.0)

    asyncio.run(run())


def test_cancelled_metadata_resave_cannot_duplicate_reconciled_bill(runner, monkeypatch):
    async def run():
        program = Program(id="target", generation=1, code="pass")
        runner.db.add(program)
        started, release, committed = threading.Event(), threading.Event(), threading.Event()
        writes = 0

        async def update(program_id, metadata):
            nonlocal writes
            writes += 1
            if writes == 1:
                raise OSError("metadata write unavailable")

            def delayed_commit():
                started.set()
                assert release.wait(5)
                with sqlite3.connect(runner.db.config.db_path) as connection:
                    connection.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (json.dumps(metadata), program_id),
                    )
                committed.set()

            await asyncio.to_thread(delayed_commit)

        read = runner.db.get
        monkeypatch.setattr(runner.db, "get", Mock(side_effect=[
            OSError("reconciliation read unavailable"), read(program.id),
        ]))
        runner.async_db.update_program_metadata_async = update
        runner.async_db.get_best_program_async = AsyncMock(return_value=program)
        runner.meta_summarizer = SimpleNamespace(
            add_evaluated_program=lambda p: None,
            should_update_meta=lambda n: True,
            update_meta_memory_async=AsyncMock(return_value=(None, 1.0)),
        )
        runner.llm_selection = None
        runner._update_best_solution_async = AsyncMock()
        runner._log_program_to_wandb = Mock()
        job = AsyncRunningJob("job", "unused", "unused", 0, 0, 0, 1)
        task = asyncio.create_task(runner._apply_persisted_program_side_effects(
            PersistedProgramEvent(job, program, 0, 0, 0)
        ))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await runner._flush_unwritten_charges()
        finally:
            release.set()
        assert await asyncio.to_thread(committed.wait, 2)
        assert await runner._get_total_api_costs() == pytest.approx(1.0)
        assert runner.total_api_cost == pytest.approx(1.0)

    asyncio.run(run())
