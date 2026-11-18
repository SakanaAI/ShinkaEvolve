"""Generated seed bills survive evaluation handoff and settle exactly once."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import shinka.llm.llm as llm_module
from shinka.api_budget import APIBudgetExceeded, run_request
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.billing import BillingPersistenceError
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.llm import AsyncLLMClient
from shinka.llm.providers import QueryResult


@pytest.fixture
def runner(tmp_path, monkeypatch):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db_config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=3)
    runner.db = ProgramDatabase(runner.db_config, embedding_model="")
    runner.prompt_db = None
    runner.total_api_cost = 0.0
    runner.evo_config = EvolutionConfig(max_api_costs=1.0, max_patch_attempts=1)
    runner.should_stop = asyncio.Event()
    runner.finalization_complete = asyncio.Event()
    runner.verbose = False
    runner.results_dir = tmp_path
    runner.lang_ext = "py"
    runner.max_proposal_jobs = runner.max_evaluation_jobs = runner.max_db_workers = 1
    runner.llm_selection = runner.meta_summarizer = None
    runner.llm = AsyncLLMClient("gpt-5-mini", verbose=False)
    runner.prompt_sampler = SimpleNamespace(initial_program_prompt=lambda: ("sys", "msg"))
    runner.scheduler = SimpleNamespace(run=Mock(return_value=(
        {"correct": {"correct": True}, "metrics": {"combined_score": 1.0}}, 0.0,
    )))
    runner._get_code_embedding_async = AsyncMock(return_value=(None, 0.0))
    runner._record_progress = Mock()
    runner._log_program_to_wandb = Mock()
    monkeypatch.setattr(llm_module, "query_async", AsyncMock(return_value=QueryResult(
        "```python\npass\n```", "msg", "sys", [], "gpt-5-mini", {}, 1, 1, cost=1.0,
    )))
    yield runner
    runner.db.close()


@pytest.mark.parametrize("outcome", ["complete", "cancel", "evaluation_error"])
def test_seed_bill_is_recoverable_while_evaluation_is_outstanding(runner, outcome):
    async def run():
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        started, release = threading.Event(), threading.Event()
        evaluate = runner.scheduler.run

        def blocked_evaluation(*args):
            started.set()
            assert release.wait(5)
            if outcome == "evaluation_error":
                raise OSError("evaluation failed")
            return evaluate(*args)

        runner.scheduler.run = blocked_evaluation
        task = asyncio.create_task(runner._generate_initial_program())
        try:
            assert await asyncio.to_thread(started.wait, 2)
            restored = await runner._get_total_api_costs()
            assert restored == pytest.approx(1.0)
            resumed = object.__new__(ShinkaEvolveRunner)
            resumed.evo_config = runner.evo_config
            resumed.total_api_cost = restored
            with pytest.raises(APIBudgetExceeded):
                await resumed._generate_initial_program()
        finally:
            if outcome == "cancel":
                task.cancel()
            release.set()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            await runner.async_db.close_async()

        assert runner.total_api_cost == pytest.approx(1.0)
        assert await runner._get_total_api_costs() == pytest.approx(1.0)
        assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        programs = runner.db.get_programs_by_generation(0)
        if outcome == "cancel":
            assert programs == []
        else:
            assert len(programs) == 3
            assert len({p.metadata["api_charge_id"] for p in programs}) == 1
            copies = [p for p in programs if p.metadata.get("_is_island_copy")]
            assert len(copies) == 2
            assert all(p.metadata["api_costs"] == 0.0 for p in copies)

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["committed", "not_committed"])
@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_seed_insert_interruption_settles_embeddings_once(runner, monkeypatch, outcome, error_type):
    async def run():
        runner.evo_config.max_api_costs = 5.0
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        insert = runner.async_db.add_program_async

        async def interrupted_insert(program, **kwargs):
            if outcome == "committed":
                await insert(program, **kwargs)
            raise error_type("insert interrupted")

        async def embed(path):
            return await run_request(
                AsyncMock(return_value=([0.1], 0.25)), lambda result: result[1],
                "text-embedding-3-small", "embedding",
            )

        monkeypatch.setattr(runner.async_db, "add_program_async", interrupted_insert)
        runner._get_code_embedding_async = embed
        try:
            with pytest.raises(error_type):
                await runner._generate_initial_program()
            await runner._flush_unwritten_charges()
            assert runner.total_api_cost == pytest.approx(1.25)
            assert await runner._get_total_api_costs() == pytest.approx(1.25)
            assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["committed", "not_committed"])
def test_failed_seed_receipt_blocks_evaluation_and_retries_same_bill(runner, monkeypatch, outcome):
    async def run():
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        record = runner.async_db.record_attempt_event_async

        async def unavailable(**kwargs):
            if outcome == "committed":
                await record(**kwargs)
            raise OSError("receipt acknowledgement unavailable")

        try:
            with monkeypatch.context() as outage:
                outage.setattr(runner.async_db, "record_attempt_event_async", unavailable)
                with pytest.raises(BillingPersistenceError):
                    await runner._generate_initial_program()
            runner.scheduler.run.assert_not_called()
            assert runner._paid_admission_closed()
            await runner._flush_unwritten_charges()
            await runner._flush_unwritten_charges()
            assert runner.total_api_cost == pytest.approx(1.0)
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())


def test_cancellation_during_seed_receipt_finishes_one_charge(runner, monkeypatch):
    async def run():
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        record = runner.async_db.record_attempt_event_async
        committed, release = asyncio.Event(), asyncio.Event()

        async def delayed_acknowledgement(**kwargs):
            await record(**kwargs)
            committed.set()
            await release.wait()

        monkeypatch.setattr(runner.async_db, "record_attempt_event_async", delayed_acknowledgement)
        task = asyncio.create_task(runner._generate_initial_program())
        try:
            await asyncio.wait_for(committed.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            runner.scheduler.run.assert_not_called()
            assert runner.total_api_cost == pytest.approx(1.0)
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        finally:
            release.set()
            await runner.async_db.close_async()

    asyncio.run(run())


def test_seed_receipt_includes_previous_billed_attempts(runner, monkeypatch):
    async def run():
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        runner.evo_config.max_patch_attempts = 2
        provider = AsyncMock(side_effect=[
            QueryResult(None, "msg", "sys", [], "gpt-5-mini", {}, 1, 1, cost=0.4),
            QueryResult("```python\npass\n```", "msg", "sys", [], "gpt-5-mini", {}, 1, 1, cost=0.6),
        ])
        monkeypatch.setattr(llm_module, "query_async", provider)
        try:
            await runner._generate_initial_program()
            assert provider.await_count == 2
            assert runner.total_api_cost == pytest.approx(1.0)
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())
