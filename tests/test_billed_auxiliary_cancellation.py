"""Prompt evolution persists received client bills when cancellation propagates."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.prompt_evolver import AsyncSystemPromptEvolver
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.prompt_dbase import create_system_prompt
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import BilledCancellationError, EmptyResponseError


@pytest.mark.parametrize("patch_type", ["diff", "full"])
def test_prompt_cancellation_books_carried_client_bill(
    tmp_path, monkeypatch, patch_type
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:

        async def run():
            runner = object.__new__(ShinkaEvolveRunner)
            runner.db = db
            runner.total_api_cost = runner.prompt_api_cost = 0.0
            runner.prompt_evolution_counter = 0
            runner.completed_generations = 1
            runner.evo_config = EvolutionConfig(
                evolve_prompts=True, prompt_evolution_interval=1
            )
            runner.meta_summarizer = None
            parent = create_system_prompt(
                prompt_text="task", generation=0, patch_type="init"
            )
            runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: parent)
            runner.prompt_db = SimpleNamespace(
                last_generation=0, get_total_evolution_costs=lambda: 0.0
            )

            async def top(count):
                return []

            async def record(**kwargs):
                db.record_attempt_event(**kwargs)

            billed = asyncio.Event()

            async def provider(**kwargs):
                billed.set()
                raise EmptyResponseError("no text", None, {"cost": 3.3})

            runner.async_db = SimpleNamespace(
                get_top_programs_async=top, record_attempt_event_async=record
            )
            runner.prompt_evolver = AsyncSystemPromptEvolver(
                AsyncLLMClient(model_names="gpt-5-mini", verbose=False),
                patch_types=[patch_type],
                patch_type_probs=[1.0],
            )
            monkeypatch.setattr(llm_module, "query_async", provider)
            task = asyncio.create_task(runner._maybe_evolve_prompt())
            await asyncio.wait_for(billed.wait(), 2)
            task.cancel()
            with pytest.raises(BilledCancellationError):
                await task
            assert task.cancelled()
            assert runner.prompt_api_cost == pytest.approx(3.3)
            assert runner.total_api_cost == pytest.approx(3.3)
            assert await runner._get_total_api_costs() == pytest.approx(3.3)
            assert (
                db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
            )

        asyncio.run(run())
    finally:
        db.close()
