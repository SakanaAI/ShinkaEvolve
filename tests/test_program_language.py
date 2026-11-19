"""Program rows record the configured language, which selects complexity analysis."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eval_integrity_utils import (
    SEED_ID,
    StaticScheduler,
    children,
    good_results,
    make_job,
    make_runner,
    open_database,
)
from shinka.database import ProgramDatabase
from shinka.database.complexity import analyze_code_metrics

JULIA_CODE = "function f(x)\n    if x > 0\n        return x\n    end\n    return -x\nend\n"


def _assert_stored_as_julia(sync_db: ProgramDatabase, program_id: str) -> None:
    stored = sync_db.get(program_id)
    assert stored is not None
    assert stored.language == "julia"
    assert (
        stored.metadata["code_analysis_metrics"]
        == analyze_code_metrics(JULIA_CODE, "julia")
        != analyze_code_metrics(JULIA_CODE, "python")
    )


def test_child_row_records_the_configured_language(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(
                async_db, StaticScheduler(good_results()), language="julia"
            )
            runner._read_file_async = lambda path: asyncio.sleep(0, result=JULIA_CODE)

            assert await runner._process_single_job_safely(make_job())

            (child,) = children(sync_db)
            _assert_stored_as_julia(sync_db, child.id)

    asyncio.run(run())


def test_seed_row_records_the_configured_language(tmp_path: Path) -> None:
    class SeedScheduler:
        def run(self, exec_fname: str, results_dir: str) -> tuple[dict[str, Any], float]:
            return good_results(), 0.1

    async def run() -> None:
        async with open_database(tmp_path, with_seed=False) as (sync_db, async_db):
            runner = make_runner(async_db, SeedScheduler(), language="julia")
            runner.results_dir = str(tmp_path)
            runner.lang_ext = "jl"
            runner.embedding_client = None
            runner.stuck_detection_count = 0

            await runner._setup_initial_program(JULIA_CODE)

            (seed,) = sync_db.get_all_programs()
            assert seed.generation == 0 and seed.correct
            _assert_stored_as_julia(sync_db, seed.id)

    asyncio.run(run())


def test_failed_generation_row_records_the_configured_language(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, StaticScheduler(), language="julia")
            runner.results_dir = str(tmp_path)
            runner.lang_ext = "jl"
            runner._update_completed_generations = lambda: asyncio.sleep(0)
            runner._log_wandb_population_progress = lambda: None
            runner._record_progress = lambda: None
            runner.slot_available = asyncio.Event()
            exec_path = tmp_path / "gen_3" / "main.jl"
            exec_path.parent.mkdir()
            exec_path.write_text(JULIA_CODE)
            runner._read_file_async = lambda path: asyncio.sleep(0, result=JULIA_CODE)

            program = await runner._persist_failed_generation(
                generation=3,
                exec_fname=str(exec_path),
                proposal_started_at=time.time(),
                sampling_worker_id=1,
                active_proposals_at_start=1,
                parent_program=SimpleNamespace(id=SEED_ID),
                archive_programs=[],
                top_k_programs=[],
                code_diff=None,
                meta_patch_data={},
                code_embedding=None,
                embed_cost=0.0,
                novelty_cost=0.0,
                api_costs=0.0,
                failure_stage="proposal_failed",
                failure_reason="no valid proposal",
            )

            assert program is not None
            _assert_stored_as_julia(sync_db, program.id)

    asyncio.run(run())
