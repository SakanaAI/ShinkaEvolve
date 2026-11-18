"""Shared scaffolding for the evaluation result integrity tests."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.core.runtime_slots import LogicalSlotPool
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase

SEED_ID = "seed"
SEED_SCORE = -5.0


def good_results(score: float = 2.5) -> dict[str, Any]:
    return {
        "correct": {"correct": True, "error": None},
        "metrics": {"combined_score": score, "public": {"acc": 1.0}, "private": {}},
        "stdout_log": "stdout",
        "stderr_log": "",
    }


class StaticScheduler:
    """Return (or raise) a fixed sequence of results, repeating the last one."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes) or [good_results()]
        self.calls = 0

    async def get_job_results_async(self, job_id: Any, results_dir: str) -> Any:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class RecordingSelection:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


@asynccontextmanager
async def open_database(
    tmp_path: Path, *, with_seed: bool = True
) -> AsyncIterator[tuple[ProgramDatabase, AsyncProgramDatabase]]:
    """Open a real database, by default holding a correct seed with a negative score."""
    sync_db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1),
        embedding_model="",
    )
    async_db = AsyncProgramDatabase(sync_db=sync_db)
    try:
        if with_seed and sync_db.get(SEED_ID) is None:
            sync_db.add(
                Program(
                    id=SEED_ID,
                    code="def f():\n    return 1\n",
                    correct=True,
                    combined_score=SEED_SCORE,
                    generation=0,
                )
            )
        yield sync_db, async_db
    finally:
        await async_db.close_async()
        sync_db.close()


def make_runner(
    async_db: Any,
    scheduler: Any,
    *,
    language: str = "python",
    max_retries: int = 3,
    llm_selection: Optional[Any] = None,
) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.scheduler = scheduler
    runner.async_db = async_db
    runner.evo_config = SimpleNamespace(
        evolve_prompts=False, meta_rec_interval=None, language=language
    )
    runner.meta_summarizer = None
    runner.llm_selection = llm_selection
    runner.MAX_DB_RETRY_ATTEMPTS = max_retries
    runner.failed_jobs_for_retry = {}
    runner.submitted_jobs = {}
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
    runner.db_config = SimpleNamespace(max_stdout_log_chars=None)
    runner._read_file_async = lambda path: asyncio.sleep(0, result="print('hi')\n")
    runner._update_best_solution_async = lambda: asyncio.sleep(0, result=None)

    async def persist_metadata(program: Program) -> None:
        await async_db.update_program_metadata_async(program.id, program.metadata)

    runner._persist_program_metadata_async = persist_metadata
    return runner


def make_job(
    job_id: Any = "job-1", generation: int = 1, **overrides: Any
) -> AsyncRunningJob:
    now = time.time()
    fields: dict[str, Any] = dict(
        job_id=job_id,
        exec_fname="program.py",
        results_dir="results",
        start_time=now - 6.0,
        proposal_started_at=now - 6.0,
        evaluation_submitted_at=now - 2.0,
        evaluation_started_at=now - 1.0,
        generation=generation,
        parent_id=SEED_ID,
        meta_patch_data={"patch_name": "patch", "model_name": "model-a"},
    )
    fields.update(overrides)
    return AsyncRunningJob(**fields)


def children(sync_db: ProgramDatabase) -> list[Program]:
    return [p for p in sync_db.get_all_programs() if p.generation > 0]
