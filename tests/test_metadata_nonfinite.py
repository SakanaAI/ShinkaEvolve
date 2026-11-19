"""Non-finite metadata values must not hide a row from SQLite's JSON lookups."""

import asyncio
import math
from pathlib import Path

from eval_integrity_utils import SEED_ID, open_database
from shinka.database import Program, ProgramDatabase


def _program(program_id: str) -> Program:
    return Program(
        id=program_id,
        code="x = 1\n",
        generation=1,
        parent_id=SEED_ID,
        metadata={
            "source_job_id": "job-9",
            "api_costs": math.nan,
            "timing": {"samples": [1.0, math.inf, -math.inf]},
        },
    )


def _metadata_is_valid_json(sync_db: ProgramDatabase, program_id: str) -> bool:
    row = sync_db.cursor.execute(
        "SELECT json_valid(metadata) AS valid FROM programs WHERE id = ?", (program_id,)
    ).fetchone()
    return bool(row["valid"])


def test_row_with_non_finite_metadata_is_found_by_the_duplicate_lookup(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            first = _program("first")

            assert await async_db.add_program_async(first)
            assert not await async_db.add_program_async(_program("retry"))

            assert _metadata_is_valid_json(sync_db, "first")
            assert sync_db.has_program_with_source_job_id("job-9")
            found = await async_db.get_program_by_source_job_id_async("job-9")
            assert found is not None and found.id == "first"
            assert found.metadata["api_costs"] is None
            assert found.metadata["timing"] == {"samples": [1.0, None, None]}
            assert sync_db.get("retry") is None

    asyncio.run(run())


def test_metadata_stays_valid_after_deferred_maintenance(tmp_path: Path) -> None:
    """Maintenance writes the program's in-memory metadata back to the row."""

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            program = _program("deferred")

            await async_db.add_program_async(program, defer_maintenance=True)
            await async_db.run_program_maintenance_async(program)

            stored = sync_db.get("deferred")
            assert stored.metadata["postprocess_db_maintenance_applied"] is True
            assert _metadata_is_valid_json(sync_db, "deferred")
            assert sync_db.has_program_with_source_job_id("job-9")

    asyncio.run(run())


def test_metadata_update_cleans_non_finite_values(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            await async_db.add_program_async(_program("updated"))

            await async_db.update_program_metadata_async(
                "updated", {"source_job_id": "job-9", "meta_cost": math.nan}
            )

            assert _metadata_is_valid_json(sync_db, "updated")
            assert sync_db.get("updated").metadata["meta_cost"] is None

    asyncio.run(run())
