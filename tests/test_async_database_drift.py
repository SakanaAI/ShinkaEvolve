"""Regression tests for async database serialization parity."""

import asyncio
import json
import sqlite3

import pytest

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


@pytest.mark.parametrize(
    ("record_event", "table"),
    [
        ("record_attempt_event_async", "attempt_log"),
        ("record_generation_event_async", "generation_event_log"),
    ],
)
def test_async_event_details_are_strict_json(tmp_path, record_event, table):
    db_path = tmp_path / "events.db"
    sync_db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(db_path), num_islands=1)
    )
    async_db = AsyncProgramDatabase(sync_db, max_workers=1)

    async def record() -> None:
        method = getattr(async_db, record_event)
        kwargs = {"generation": 1, "status": "failed"}
        if record_event == "record_attempt_event_async":
            kwargs["stage"] = "proposal"
        await method(details={"nan": float("nan"), "inf": float("inf"), "ok": 1.0}, **kwargs)

    try:
        asyncio.run(record())
        with sqlite3.connect(db_path) as connection:
            payload = connection.execute(f"SELECT details FROM {table}").fetchone()[0]
        parsed = json.loads(payload, parse_constant=_reject_nonstandard_constant)
        assert parsed == {"inf": None, "nan": None, "ok": 1.0}
    finally:
        asyncio.run(async_db.close_async())
        sync_db.close()
