"""Regression tests for drift bugs fixed on the dedup branch (local group).

- #13: embedding_overrides are applied at runtime, not only at CSV-build time.
- #1:  the async event log strips NaN/Inf (valid JSON), like the sync log.
- #2:  weighted parent sampling tolerates a corrupt JSON cell.
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.pricing.catalog import ModelPrice
from shinka.pricing.normalization import MILLION, _apply_embedding_overrides


def _embedding_price(input_price: float) -> ModelPrice:
    return ModelPrice(
        model_name="gemini-embedding-exp-03-07",
        api_model_name="gemini-embedding-exp-03-07",
        provider="google",
        kind="embedding",
        input_price=input_price,
        output_price=0.0,
    )


def test_embedding_overrides_applied_at_runtime():
    key = ("embedding", "google", "gemini-embedding-exp-03-07")
    entries = {key: _embedding_price(input_price=1.5 / MILLION)}

    _apply_embedding_overrides(
        entries,
        [{"provider": "google", "model_name": "gemini-embedding-exp-03-07",
          "input_price": 0.0}],
    )

    # The pinned 0.0 overrides the upstream price (was drifting on live refresh).
    assert entries[key].input_price == 0.0


def test_embedding_overrides_ignores_unknown_and_malformed():
    key = ("embedding", "google", "gemini-embedding-exp-03-07")
    entries = {key: _embedding_price(input_price=2.0 / MILLION)}
    _apply_embedding_overrides(entries, "not-a-list")  # tolerated
    _apply_embedding_overrides(entries, [{"provider": "google"}])  # no model_name
    _apply_embedding_overrides(
        entries, [{"provider": "x", "model_name": "y", "input_price": 0.0}]
    )  # unknown key
    assert entries[key].input_price == 2.0 / MILLION


def _program(pid: str) -> Program:
    return Program(
        id=pid,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=0,
        embedding=[0.1, 0.2],
    )


def test_async_event_log_strips_nan():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "events.db"
        sync_db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1)
        )
        sync_db.add(_program("p0"))
        adb = AsyncProgramDatabase(sync_db, max_workers=1)

        async def _run():
            await adb.record_attempt_event_async(
                generation=1,
                stage="proposal",
                status="failed",
                details={"score": float("nan"), "ok": 1.0},
            )

        asyncio.run(_run())

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT details FROM attempt_log").fetchall()
        conn.close()
        assert rows, "no attempt event recorded"
        # Strict JSON parse must succeed (NaN would be an invalid token).
        for row in rows:
            if row["details"]:
                parsed = json.loads(row["details"])  # raises on 'NaN'
                assert parsed.get("score") is None
                assert parsed.get("ok") == 1.0


def test_parent_sampling_tolerates_corrupt_json_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "corrupt.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1)
        )
        db.add(_program("good"))  # a correct program is auto-added to the archive
        # Corrupt a JSON column that weighted sampling parses.
        db.cursor.execute(
            "UPDATE programs SET metadata = ? WHERE id = ?",
            ("{not valid json", "good"),
        )
        db.conn.commit()

        # Should not raise even though a cell is corrupt.
        from shinka.database.parents import WeightedSamplingStrategy

        selector = WeightedSamplingStrategy(
            cursor=db.cursor,
            conn=db.conn,
            config=db.config,
            get_program_func=db.get,
            island_idx=0,
        )
        # The internal safe-parse must swallow the bad cell; the call returns a
        # program or None but never raises JSONDecodeError.
        result = selector.sample_parent()
        assert result is None or hasattr(result, "id")
