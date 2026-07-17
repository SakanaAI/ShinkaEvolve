"""Regression tests for the performance indexes on the programs table."""

import sqlite3
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _program(pid: str, score: float, correct: bool = True) -> Program:
    return Program(
        id=pid,
        code="def f():\n    return 1\n",
        correct=correct,
        combined_score=score,
        generation=0,
        island_idx=0,
    )


def test_score_indexes_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "idx.db"
        ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))

        conn = sqlite3.connect(str(db_path))
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_programs_correct_score" in names
        assert "idx_programs_island_correct_score" in names


def test_top_programs_query_uses_score_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "plan.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        for i in range(5):
            db.add(_program(f"p{i}", float(i)))

        conn = sqlite3.connect(str(db_path))
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM programs "
            "WHERE combined_score IS NOT NULL AND correct = 1 "
            "ORDER BY combined_score DESC LIMIT 3"
        ).fetchall()
        conn.close()
        plan_text = " ".join(str(row) for row in plan)
        # The planner should use an index rather than scanning + sorting.
        assert "idx_programs_correct_score" in plan_text
        assert "SCAN programs" not in plan_text
