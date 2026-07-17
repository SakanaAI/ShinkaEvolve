"""The single-pass similarity scan must match the old two-method behavior."""

import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _program(pid: str, emb, island: int = 0) -> Program:
    return Program(
        id=pid,
        code=f"# {pid}\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=island,
        embedding=emb,
    )


def test_details_match_legacy_methods():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "sim.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        db.add(_program("a", [1.0, 0.0, 0.0]))
        db.add(_program("b", [0.9, 0.1, 0.0]))
        db.add(_program("c", [0.0, 1.0, 0.0]))

        query = [1.0, 0.05, 0.0]

        legacy_scores = db.compute_similarity(query, 0)
        legacy_best = db.get_most_similar_program(query, 0)

        scores, best = db.compute_similarity_details(query, 0)

        assert scores == legacy_scores
        assert best is not None and legacy_best is not None
        assert best.id == legacy_best.id
        # "a" (unit x-axis) is closest to the query.
        assert best.id == "a"


def test_details_thread_safe_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "sim_ts.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        db.add(_program("a", [1.0, 0.0]))
        db.add(_program("b", [0.0, 1.0]))

        query = [0.1, 1.0]
        scores, best = db.compute_similarity_details_thread_safe(query, 0)
        assert len(scores) == 2
        assert best is not None and best.id == "b"


def test_details_empty_embedding_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "sim_empty.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        db.add(_program("a", [1.0, 0.0]))
        scores, best = db.compute_similarity_details([], 0)
        assert scores == []
        assert best is None
