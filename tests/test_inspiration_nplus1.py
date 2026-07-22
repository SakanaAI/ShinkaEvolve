"""Functional test for the single-query archive-inspiration sampling.

The archive/random inspiration loops now fetch full rows (SELECT p.*) and build
Program objects directly instead of issuing a per-id follow-up query. This
verifies the sampled inspirations still come back as fully-populated programs.
"""

import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _program(pid: str, score: float, island: int = 0) -> Program:
    return Program(
        id=pid,
        code=f"# code for {pid}\ndef f():\n    return {score}\n",
        correct=True,
        combined_score=score,
        generation=int(score),
        island_idx=island,
        embedding=[score, score + 1.0],
    )


def test_sample_returns_fully_populated_inspirations():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "insp.db"
        config = DatabaseConfig(
            db_path=str(db_path),
            num_islands=1,
            num_archive_inspirations=2,
            num_top_k_inspirations=1,
            elite_selection_ratio=0.5,
            enforce_island_separation=False,
        )
        db = ProgramDatabase(config=config)
        for i in range(6):
            db.add(_program(f"p{i}", float(i)))

        parent, archive_insp, topk_insp = db.sample()

        # Inspirations must be real, fully-hydrated Program objects (code + id),
        # proving the SELECT p.* + program_from_row path works end to end.
        for prog in archive_insp + topk_insp:
            assert isinstance(prog, Program)
            assert prog.id
            assert prog.code and prog.code.startswith("# code for")
