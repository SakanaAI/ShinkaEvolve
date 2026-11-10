import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase

CRITERIA = {"combined_score": 1.0, "loc": -0.1}


def _program(program_id: str, score: float) -> Program:
    # Same code length for every program, so only combined_score differs.
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=score,
        generation=0,
        island_idx=0,
    )


def _archive_ids(db: ProgramDatabase) -> set[str]:
    db.cursor.execute("SELECT program_id FROM archive")
    return {row["program_id"] for row in db.cursor.fetchall()}


def test_ranked_archive_keeps_better_program_over_newer_worse_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgramDatabase(
            config=DatabaseConfig(
                db_path=str(Path(tmpdir) / "archive.db"),
                num_islands=1,
                archive_size=3,
                archive_criteria=CRITERIA,
            )
        )
        try:
            programs = [
                _program(program_id, score)
                for program_id, score in [
                    ("a", 0.9),
                    ("b", 0.8),
                    ("worst", 0.5),
                    ("newer_but_worse", 0.3),
                ]
            ]
            for offset, program in enumerate(programs):
                program.timestamp = 1000.0 + offset  # added in this order
            for program in programs[:3]:
                db.add(program)
            assert _archive_ids(db) == {"a", "b", "worst"}

            db.add(programs[3])

            assert _archive_ids(db) == {"a", "b", "worst"}
        finally:
            db.close()


def test_ranked_is_better_compares_the_two_programs_directly():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgramDatabase(
            config=DatabaseConfig(
                db_path=str(Path(tmpdir) / "is_better.db"),
                num_islands=1,
                archive_criteria=CRITERIA,
            )
        )
        try:
            context = [_program("a", 0.9), _program("b", 0.8)]
            better, worse = _program("better", 0.5), _program("worse", 0.3)
            worse.timestamp = better.timestamp + 1.0  # newer, would win a tie

            assert db._is_better(better, worse, context)
            assert not db._is_better(worse, better, context)
        finally:
            db.close()
