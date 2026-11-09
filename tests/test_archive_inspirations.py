from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.inspirations import ArchiveInspirationSelector

# (id, island, score): island 0 holds the parent, one elite and one more
# program, so the island-local steps run short and the global fallback runs.
PROGRAMS = [
    ("parent", 0, 0.5),
    ("elite0", 0, 0.9),
    ("other0", 0, 0.4),
    ("a1", 1, 0.8),
    ("b1", 1, 0.7),
    ("c1", 1, 0.6),
]


def _populated_db(db_path: str) -> ProgramDatabase:
    db = ProgramDatabase(
        config=DatabaseConfig(
            db_path=db_path,
            num_islands=2,
            archive_size=20,
            elite_selection_ratio=0.3,
            enforce_island_separation=False,
        )
    )
    for program_id, island, score in PROGRAMS:
        db.add(
            Program(
                id=program_id,
                code=f"x = '{program_id}'\n",
                correct=True,
                combined_score=score,
                generation=1,
                island_idx=island,
            )
        )
        db.cursor.execute(
            "UPDATE programs SET island_idx = ? WHERE id = ?", (island, program_id)
        )
        db.conn.commit()
    # Drop the automatic copy of the first program to the other island
    db.cursor.execute(
        "DELETE FROM archive WHERE program_id NOT IN (%s)"
        % ",".join("?" * len(PROGRAMS)),
        [program_id for program_id, _, _ in PROGRAMS],
    )
    db.cursor.execute(
        "DELETE FROM programs WHERE id NOT IN (%s)" % ",".join("?" * len(PROGRAMS)),
        [program_id for program_id, _, _ in PROGRAMS],
    )
    db.conn.commit()
    return db


def test_archive_inspirations_are_distinct_when_falling_back_to_global(tmp_path):
    # Repeated because the island-local and global picks are random.
    for trial in range(20):
        db = _populated_db(str(tmp_path / f"inspirations_{trial}.db"))
        try:
            selector = ArchiveInspirationSelector(
                cursor=db.cursor,
                conn=db.conn,
                config=db.config,
                get_program_func=db.get,
                best_program_id=None,
                get_island_idx_func=db.island_manager.get_island_idx,
                program_from_row_func=db._program_from_row,
            )
            inspirations = selector.sample_context(db.get("parent"), 4)
            ids = [program.id for program in inspirations]

            assert len(ids) == len(set(ids)) == 4, (trial, ids)
            assert "parent" not in ids
        finally:
            db.close()
