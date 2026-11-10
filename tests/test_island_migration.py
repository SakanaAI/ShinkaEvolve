from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _populated_db(db_path: str) -> ProgramDatabase:
    db = ProgramDatabase(
        config=DatabaseConfig(
            db_path=db_path, num_islands=2, migration_rate=0.5, archive_size=0
        )
    )
    for island in range(2):
        for rank, score in enumerate([0.9, 0.5, 0.4, 0.3]):
            program_id = f"i{island}_{rank}"
            db.add(
                Program(
                    id=program_id,
                    code=f"x = {island}{rank}\n",
                    correct=True,
                    combined_score=score,
                    generation=1,
                    island_idx=island,
                )
            )
            db.cursor.execute(
                "UPDATE programs SET island_idx = ? WHERE id = ?",
                (island, program_id),
            )
            db.conn.commit()
    # Drop the automatic copy of the first program, so each island holds only
    # the four programs above
    db.cursor.execute("DELETE FROM programs WHERE id NOT LIKE 'i%'")
    db.conn.commit()
    return db


def test_migration_counts_come_from_pre_migration_populations(tmp_path):
    # Each island holds 1 elite + 3 programs, so with rate 0.5 it sends exactly 2
    # of its own programs and keeps its elite, regardless of what arrives from
    # the other island. Repeated because migrants are drawn at random.
    for trial in range(20):
        db = _populated_db(str(tmp_path / f"migration_{trial}.db"))
        try:
            assert db.island_manager.perform_migration(current_generation=10)
            db.cursor.execute("SELECT id, island_idx FROM programs")
            islands = {row["id"]: row["island_idx"] for row in db.cursor.fetchall()}
            for island in range(2):
                own = [pid for pid in islands if pid.startswith(f"i{island}_")]
                moved = [pid for pid in own if islands[pid] != island]
                assert len(moved) == 2, (trial, island, moved)
                assert islands[f"i{island}_0"] == island
        finally:
            db.close()
