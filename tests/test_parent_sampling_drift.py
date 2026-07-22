"""Regression tests for defensive weighted-parent deserialization."""

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.parents import WeightedSamplingStrategy


def test_weighted_sampling_tolerates_corrupt_json_cell(tmp_path):
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "corrupt.db"), num_islands=1)
    )
    db.add(
        Program(
            id="good",
            code="def f():\n    return 1\n",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
            embedding=[0.1, 0.2],
        )
    )
    db.cursor.execute(
        "UPDATE programs SET metadata = ? WHERE id = ?",
        ("{not valid json", "good"),
    )
    db.conn.commit()

    selector = WeightedSamplingStrategy(
        cursor=db.cursor,
        conn=db.conn,
        config=db.config,
        get_program_func=db.get,
        island_idx=0,
    )

    try:
        selected = selector.sample_parent()
        assert selected is None or selected.id == "good"
    finally:
        db.close()
