"""Opening databases written by older schema versions."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase

# The `programs` table as created by the initial release, before the
# `system_prompt_id` column existed.
_INITIAL_SCHEMA = """
    CREATE TABLE programs (
        id TEXT PRIMARY KEY,
        code TEXT NOT NULL,
        language TEXT NOT NULL,
        parent_id TEXT,
        archive_inspiration_ids TEXT,
        top_k_inspiration_ids TEXT,
        generation INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        code_diff TEXT,
        combined_score REAL,
        public_metrics TEXT,
        private_metrics TEXT,
        text_feedback TEXT,
        complexity REAL,
        embedding TEXT,
        embedding_pca_2d TEXT,
        embedding_pca_3d TEXT,
        embedding_cluster_id INTEGER,
        correct BOOLEAN DEFAULT 0,
        children_count INTEGER NOT NULL DEFAULT 0,
        metadata TEXT,
        migration_history TEXT,
        island_idx INTEGER
    )
"""


def _config(tmp_path: Path) -> DatabaseConfig:
    return DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)


def _write_initial_schema_db(db_path: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(_INITIAL_SCHEMA)
        conn.execute(
            "CREATE TABLE archive (program_id TEXT PRIMARY KEY, "
            "FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE)"
        )
        conn.execute("CREATE TABLE metadata_store (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO programs (id, code, language, generation, timestamp, "
            "combined_score, correct, metadata, island_idx) "
            "VALUES ('seed', 'pass', 'python', 0, 1.0, 2.5, 1, '{}', 0)"
        )
        conn.execute("INSERT INTO archive (program_id) VALUES ('seed')")
        conn.commit()


def _columns_and_indexes(db_path: str) -> tuple[set[str], set[str]]:
    with closing(sqlite3.connect(db_path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(programs)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(programs)")}
    return columns, indexes


def test_database_without_system_prompt_id_column_opens(tmp_path):
    config = _config(tmp_path)
    assert config.db_path is not None
    _write_initial_schema_db(config.db_path)

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        seed = db.get("seed")
        assert seed is not None
        assert (seed.correct, seed.combined_score) == (True, 2.5)
        assert seed.system_prompt_id is None
        db.add(
            Program(
                id="child",
                code="pass",
                parent_id="seed",
                generation=1,
                correct=True,
                combined_score=3.0,
                system_prompt_id="prompt-1",
            )
        )
        child = db.get("child")
        assert child is not None
        assert child.system_prompt_id == "prompt-1"

    columns, indexes = _columns_and_indexes(config.db_path)
    assert "system_prompt_id" in columns
    assert "idx_programs_system_prompt_id" in indexes


def test_migrated_database_reopens(tmp_path):
    config = _config(tmp_path)
    assert config.db_path is not None
    _write_initial_schema_db(config.db_path)

    with closing(ProgramDatabase(config, embedding_model="")):
        pass
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        assert db.get("seed") is not None


def _compute_time(db_path: str, program_id: str) -> float:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT metadata FROM programs WHERE id = ?", (program_id,)
        ).fetchone()
    return json.loads(row[0])["compute_time"]


def _set_metadata(db_path: str, program_id: str, metadata: dict) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE programs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), program_id),
        )
        conn.commit()


def test_compute_time_migration_runs_once_per_database(tmp_path):
    config = _config(tmp_path)
    assert config.db_path is not None
    _write_initial_schema_db(config.db_path)
    stale = {"compute_time": 9.0, "evaluation_seconds": 4.0}
    _set_metadata(config.db_path, "seed", stale)

    with closing(ProgramDatabase(config, embedding_model="")):
        pass
    assert _compute_time(config.db_path, "seed") == 4.0

    # The rewrite is a one-off repair of old rows, not part of every open.
    _set_metadata(config.db_path, "seed", stale)
    with closing(ProgramDatabase(config, embedding_model="")):
        pass
    assert _compute_time(config.db_path, "seed") == 9.0
