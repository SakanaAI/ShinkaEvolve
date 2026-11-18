from contextlib import closing
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase


@pytest.fixture
def config(tmp_path: Path) -> DatabaseConfig:
    return DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=1,
        enable_dynamic_islands=True,
        stagnation_threshold=5,
    )


def _program(
    generation: int, score: float | None, metadata: dict | None = None
) -> Program:
    return Program(
        id=f"g{generation}",
        code="pass",
        generation=generation,
        timestamp=float(generation),
        combined_score=score,
        correct=True,
        island_idx=0,
        metadata=metadata or {},
    )


def _write_old_database(
    config: DatabaseConfig,
    preview: str | None,
    seed_score: float,
    *,
    reset: int = 0,
    pending_score: float = 20.0,
    tied_generation: int | None = None,
) -> None:
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.add(_program(0, seed_score))
        if reset:
            assert db.check_and_spawn_island_if_stagnant(reset)
        db.add(
            _program(5, seed_score - 1, {"postprocess_db_maintenance_applied": False}),
            defer_maintenance=True,
        )
        if tied_generation is not None:
            db.add(
                _program(
                    tied_generation,
                    pending_score,
                    {"postprocess_db_maintenance_applied": False},
                ),
                defer_maintenance=True,
            )
        db.add(
            _program(
                9,
                pending_score,
                {
                    "safe_processing": True,
                    "source_job_id": "job-9",
                    "source_generation": 9,
                    "pipeline_accounted_seconds": 0.0,
                },
            ),
            defer_maintenance=True,
        )
        with db.conn:
            # A previously pruned row retains its completed-maintenance evidence.
            db.conn.execute("DELETE FROM archive WHERE program_id = 'g0'")
            db.conn.execute(
                "DELETE FROM metadata_store WHERE key LIKE 'run_state_reconciled_%'"
            )
            if preview:
                db.conn.executemany(
                    "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
                    [
                        (f"run_state_reconciled_{preview}", "1"),
                        ("best_program_id", "g9"),
                        (
                            "best_score_generation",
                            str(max(tied_generation or 9, reset)),
                        ),
                        ("best_score_ever", str(pending_score)),
                    ],
                )


@pytest.mark.parametrize("preview", [None, "v1", "v2"])
@pytest.mark.parametrize("seed_score", [-3.0, 0.0])
def test_deferred_upgrade_preserves_spawn_during_ordered_replay(
    config, preview, seed_score
):
    _write_old_database(config, preview, seed_score)

    for _ in range(2):
        with closing(ProgramDatabase(config, embedding_model="")) as db:
            assert db.last_iteration == 9
            assert db.best_program_id == "g0"
            assert (db.best_score_ever, db.best_score_generation) == (seed_score, 0)
            assert db.island_manager.get_island_populations() == {0: 3}
            assert db.get("g5").metadata["postprocess_db_maintenance_applied"] is False
            assert "postprocess_db_maintenance_applied" not in db.get("g9").metadata

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.run_post_add_maintenance(db.get("g5"))
        assert db.island_manager.get_island_populations() == {0: 3, 1: 1}
        assert db.best_score_generation == 5
        db.run_post_add_maintenance(db.get("g9"))
        assert db.best_program_id == "g9"
        assert (db.best_score_ever, db.best_score_generation) == (20.0, 9)

    with closing(ProgramDatabase(config, read_only=True)) as db:
        assert db.island_manager.get_island_populations() == {0: 3, 1: 1}
        assert db.best_program_id == "g9"
        assert (db.best_score_ever, db.best_score_generation) == (20.0, 9)


@pytest.mark.parametrize(("reset", "pending_score"), [(12, 20.0), (9, 20.0), (9, 0.0)])
def test_preview_upgrade_repairs_pending_id_and_preserves_later_reset(
    config, reset, pending_score
):
    _write_old_database(config, "v1", 0.0, reset=reset, pending_score=pending_score)

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        assert db.best_program_id == "g0"
        assert (db.best_score_ever, db.best_score_generation) == (0.0, reset)
        assert db.island_manager.get_island_populations() == {0: 3, 1: 1}


def test_preview_upgrade_uses_pending_tie_at_watermark(config):
    _write_old_database(config, "v1", 10.0, tied_generation=7)

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        assert db.best_program_id == "g0"
        assert (db.best_score_ever, db.best_score_generation) == (10.0, 0)
        db.run_post_add_maintenance(db.get("g5"))
        assert db.island_manager.get_island_populations() == {0: 4, 1: 1}


def test_preview_upgrade_keeps_watermark_supported_by_markerless_incumbent(config):
    config.archive_size = 0
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        seed = _program(0, 10.0)
        seed.public_metrics = {"quality": 1.0}
        db.add(seed)
        assert db.check_and_spawn_island_if_stagnant(9)
        db.add(
            _program(9, 10.0, {"postprocess_db_maintenance_applied": False}),
            defer_maintenance=True,
        )
        with db.conn:
            db.conn.execute("UPDATE programs SET metadata = '{}' WHERE id = 'g0'")
            db.conn.execute(
                "DELETE FROM metadata_store WHERE key LIKE 'run_state_reconciled_%'"
            )
            db.conn.execute(
                "INSERT INTO metadata_store (key, value) "
                "VALUES ('run_state_reconciled_v1', '1')"
            )

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        assert db.best_program_id == "g0"
        assert (db.best_score_ever, db.best_score_generation) == (10.0, 9)


@pytest.mark.parametrize(
    (
        "seed_score",
        "weak_score",
        "pending_score",
        "preview_score",
        "preview_marker",
        "best",
        "expected_score",
        "reset",
    ),
    [
        (None, -4.0, 20.0, 20.0, 9, "g1", 0.0, 0),
        (-3.0, None, -2.0, 0.0, 5, "g0", -3.0, 0),
        (-3.0, None, -2.0, 0.0, 12, "g0", -3.0, 12),
    ],
)
def test_preview_upgrade_reconstructs_watermark_independently_of_ranking(
    config,
    seed_score,
    weak_score,
    pending_score,
    preview_score,
    preview_marker,
    best,
    expected_score,
    reset,
):
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.add(_program(0, seed_score))
        if seed_score is None:
            db.add(_program(1, -3.0))
        assert db.best_score_ever == expected_score
        if reset:
            assert db.check_and_spawn_island_if_stagnant(reset)
        for generation, score in ((5, weak_score), (9, pending_score)):
            db.add(
                _program(
                    generation, score, {"postprocess_db_maintenance_applied": False}
                ),
                defer_maintenance=True,
            )
        with db.conn:
            db.conn.execute(
                "DELETE FROM metadata_store WHERE key LIKE 'run_state_reconciled_%'"
            )
            db.conn.executemany(
                "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
                [
                    ("run_state_reconciled_v1", "1"),
                    ("best_program_id", "g9"),
                    ("best_score_generation", str(preview_marker)),
                    ("best_score_ever", str(preview_score)),
                ],
            )

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        assert db.best_program_id == best
        assert (db.best_score_ever, db.best_score_generation) == (expected_score, reset)
        db.run_post_add_maintenance(db.get("g5"))
        assert db.island_manager.get_island_populations().get(1) == 1
        db.run_post_add_maintenance(db.get("g9"))
        assert db.best_program_id == "g9"
        assert (db.best_score_ever, db.best_score_generation) == (
            pending_score,
            max(9, reset),
        )
