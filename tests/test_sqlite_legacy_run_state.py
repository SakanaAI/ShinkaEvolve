import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase


@pytest.fixture
def config(tmp_path: Path) -> DatabaseConfig:
    return DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)


def _open_db(config: DatabaseConfig) -> ProgramDatabase:
    return ProgramDatabase(config, embedding_model="")


def _program(generation: int, score: float | None, *, correct: bool = True) -> Program:
    return Program(
        id=f"g{generation}",
        generation=generation,
        combined_score=score,
        correct=correct,
        timestamp=float(generation),
        code="pass",
        complexity=1.0,
        island_idx=0,
    )


def _write_legacy_run_state(
    db: ProgramDatabase, best: Program, *, marker: int, score: float | None = None
):
    with db.conn:
        db.conn.execute(
            "DELETE FROM metadata_store "
            "WHERE key IN ('run_state_reconciled_v1', 'run_state_reconciled_v2')"
        )
        db.conn.executemany(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            [
                ("last_iteration", str(best.generation)),
                ("best_program_id", best.id),
                ("best_score_generation", str(marker)),
                (
                    "best_score_ever",
                    str(best.combined_score if score is None else score),
                ),
            ],
        )


@pytest.mark.parametrize(
    "legacy_score, legacy_marker, expected_marker",
    [(1.0, 9, 9), (1.0, 1, 5), (10.0, 1, 5), (10.0, 5, 5)],
)
def test_writable_reopen_repairs_legacy_run_state_from_original_rows(
    config, legacy_score, legacy_marker, expected_marker
):
    with closing(_open_db(config)) as writer:
        stale, best = _program(1, 1.0), _program(8, 10.0)
        best.public_metrics = {"quality": 2.0}
        lower_metric = _program(6, 10.0)
        lower_metric.public_metrics = {"quality": 1.0}
        lower_metric.timestamp = 10.0
        earlier_tie = _program(5, 10.0)
        earlier_tie.public_metrics = best.public_metrics.copy()
        unscored = _program(4, 0.0)
        unscored.combined_score = None
        unscored.public_metrics = {"quality": 1000.0}
        island_copy = _program(20, 100.0)
        island_copy.metadata = {"_is_island_copy": True}
        spawned_copy = _program(21, 100.0)
        spawned_copy.metadata = {"_spawned_island": True}
        for program in (
            stale,
            earlier_tie,
            best,
            unscored,
            lower_metric,
            _program(7, 999.0, correct=False),
            island_copy,
            spawned_copy,
        ):
            writer.add(program, defer_maintenance=True)
        _write_legacy_run_state(writer, stale, marker=legacy_marker, score=legacy_score)

    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert (reader.last_iteration, reader.get_best_program().id) == (1, stale.id)

    with closing(_open_db(config)) as repaired:
        assert (repaired.last_iteration, repaired.get_best_program().id) == (8, best.id)
        assert (repaired.best_score_ever, repaired.best_score_generation) == (
            10.0,
            expected_marker,
        )
        repaired.add(_program(3, 2.0))

    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert (reader.last_iteration, reader.get_best_program().id) == (8, best.id)


def test_legacy_repair_rolls_back_and_retries_on_next_open(config):
    stale, best = _program(1, 1.0), _program(8, 10.0)
    with closing(_open_db(config)) as writer:
        writer.add(stale)
        writer.add(best)
        _write_legacy_run_state(writer, stale, marker=1)
        writer.conn.execute("""
            CREATE TRIGGER fail_legacy_repair BEFORE INSERT ON metadata_store
            WHEN NEW.key = 'run_state_reconciled_v2'
            BEGIN SELECT RAISE(ABORT, 'legacy run state repair failed'); END
            """)

    with pytest.raises(sqlite3.IntegrityError, match="legacy run state repair failed"):
        with closing(_open_db(config)):
            pass

    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert (reader.last_iteration, reader.best_program_id) == (1, stale.id)
        assert (reader.best_score_ever, reader.best_score_generation) == (1.0, 1)

    with closing(sqlite3.connect(config.db_path)) as conn:
        conn.execute("DROP TRIGGER fail_legacy_repair")
    with closing(_open_db(config)) as repaired:
        assert (repaired.last_iteration, repaired.best_program_id) == (8, best.id)
        assert (repaired.best_score_ever, repaired.best_score_generation) == (10.0, 8)


@pytest.mark.parametrize("legacy", [False, True])
def test_worker_reopen_preserves_deferred_best_in_current_database(config, legacy):
    seed, pending = _program(1, 1.0), _program(2, 10.0)
    with closing(_open_db(config)) as writer:
        writer.add(seed)
        if legacy:
            _write_legacy_run_state(writer, seed, marker=seed.generation)

    with closing(_open_db(config)) as writer:
        writer.add(pending, defer_maintenance=True)

    with closing(_open_db(config)) as worker:
        assert worker.last_iteration == pending.generation
        assert worker.get_best_program().id == seed.id
        worker.run_post_add_maintenance(pending)
        assert worker.get_best_program().id == pending.id


def test_interleaved_legacy_writes_repair_marker_with_correct_best_id(config):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 5
    with closing(_open_db(config)) as newer, closing(_open_db(config)) as older:
        for program in (_program(0, 0.0), _program(1, 1.0), _program(8, 10.0)):
            newer.add(program, defer_maintenance=True)
        with newer.conn:
            newer.conn.execute(
                "DELETE FROM metadata_store "
                "WHERE key IN ('run_state_reconciled_v1', 'run_state_reconciled_v2')"
            )

        # Legacy writers committed these three fields independently.
        for writer, key, value in (
            (older, "best_program_id", "g1"),
            (newer, "best_program_id", "g8"),
            (newer, "best_score_generation", "8"),
            (older, "best_score_generation", "1"),
            (older, "best_score_ever", "1.0"),
            (newer, "best_score_ever", "10.0"),
        ):
            writer._update_metadata_in_db(key, value)

    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert (reader.best_program_id, reader.best_score_ever) == ("g8", 10.0)
        assert reader.best_score_generation == 1

    with closing(_open_db(config)) as repaired:
        assert (repaired.best_program_id, repaired.best_score_ever) == ("g8", 10.0)
        assert repaired.best_score_generation == 8
        assert not repaired.check_and_spawn_island_if_stagnant(9)
        assert repaired.check_and_spawn_island_if_stagnant(13)

    with closing(_open_db(config)) as reopened:
        assert reopened.best_score_generation == 13
        assert not reopened.check_and_spawn_island_if_stagnant(13)


@pytest.mark.parametrize("tie_order", [(5, 8), (8, 5)])
@pytest.mark.parametrize(
    "legacy_score, legacy_marker, expected_marker",
    [(1.0, 1, 5), (10.0, 1, 5), (10.0, 5, 5), (10.0, 6, 6), (10.0, 9, 9)],
)
def test_legacy_marker_uses_earliest_original_score_without_losing_resets(
    config, tie_order, legacy_score, legacy_marker, expected_marker
):
    with closing(_open_db(config)) as writer:
        best = _program(8, 10.0)
        best.public_metrics = {"quality": 2.0}
        earlier_tie = _program(5, 10.0)
        earlier_tie.public_metrics = {"quality": 1.0}
        ties = {5: earlier_tie, 8: best}
        for generation in tie_order:
            writer.add(ties[generation], defer_maintenance=True)
        for generation, flag in ((1, "_is_island_copy"), (2, "_spawned_island")):
            copy = _program(generation, 10.0)
            copy.metadata = {flag: True}
            writer.add(copy, defer_maintenance=True)
        writer.add(_program(3, 10.0, correct=False), defer_maintenance=True)
        _write_legacy_run_state(writer, best, marker=legacy_marker, score=legacy_score)

    with closing(_open_db(config)) as repaired:
        assert (repaired.best_program_id, repaired.best_score_ever) == (best.id, 10.0)
        assert repaired.best_score_generation == expected_marker


@pytest.mark.parametrize("best_score", [-2.0, 0.0])
def test_legacy_marker_handles_negative_and_zero_scores(config, best_score):
    with closing(_open_db(config)) as writer:
        best = _program(8, best_score)
        writer.add(best, defer_maintenance=True)
        _write_legacy_run_state(writer, best, marker=1)

    with closing(_open_db(config)) as repaired:
        assert (repaired.best_score_ever, repaired.best_score_generation) == (
            best_score,
            8,
        )


def test_legacy_marker_preserves_zero_score_established_by_unscored_program(config):
    with closing(_open_db(config)) as writer:
        best = _program(8, 0.0)
        for program in (_program(1, None), _program(4, -1.0), best):
            writer.add(program, defer_maintenance=True)
        _write_legacy_run_state(writer, best, marker=1)

    with closing(_open_db(config)) as repaired:
        assert (repaired.best_program_id, repaired.best_score_ever) == ("g8", 0.0)
        assert repaired.best_score_generation == 1


def test_legacy_marker_preserves_higher_historical_score(config):
    with closing(_open_db(config)) as writer:
        best = _program(8, 10.0)
        writer.add(best, defer_maintenance=True)
        _write_legacy_run_state(writer, best, marker=1, score=20.0)

    with closing(_open_db(config)) as repaired:
        assert (repaired.best_score_ever, repaired.best_score_generation) == (20.0, 1)


@pytest.mark.parametrize("legacy_marker, expected_marker", [(0, 1), (6, 6)])
@pytest.mark.parametrize("insert_scored_first", [False, True])
def test_legacy_unscored_improvement_is_independent_of_negative_ranked_winner(
    config, legacy_marker, expected_marker, insert_scored_first
):
    with closing(_open_db(config)) as writer:
        unscored, best = _program(1, None), _program(8, -2.0)
        if insert_scored_first:
            writer.add(best, defer_maintenance=True)
        writer.add(unscored)
        if insert_scored_first:
            writer.run_post_add_maintenance(best)
        else:
            writer.add(best)
        assert (writer.best_score_ever, writer.best_score_generation) == (0.0, 1)
        _write_legacy_run_state(writer, best, marker=legacy_marker, score=0.0)

    with closing(_open_db(config)) as repaired:
        assert repaired.get_best_program().id == best.id
        assert (repaired.best_score_ever, repaired.best_score_generation) == (
            0.0,
            expected_marker,
        )


@pytest.mark.parametrize("insert_unscored_first", [False, True])
@pytest.mark.parametrize("replay_scored_maintenance", [False, True])
def test_legacy_negative_record_ignores_unscored_challenger(
    config, insert_unscored_first, replay_scored_maintenance
):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 5
    seed, challenger = _program(0, -3.0), _program(2, None)
    with closing(_open_db(config)) as writer:
        if insert_unscored_first:
            writer.add(challenger, defer_maintenance=True)
        writer.add(seed)
        if insert_unscored_first:
            writer.run_post_add_maintenance(challenger)
        else:
            writer.add(challenger)
        if replay_scored_maintenance:
            writer.run_post_add_maintenance(seed)
        assert (writer.best_score_ever, writer.best_score_generation) == (-3.0, 0)
        _write_legacy_run_state(writer, seed, marker=0)

    with closing(_open_db(config)) as repaired:
        assert repaired.best_program_id == seed.id
        assert (repaired.best_score_ever, repaired.best_score_generation) == (-3.0, 0)
        repaired.add(_program(7, -2.0))
        assert repaired.island_manager.get_island_populations() == {0: 3}

    with closing(_open_db(config)) as reopened:
        assert reopened.best_program_id == "g7"
        assert (reopened.best_score_ever, reopened.best_score_generation) == (-2.0, 7)


@pytest.mark.parametrize("preview_marker, expected_marker", [(1, 8), (12, 12)])
@pytest.mark.parametrize("maintenance_recorded", [False, True])
def test_preview_repair_upgrades_once_across_two_reopens(
    config, monkeypatch, preview_marker, expected_marker, maintenance_recorded
):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 5
    with closing(_open_db(config)) as writer:
        best = _program(8, 10.0)
        writer.add(best)
        if not maintenance_recorded:
            # Older maintained rows and partial maintenance lack the completion flag.
            with writer.conn:
                writer.conn.execute(
                    "UPDATE programs SET metadata = '{}' WHERE id = ?", (best.id,)
                )
        _write_legacy_run_state(writer, best, marker=preview_marker)
        writer._update_metadata_in_db("run_state_reconciled_v1", "1")

    scans = []
    scan_originals = ProgramDatabase._scan_original_run_programs

    def track_scan(db, published, programs):
        scans.append(published.best_program_id)
        return scan_originals(db, published, programs)

    monkeypatch.setattr(ProgramDatabase, "_scan_original_run_programs", track_scan)
    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert reader.best_score_generation == preview_marker
        assert not scans

    for _ in range(2):
        with closing(_open_db(config)) as repaired:
            assert (repaired.best_program_id, repaired.best_score_ever) == (
                best.id,
                10.0,
            )
            assert repaired.best_score_generation == expected_marker
            assert not repaired.check_and_spawn_island_if_stagnant(9)
            assert (
                repaired.conn.execute(
                    "SELECT value FROM metadata_store WHERE key = 'run_state_reconciled_v2'"
                ).fetchone()[0]
                == "1"
            )
        assert scans == [best.id]


@pytest.mark.parametrize("watermark", [5, 7])
def test_preview_repair_preserves_markerless_tied_incumbent_watermark(config, watermark):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 2
    config.island_spawn_strategy = "best"
    first, best = _program(5, 10.0), _program(8, 10.0)
    with closing(_open_db(config)) as writer:
        writer.add(first)
        if watermark == 7:
            assert writer.check_and_spawn_island_if_stagnant(watermark)
        config.stagnation_threshold = 4
        writer.add(best)
        with writer.conn:
            writer.conn.execute(
                "UPDATE programs SET metadata = '{}' WHERE id IN (?, ?)",
                (first.id, best.id),
            )
        assert writer.best_score_generation == watermark
        _write_legacy_run_state(writer, best, marker=watermark)
        writer._update_metadata_in_db("run_state_reconciled_v1", "1")

    with closing(_open_db(config)) as repaired:
        assert repaired.best_program_id == best.id
        assert (repaired.best_score_ever, repaired.best_score_generation) == (
            10.0,
            watermark,
        )
        assert repaired.check_and_spawn_island_if_stagnant(watermark + 4)


def test_preview_repair_keeps_reset_after_markerless_unscored_incumbent(config):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 2
    config.island_spawn_strategy = "best"
    seed, unscored, best = _program(0, -3.0), _program(5, None), _program(8, 0.0)
    with closing(_open_db(config)) as writer:
        writer.add(seed, defer_maintenance=True)
        writer.add(unscored)
        writer.run_post_add_maintenance(seed)
        writer.add(_program(7, -4.0))
        writer.add(best)
        assert (writer.best_score_ever, writer.best_score_generation) == (0.0, 7)
        assert writer.island_manager.get_island_populations() == {0: 4, 1: 1}
        with writer.conn:
            writer.conn.execute(
                "UPDATE programs SET metadata = '{}' WHERE id IN (?, ?)",
                (unscored.id, best.id),
            )
        _write_legacy_run_state(writer, best, marker=7)
        writer._update_metadata_in_db("run_state_reconciled_v1", "1")

    with closing(_open_db(config)) as repaired:
        assert repaired.best_program_id == best.id
        assert repaired.best_score_generation == 7
        assert repaired.check_and_spawn_island_if_stagnant(9)


def test_preview_repair_preserves_deferred_maintenance_on_reopen(config):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 5
    seed, pending = _program(0, 1.0), _program(9, 10.0)
    with closing(_open_db(config)) as writer:
        writer.add(seed)
        writer.conn.execute(
            "DELETE FROM metadata_store WHERE key = 'run_state_reconciled_v2'"
        )
        writer._update_metadata_in_db("run_state_reconciled_v1", "1")
        writer.add(pending, defer_maintenance=True)

    with closing(_open_db(config)) as repaired:
        assert repaired.last_iteration == 9
        assert repaired.get_best_program().id == seed.id
        assert (repaired.best_score_ever, repaired.best_score_generation) == (1.0, 0)
        assert {program.id for program in repaired._get_archive_programs()} == {seed.id}
        assert repaired.check_and_spawn_island_if_stagnant(5)
        repaired.run_post_add_maintenance(repaired.get(pending.id))

    with closing(_open_db(config)) as reopened:
        assert reopened.get_best_program().id == pending.id
        assert (reopened.best_score_ever, reopened.best_score_generation) == (10.0, 9)


def test_preview_repair_rolls_back_new_marker_and_retries(config):
    with closing(_open_db(config)) as writer:
        best = _program(8, 10.0)
        writer.add(best)
        _write_legacy_run_state(writer, best, marker=1)
        writer._update_metadata_in_db("run_state_reconciled_v1", "1")
        writer.conn.execute("""
            CREATE TRIGGER fail_preview_repair BEFORE INSERT ON metadata_store
            WHEN NEW.key = 'run_state_reconciled_v2'
            BEGIN SELECT RAISE(ABORT, 'preview repair failed'); END
            """)

    with pytest.raises(sqlite3.IntegrityError, match="preview repair failed"):
        with closing(_open_db(config)):
            pass

    with closing(ProgramDatabase(config, read_only=True)) as reader:
        assert (reader.best_program_id, reader.best_score_ever) == (best.id, 10.0)
        assert reader.best_score_generation == 1
        assert dict(
            reader.conn.execute(
                "SELECT key, value FROM metadata_store "
                "WHERE key IN ('run_state_reconciled_v1', 'run_state_reconciled_v2')"
            )
        ) == {"run_state_reconciled_v1": "1"}

    with closing(sqlite3.connect(config.db_path)) as conn:
        conn.execute("DROP TRIGGER fail_preview_repair")
    with closing(_open_db(config)) as repaired:
        assert repaired.best_score_generation == 8
        assert (
            repaired.conn.execute(
                "SELECT value FROM metadata_store WHERE key = 'run_state_reconciled_v2'"
            ).fetchone()[0]
            == "1"
        )
