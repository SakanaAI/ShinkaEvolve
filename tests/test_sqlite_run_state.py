import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase


@pytest.fixture
def config(tmp_path: Path) -> DatabaseConfig:
    return DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)


def _open_db(config: DatabaseConfig) -> ProgramDatabase:
    return ProgramDatabase(config, embedding_model="")


def _program(generation: int, score: float, *, correct: bool = True) -> Program:
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


@pytest.mark.parametrize("deferred", [False, True])
def test_out_of_order_writes_keep_latest_iteration_and_best(config, deferred):
    with closing(_open_db(config)) as newer:
        newer.add(_program(0, 0.0))
        with closing(_open_db(config)) as older:
            assert older.conn is not newer.conn
            high, low = _program(2, 10.0), _program(1, 1.0)
            newer.add(high, defer_maintenance=deferred)
            older.add(low, defer_maintenance=deferred)
            if deferred:
                newer.run_post_add_maintenance(high)
                older.run_post_add_maintenance(low)

            with closing(_open_db(config)) as reopened:
                assert sorted(p.generation for p in reopened.get_all_programs()) == [
                    0,
                    1,
                    2,
                ]
                assert (
                    reopened.last_iteration,
                    reopened.get_best_program().combined_score,
                    reopened.best_score_ever,
                    reopened.best_score_generation,
                ) == (2, 10.0, 10.0, 2)
            assert (older.last_iteration, older.best_program_id) == (2, high.id)


def test_stale_save_and_best_lookup_follow_committed_state(config):
    with closing(_open_db(config)) as writer:
        seed = _program(0, 1.0)
        seed.public_metrics = {"speed": 2.0}
        writer.add(seed)
        with closing(_open_db(config)) as stale:
            best = _program(3, 10.0)
            best.public_metrics = {"speed": 1.0}
            writer.add(best)
            stale.save()

            assert stale.get_best_program(metric="speed").id == seed.id
            assert stale.get_best_program().id == best.id
            with closing(_open_db(config)) as reopened:
                assert reopened.last_iteration == 3


def test_save_cannot_publish_an_older_snapshot_than_a_worker(config, monkeypatch):
    save_committed = threading.Event()
    worker_published_or_waiting = threading.Event()

    class PublicationLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                worker_published_or_waiting.set()
                self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()

    async def run():
        with closing(_open_db(config)) as sync_db:
            first, second = _program(1, 1.0), _program(2, 10.0)
            sync_db.add(first)
            db = AsyncProgramDatabase(sync_db)
            publish = sync_db._cache_run_state

            def pause_save_publication(state):
                if state.last_iteration == first.generation:
                    save_committed.set()
                    assert worker_published_or_waiting.wait(5)
                publish(state)

            def add_from_worker():
                assert save_committed.wait(5)
                with closing(_open_db(config)) as worker_db:
                    worker_db.add(second)
                    db._refresh_run_state_from_worker(worker_db)
                worker_published_or_waiting.set()

            monkeypatch.setattr(
                sync_db, "_run_state_lock", PublicationLock(), raising=False
            )
            monkeypatch.setattr(sync_db, "_cache_run_state", pause_save_publication)
            worker = db.write_executor.submit(add_from_worker)
            try:
                sync_db.save()
                await asyncio.wait_for(asyncio.wrap_future(worker), 5)
                with closing(ProgramDatabase(config, read_only=True)) as observer:
                    assert (observer.last_iteration, observer.best_program_id) == (
                        2,
                        second.id,
                    )
                assert (sync_db.last_iteration, sync_db.best_program_id) == (
                    2,
                    second.id,
                )
            finally:
                await db.close_async()

    asyncio.run(run())


def test_save_commits_pending_changes_on_its_own_connection(config):
    with closing(_open_db(config)) as writer:
        writer.conn.execute(
            "INSERT INTO metadata_store (key, value) VALUES ('beam_search_parent_id', 'parent')"
        )
        writer.last_iteration = 3
        writer.save()

        with closing(_open_db(config)) as reopened:
            assert reopened.beam_search_parent_id == "parent"
            assert reopened.last_iteration == 3


@pytest.mark.parametrize("explicit_transaction", [False, True])
def test_save_refreshes_with_an_open_read_cursor(config, explicit_transaction):
    async def run():
        with closing(_open_db(config)) as sync_db, closing(_open_db(config)) as worker:
            first, best = _program(8, 1.0), _program(7, 10.0)
            sync_db.add(first)
            db = AsyncProgramDatabase(sync_db)
            try:
                if explicit_transaction:
                    sync_db.conn.execute("BEGIN")
                snapshot = sync_db.conn.execute(
                    "SELECT key, value FROM metadata_store ORDER BY key"
                )
                assert snapshot.fetchone() is not None
                assert sync_db._read_run_state().best_program_id == first.id
                worker.add(best)
                db._refresh_run_state_from_worker(worker)
                assert sync_db.best_program_id == best.id

                sync_db.save()

                assert not sync_db.conn.in_transaction
                assert (sync_db.last_iteration, sync_db.best_program_id) == (8, best.id)
                assert snapshot.fetchone() is not None
                snapshot.close()
            finally:
                await db.close_async()

    asyncio.run(run())


def _write_legacy_run_state(
    db: ProgramDatabase, best: Program, *, marker: int, score: float | None = None
):
    with db.conn:
        db.conn.execute(
            "DELETE FROM metadata_store WHERE key = 'run_state_reconciled_v1'"
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
    [(1.0, 9, 9), (10.0, 1, 8), (10.0, 5, 5)],
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
            WHEN NEW.key = 'run_state_reconciled_v1'
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


def test_best_lookup_before_deferred_maintenance_still_records_improvement(config):
    with closing(_open_db(config)) as writer, closing(_open_db(config)) as reader:
        best = _program(2, 10.0)
        writer.add(best, defer_maintenance=True)
        assert reader.get_best_program().id == best.id

        writer.run_post_add_maintenance(best)
        assert (writer.best_score_ever, writer.best_score_generation) == (10.0, 2)


def test_delayed_improvement_and_ties_keep_stagnation_marker_monotonic(config):
    with closing(_open_db(config)) as newer, closing(_open_db(config)) as older:
        newer.add(_program(8, 10.0))
        better = _program(3, 11.0)
        older.add(better, defer_maintenance=True)
        older.run_post_add_maintenance(better)

        equal_score = _program(4, 11.0)
        equal_score.public_metrics = {"quality": 1.0}
        newer.add(equal_score)
        older.add(_program(9, 11.0))
        older.add(_program(10, 99.0, correct=False))
        assert older.get_best_program().id == equal_score.id

        later_tie = _program(6, 11.0)
        later_tie.public_metrics = {"quality": 1.0}
        newer.add(later_tie)

        with closing(_open_db(config)) as reopened:
            assert reopened.get_best_program().id == later_tie.id
            assert (reopened.best_score_ever, reopened.best_score_generation) == (
                11.0,
                8,
            )


def test_stale_worker_checks_stagnation_against_latest_improvement(config):
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 3
    with closing(_open_db(config)) as newer:
        newer.add(_program(0, 1.0))
        with closing(_open_db(config)) as older:
            newer.add(_program(8, 10.0))
            assert not older.check_and_spawn_island_if_stagnant(4)
            assert older.check_and_spawn_island_if_stagnant(11)
            assert not newer.check_and_spawn_island_if_stagnant(11)


@pytest.mark.parametrize("key", ["last_iteration", "best_score_ever"])
def test_metadata_failure_rolls_back_and_can_retry(config, key):
    config.archive_size = 2
    with closing(_open_db(config)) as writer, closing(_open_db(config)) as observer:
        seed, best = _program(0, 1.0), _program(2, 10.0)
        best.parent_id = seed.id
        writer.add(seed)

        assert writer.conn is not None
        writer.conn.execute(f"""
            CREATE TRIGGER fail_metadata BEFORE INSERT ON metadata_store
            WHEN NEW.key = '{key}'
            BEGIN SELECT RAISE(ABORT, 'metadata write failed'); END
            """)

        with pytest.raises(sqlite3.IntegrityError, match="metadata write failed"):
            if key == "last_iteration":
                writer.add(best, defer_maintenance=True)
            else:
                writer.add(best)

        assert not writer.conn.in_transaction
        assert writer.best_program_id == seed.id
        assert observer.get_best_program().id == seed.id
        if key == "last_iteration":
            assert writer.last_iteration == 0
            assert observer.get(best.id) is None
            assert observer.get(seed.id).children_count == 0
        else:
            assert writer.best_score_ever == 1.0
            assert writer.best_score_generation == 0
            assert observer.get(best.id) is not None

        writer.conn.execute("DROP TRIGGER fail_metadata")
        if key == "last_iteration":
            writer.add(best, defer_maintenance=True)
        writer.run_post_add_maintenance(best)
        assert observer.get_best_program().id == best.id
        assert observer.get(seed.id).children_count == 1


def test_failed_stagnation_reset_rolls_back_spawn(config):
    config.archive_size = 3
    config.enable_dynamic_islands = True
    config.stagnation_threshold = 3
    with closing(_open_db(config)) as writer, closing(_open_db(config)) as observer:
        seed, weak, delayed = _program(0, 10.0), _program(1, 1.0), _program(4, 5.0)
        writer.add(seed)
        writer.add(weak)
        writer.add(delayed, defer_maintenance=True)
        assert writer.conn is not None
        writer.conn.execute("""
            CREATE TRIGGER fail_stagnation_reset BEFORE INSERT ON metadata_store
            WHEN NEW.key = 'best_score_generation'
            BEGIN SELECT RAISE(ABORT, 'stagnation reset failed'); END
            """)

        with pytest.raises(sqlite3.IntegrityError, match="stagnation reset failed"):
            writer.run_post_add_maintenance(delayed)

        assert not writer.conn.in_transaction
        assert observer.island_manager.get_island_populations() == {0: 3}
        assert observer.is_stagnant(4)

        writer.conn.execute("DROP TRIGGER fail_stagnation_reset")
        writer.run_post_add_maintenance(delayed)
        assert observer.island_manager.get_island_populations() == {0: 3, 1: 1}
        assert {
            p.id for p in observer._get_archive_programs() if p.island_idx == 0
        } == {
            seed.id,
            weak.id,
            delayed.id,
        }
        assert not observer.is_stagnant(4)


def test_async_cache_keeps_equal_score_winner_when_worker_finishes_late(
    config, monkeypatch
):
    first, second = _program(1, 10.0), _program(2, 10.0)
    first_committed = threading.Event()
    release_first = threading.Event()
    original_add = ProgramDatabase.add

    def delay_first(db, program, **kwargs):
        result = original_add(db, program, **kwargs)
        if program is first:
            first_committed.set()
            assert release_first.wait(5)
        return result

    monkeypatch.setattr(ProgramDatabase, "add", delay_first)

    async def run():
        with closing(_open_db(config)) as sync_db:
            db = AsyncProgramDatabase(sync_db, max_workers=2)
            worker = asyncio.create_task(db.add_program_async(first))
            try:
                assert await asyncio.to_thread(first_committed.wait, 5)
                assert await db.add_program_async(second)
            finally:
                release_first.set()
                try:
                    assert await worker
                finally:
                    await db.close_async()

            assert (sync_db.last_iteration, sync_db.best_program_id) == (2, second.id)
            assert (sync_db.best_score_ever, sync_db.best_score_generation) == (10.0, 1)

    asyncio.run(run())


def test_async_cache_reflects_best_commit_before_later_maintenance_failure(config):
    async def run():
        with closing(_open_db(config)) as sync_db:
            db = AsyncProgramDatabase(sync_db)
            best = _program(2, 10.0)
            try:
                assert await db.add_program_async(best, defer_maintenance=True)
                assert sync_db.conn is not None
                sync_db.conn.execute("""
                    CREATE TRIGGER fail_maintenance BEFORE UPDATE OF metadata ON programs
                    BEGIN SELECT RAISE(ABORT, 'maintenance write failed'); END
                    """)

                with pytest.raises(
                    sqlite3.IntegrityError, match="maintenance write failed"
                ):
                    await db.run_program_maintenance_async(best)

                assert (sync_db.best_program_id, sync_db.best_score_ever) == (
                    best.id,
                    10.0,
                )
                sync_db.conn.execute("DROP TRIGGER fail_maintenance")
                await db.run_program_maintenance_async(best)
            finally:
                await db.close_async()

    asyncio.run(run())
