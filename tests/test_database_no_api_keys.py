import asyncio
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase


def _program(program_id: str) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=0,
    )


def test_program_database_init_without_openai_key(monkeypatch):
    """DB construction should not require API credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "no_key_init.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        try:
            db.add(_program("p0"))
            assert db.get("p0") is not None
        finally:
            db.close()


def test_program_database_can_defer_post_add_maintenance(monkeypatch):
    """Deferred maintenance should keep insert hot path separate from archive updates."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "deferred_maintenance.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
        )
        try:
            program = _program("p0")

            db.add(program, defer_maintenance=True)

            db.cursor.execute("SELECT COUNT(*) FROM archive")
            assert db.cursor.fetchone()[0] == 0
            assert db.best_program_id is None

            db.run_post_add_maintenance(program)

            db.cursor.execute("SELECT COUNT(*) FROM archive")
            assert db.cursor.fetchone()[0] == 1
            assert db.best_program_id == "p0"
        finally:
            db.close()


def test_async_db_add_without_openai_key_when_embeddings_disabled(monkeypatch):
    """Async wrapper should preserve disabled embedding mode in worker DBs."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "no_key_async.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                await async_db.add_program_async(_program("async-p0"))
                assert sync_db.get("async-p0") is not None
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_add_forwards_verbose_flag(monkeypatch):
    """Async add should forward verbose to the underlying writer database."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    observed = {}
    original_add = ProgramDatabase.add

    def tracking_add(self, program, verbose=False, defer_maintenance=False):
        observed["verbose"] = verbose
        observed["defer_maintenance"] = defer_maintenance
        return original_add(
            self,
            program,
            verbose=verbose,
            defer_maintenance=defer_maintenance,
        )

    monkeypatch.setattr(ProgramDatabase, "add", tracking_add)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "verbose_forwarding.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                await async_db.add_program_async(_program("async-p0"), verbose=True)
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())

    assert observed == {"verbose": True, "defer_maintenance": False}


def test_async_db_add_skips_duplicate_source_job_id(monkeypatch):
    """Async DB writes should be idempotent for the same completed scheduler job."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "duplicate_source_job.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                first = _program("async-p0")
                first.metadata = {"source_job_id": "job-123"}
                second = _program("async-p1")
                second.metadata = {"source_job_id": "job-123"}

                await async_db.add_program_async(first)
                await async_db.add_program_async(second)

                assert sync_db.get("async-p0") is not None
                assert sync_db.get("async-p1") is None
                assert sync_db._count_programs_in_db() == 1
                assert sync_db.has_program_with_source_job_id("job-123") is True
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_source_job_id_check_treats_inflight_insert_as_existing(monkeypatch):
    """Retries should see an in-flight source_job_id before commit finishes."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "inflight_source_job.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                async_db._in_flight_source_job_ids.add("job-123")
                assert await async_db.has_program_with_source_job_id_async("job-123")
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_can_fetch_program_by_source_job_id(monkeypatch):
    """Async DB should recover the already-persisted row for retry side effects."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fetch_source_job.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                program = _program("async-p0")
                program.metadata = {"source_job_id": "job-123"}

                await async_db.add_program_async(program)

                recovered = await async_db.get_program_by_source_job_id_async("job-123")

                assert recovered is not None
                assert recovered.id == "async-p0"
                assert recovered.metadata["source_job_id"] == "job-123"
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_add_skips_source_job_id_while_another_insert_is_inflight(monkeypatch):
    """Do not insert a duplicate row while the same source job is still in flight."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "inflight_duplicate_source_job.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                async_db._in_flight_source_job_ids.add("job-123")
                duplicate = _program("async-p1")
                duplicate.metadata = {"source_job_id": "job-123"}

                await async_db.add_program_async(duplicate)

                assert sync_db.get("async-p1") is None
                assert sync_db._count_programs_in_db() == 0
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_reuses_writer_database_for_multiple_adds(monkeypatch):
    """Async DB should keep one long-lived writer DB instead of reopening per add."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "writer_reuse.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            original_factory = async_db._create_writer_program_db
            writer_db_ids = []

            def tracking_factory():
                writer_db = original_factory()
                writer_db_ids.append(id(writer_db))
                return writer_db

            async_db._create_writer_program_db = tracking_factory
            try:
                await async_db.add_program_async(_program("async-p0"))
                await async_db.add_program_async(_program("async-p1"))

                assert len(writer_db_ids) == 1
                assert sync_db.get("async-p0") is not None
                assert sync_db.get("async-p1") is not None
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_can_defer_program_maintenance(monkeypatch):
    """Async hot-path adds can skip archive maintenance until explicitly replayed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "async_deferred_maintenance.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                program = _program("async-p0")

                await async_db.add_program_async(program, defer_maintenance=True)

                sync_db.cursor.execute("SELECT COUNT(*) FROM archive")
                assert sync_db.cursor.fetchone()[0] == 0
                assert sync_db.best_program_id is None

                await async_db.run_program_maintenance_async(program)

                sync_db.cursor.execute("SELECT COUNT(*) FROM archive")
                assert sync_db.cursor.fetchone()[0] == 1
                assert sync_db.best_program_id == "async-p0"
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_batches_deferred_program_maintenance(monkeypatch):
    """Deferred maintenance should not flush until forced when below batch size."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "async_batched_maintenance.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                first = _program("async-p0")
                second = _program("async-p1")
                second.generation = 1

                await async_db.add_program_async(first, defer_maintenance=True)
                await async_db.add_program_async(second, defer_maintenance=True)
                async_db.enqueue_program_maintenance(first)
                async_db.enqueue_program_maintenance(second)

                assert async_db.pending_program_maintenance_count() == 2

                await async_db.flush_program_maintenance_async(force=False)
                sync_db.cursor.execute("SELECT COUNT(*) FROM archive")
                assert sync_db.cursor.fetchone()[0] == 0

                await async_db.flush_program_maintenance_async(force=True)
                sync_db.cursor.execute("SELECT COUNT(*) FROM archive")
                assert sync_db.cursor.fetchone()[0] == 2
                assert async_db.pending_program_maintenance_count() == 0
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_flush_forwards_verbose_flag(monkeypatch):
    """Deferred maintenance flush should forward verbose to maintenance replay."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    observed = {}
    original_run = ProgramDatabase.run_post_add_maintenance_batch

    def tracking_run(self, programs, verbose=False, recompute_embeddings=False):
        observed["verbose"] = verbose
        observed["count"] = len(programs)
        return original_run(
            self,
            programs,
            verbose=verbose,
            recompute_embeddings=recompute_embeddings,
        )

    monkeypatch.setattr(ProgramDatabase, "run_post_add_maintenance_batch", tracking_run)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "verbose_flush.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                program = _program("async-p0")
                await async_db.add_program_async(program, defer_maintenance=True)
                async_db.enqueue_program_maintenance(program)

                await async_db.flush_program_maintenance_async(
                    force=True,
                    verbose=True,
                )
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())

    assert observed == {"verbose": True, "count": 1}
