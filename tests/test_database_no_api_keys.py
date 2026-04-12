import asyncio
import tempfile
import threading
import time
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

    assert observed == {"verbose": True}


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


def test_async_db_can_record_attempt_events(monkeypatch):
    """Attempt log writes should work without API credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "attempt_log.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                await async_db.record_attempt_event_async(
                    generation=7,
                    stage="proposal",
                    status="failed",
                    details={"reason": "test"},
                )
                sync_db.cursor.execute(
                    "SELECT generation, stage, status FROM attempt_log"
                )
                rows = [tuple(row) for row in sync_db.cursor.fetchall()]
                assert rows == [(7, "proposal", "failed")]
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_uses_fresh_writer_database_per_add(monkeypatch):
    """Multi-writer async DB should build a fresh writer DB per add operation."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "writer_reuse.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            original_init = ProgramDatabase.__init__
            writer_db_ids = []

            def tracking_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                if kwargs.get("read_only", False) is False:
                    writer_db_ids.append(id(self))

            monkeypatch.setattr(ProgramDatabase, "__init__", tracking_init)
            try:
                await async_db.add_program_async(_program("async-p0"))
                await async_db.add_program_async(_program("async-p1"))

                assert len(writer_db_ids) >= 2
                assert sync_db.get("async-p0") is not None
                assert sync_db.get("async-p1") is not None
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_can_run_multiple_writes_concurrently(monkeypatch):
    """Async DB should allow multiple write tasks to overlap when workers > 1."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    active_adds = 0
    peak_adds = 0
    lock = threading.Lock()
    original_add = ProgramDatabase.add

    def tracking_add(self, program, verbose=False, defer_maintenance=False):
        nonlocal active_adds, peak_adds
        with lock:
            active_adds += 1
            peak_adds = max(peak_adds, active_adds)
        try:
            time.sleep(0.05)
            return original_add(
                self,
                program,
                verbose=verbose,
                defer_maintenance=defer_maintenance,
            )
        finally:
            with lock:
                active_adds -= 1

    monkeypatch.setattr(ProgramDatabase, "add", tracking_add)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent_writer.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db, max_workers=2)
            try:
                await asyncio.gather(
                    async_db.add_program_async(_program("async-p0")),
                    async_db.add_program_async(_program("async-p1")),
                )
                assert sync_db.get("async-p0") is not None
                assert sync_db.get("async-p1") is not None
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())

    assert peak_adds >= 2
