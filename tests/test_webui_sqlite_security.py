import errno
import os
import shutil
import sqlite3
import tempfile

import pytest

from shinka.webui.visualization import (
    DatabaseRequestHandler,
    DatabaseViewRaceError,
    PathValidationError,
)

requires_descriptor_traversal = pytest.mark.skipif(
    not DatabaseRequestHandler._supports_descriptor_traversal(),
    reason="database race hardening requires descriptor traversal",
)


def _handler(root):
    handler = object.__new__(DatabaseRequestHandler)
    handler.search_root = os.fspath(root)
    handler._canonical_search_root = os.path.realpath(root)
    return handler


def _create_stats_database(path, *, program_count=0):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE programs (
                generation INTEGER,
                correct INTEGER,
                combined_score REAL,
                timestamp REAL,
                metadata TEXT
            )
            """
        )
        for generation in range(program_count):
            connection.execute(
                "INSERT INTO programs VALUES (?, 1, 1.0, 1.0, '{}')",
                (generation,),
            )


@requires_descriptor_traversal
def test_database_stats_rejects_main_database_swap_after_validation(
    tmp_path, monkeypatch
):
    search_root = tmp_path / "served"
    run_dir = search_root / "run"
    run_dir.mkdir(parents=True)
    programs_db = run_dir / "programs.sqlite"
    _create_stats_database(programs_db)
    outside_db = tmp_path / "outside.sqlite"
    _create_stats_database(outside_db, program_count=1)
    handler = _handler(search_root)
    resolve_path = handler._resolve_within_root
    swapped = False

    def resolve_then_swap(path):
        nonlocal swapped
        resolved = resolve_path(path)
        if not swapped and os.fspath(path).endswith("programs.sqlite"):
            swapped = True
            programs_db.rename(run_dir / "parked.sqlite")
            programs_db.symlink_to(outside_db)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)
    handler.send_json_response = lambda data: None

    with pytest.raises(PathValidationError):
        handler.handle_get_database_stats("run/programs.sqlite")


@requires_descriptor_traversal
def test_database_stats_rejects_prompt_database_swap_after_validation(
    tmp_path, monkeypatch
):
    search_root = tmp_path / "served"
    run_dir = search_root / "run"
    run_dir.mkdir(parents=True)
    _create_stats_database(run_dir / "programs.sqlite")
    prompts_db = run_dir / "prompts.sqlite"
    with sqlite3.connect(prompts_db) as connection:
        connection.execute("CREATE TABLE system_prompts (metadata TEXT)")
    outside_db = tmp_path / "outside-prompts.sqlite"
    with sqlite3.connect(outside_db) as connection:
        connection.execute("CREATE TABLE system_prompts (metadata TEXT)")
        connection.execute("INSERT INTO system_prompts VALUES ('{}')")
    handler = _handler(search_root)
    resolve_path = handler._resolve_within_root
    swapped = False

    def resolve_then_swap(path):
        nonlocal swapped
        resolved = resolve_path(path)
        if not swapped and os.fspath(path).endswith("prompts.sqlite"):
            swapped = True
            prompts_db.rename(run_dir / "parked-prompts.sqlite")
            prompts_db.symlink_to(outside_db)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)
    handler.send_json_response = lambda data: None

    with pytest.raises(PathValidationError):
        handler.handle_get_database_stats("run/programs.sqlite")


@requires_descriptor_traversal
def test_program_details_propagates_database_swap_rejection(tmp_path, monkeypatch):
    search_root = tmp_path / "served"
    search_root.mkdir()
    programs_db = search_root / "programs.sqlite"
    programs_db.write_bytes(b"inside")
    outside_db = tmp_path / "outside.sqlite"
    outside_db.write_bytes(b"outside")
    handler = _handler(search_root)
    resolve_path = handler._resolve_within_root
    swapped = False

    def resolve_then_swap(path):
        nonlocal swapped
        resolved = resolve_path(path)
        if not swapped:
            swapped = True
            programs_db.rename(search_root / "parked.sqlite")
            programs_db.symlink_to(outside_db)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)

    with pytest.raises(PathValidationError):
        handler.handle_get_program_details("programs.sqlite", "program-1")


@requires_descriptor_traversal
def test_stable_database_view_reads_uncheckpointed_wal(tmp_path):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("CREATE TABLE programs (id INTEGER)")
        writer.commit()
        writer.execute("INSERT INTO programs VALUES (1)")
        writer.commit()
        assert (tmp_path / "programs.sqlite-wal").exists()

        with _handler(tmp_path)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1
    finally:
        writer.close()


@requires_descriptor_traversal
def test_stable_database_connection_is_read_only(tmp_path):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path)
    contents_before = database_path.read_bytes()
    mtime_before = database_path.stat().st_mtime_ns

    with _handler(tmp_path)._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as connection, pytest.raises(sqlite3.OperationalError, match="readonly"):
        connection.execute("INSERT INTO programs DEFAULT VALUES")

    assert database_path.read_bytes() == contents_before
    assert database_path.stat().st_mtime_ns == mtime_before


@requires_descriptor_traversal
def test_stable_database_view_retries_sidecar_rotation(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path)
    handler = _handler(tmp_path)
    copy_sidecar = handler._copy_optional_database_sidecar
    attempts = 0

    def race_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DatabaseViewRaceError("sidecar rotated")
        return copy_sidecar(*args, **kwargs)

    monkeypatch.setattr(handler, "_copy_optional_database_sidecar", race_once)

    with handler._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as connection:
        connection.execute("SELECT COUNT(*) FROM programs").fetchone()

    assert attempts > 1


@requires_descriptor_traversal
def test_stable_database_view_supports_nondefault_filesystem():
    shared_memory = "/dev/shm"
    if not os.path.isdir(shared_memory) or not os.access(shared_memory, os.W_OK):
        pytest.skip("writable secondary filesystem unavailable")
    if os.stat(shared_memory).st_dev == os.stat(tempfile.gettempdir()).st_dev:
        pytest.skip("secondary filesystem shares the default temp device")

    with tempfile.TemporaryDirectory(dir=shared_memory) as search_root:
        database_path = os.path.join(search_root, "programs.sqlite")
        _create_stats_database(database_path)

        with _handler(search_root)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as connection:
            assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 0


@requires_descriptor_traversal
def test_stable_database_view_supports_read_only_parent(tmp_path):
    search_root = tmp_path / "results"
    search_root.mkdir()
    database_path = search_root / "programs.sqlite"
    _create_stats_database(database_path)
    search_root.chmod(0o555)
    try:
        with _handler(search_root)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as connection:
            assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 0
        assert list(search_root.iterdir()) == [database_path]
    finally:
        search_root.chmod(0o755)


@requires_descriptor_traversal
def test_database_snapshot_supports_unavailable_hardlinks(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path, program_count=1)
    contents_before = database_path.read_bytes()

    def reject_hardlink(*args, **kwargs):
        raise OSError(errno.EPERM, "hardlinks disabled")

    monkeypatch.setattr(os, "link", reject_hardlink)

    with _handler(tmp_path)._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1

    assert database_path.read_bytes() == contents_before


@requires_descriptor_traversal
def test_database_snapshot_reads_wal_without_hardlinks(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("CREATE TABLE programs (id INTEGER)")
        writer.execute("INSERT INTO programs VALUES (1)")
        writer.commit()

        def reject_hardlink(*args, **kwargs):
            raise OSError(errno.EPERM, "hardlinks disabled")

        monkeypatch.setattr(os, "link", reject_hardlink)

        with _handler(tmp_path)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1
    finally:
        writer.close()


@requires_descriptor_traversal
def test_database_snapshot_reports_active_rollback_journal_as_busy(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode = DELETE")
        writer.execute("PRAGMA cache_size = 5")
        writer.execute("CREATE TABLE programs (value BLOB)")
        writer.executemany(
            "INSERT INTO programs VALUES (?)",
            [(b"a" * 3000,) for _ in range(100)],
        )
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE programs SET value = ?", (b"b" * 3000,))
        assert (tmp_path / "programs.sqlite-journal").exists()

        def reject_hardlink(*args, **kwargs):
            raise OSError(errno.EPERM, "hardlinks disabled")

        monkeypatch.setattr(os, "link", reject_hardlink)

        with (
            pytest.raises(sqlite3.OperationalError, match="busy"),
            _handler(tmp_path)._connect_database_within_root(
                database_path,
                timeout=0.1,
            ),
        ):
            pass
    finally:
        writer.rollback()
        writer.close()


@requires_descriptor_traversal
def test_database_snapshot_accepts_idle_persist_journal(tmp_path):
    database_path = tmp_path / "programs.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA journal_mode = PERSIST")
        connection.execute("CREATE TABLE programs (id INTEGER)")
        connection.execute("INSERT INTO programs VALUES (1)")
        connection.commit()
    assert (tmp_path / "programs.sqlite-journal").exists()

    with _handler(tmp_path)._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as reader:
        assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1


@requires_descriptor_traversal
def test_database_main_snapshot_is_reused_between_polls(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path, program_count=1)
    handler = _handler(tmp_path)
    copy_database_descriptor = handler._copy_database_descriptor
    main_copies = 0

    def count_main_copies(*args, **kwargs):
        nonlocal main_copies
        if args[2] == "database.sqlite":
            main_copies += 1
        return copy_database_descriptor(*args, **kwargs)

    monkeypatch.setattr(handler, "_copy_database_descriptor", count_main_copies)

    for _ in range(2):
        with handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as connection:
            assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1

    assert main_copies == 1


@requires_descriptor_traversal
def test_database_snapshot_closes_connection_when_initial_read_fails(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path)
    real_connect = sqlite3.connect
    staged_connection = None

    class FailingConnection:
        closed = False

        def execute(self, statement):
            if statement == "BEGIN":
                return
            raise sqlite3.DatabaseError("initial read failed")

        def close(self):
            self.closed = True

    def fail_staged_connect(database, *args, **kwargs):
        nonlocal staged_connection
        if "shinka-webui-db-" not in os.fspath(database):
            return real_connect(database, *args, **kwargs)
        staged_connection = FailingConnection()
        return staged_connection

    monkeypatch.setattr(sqlite3, "connect", fail_staged_connect)

    with (
        pytest.raises(sqlite3.DatabaseError, match="initial read failed"),
        _handler(tmp_path)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ),
    ):
        pass

    assert staged_connection is not None
    assert staged_connection.closed


@requires_descriptor_traversal
def test_database_view_retries_rotation_before_sqlite_connect(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE programs (id INTEGER)")
    writer.execute("INSERT INTO programs VALUES (1)")
    writer.commit()
    original_connect = sqlite3.connect
    replacement_writers = []
    staged_connections = 0

    def rotate_before_first_staged_connect(database, *args, **kwargs):
        nonlocal staged_connections, writer
        if "shinka-webui-db-" in os.fspath(database):
            staged_connections += 1
            if staged_connections == 1:
                writer.close()
                replacement = original_connect(database_path)
                replacement.execute("PRAGMA journal_mode = WAL")
                replacement.execute("PRAGMA wal_autocheckpoint = 0")
                replacement.execute("INSERT INTO programs VALUES (2)")
                replacement.commit()
                replacement_writers.append(replacement)
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", rotate_before_first_staged_connect)
    try:
        with _handler(tmp_path)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 2
        assert staged_connections == 2
    finally:
        for replacement in replacement_writers:
            replacement.close()
        if not replacement_writers:
            writer.close()


@requires_descriptor_traversal
def test_database_view_rejects_main_and_sidecar_pairing_swap(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    with sqlite3.connect(database_path) as writer:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("CREATE TABLE programs (value INTEGER)")
        writer.execute("INSERT INTO programs VALUES (111)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    replacement_path = tmp_path / "replacement.sqlite"
    shutil.copyfile(database_path, replacement_path)
    replacement_writer = sqlite3.connect(replacement_path)
    replacement_writer.execute("PRAGMA journal_mode = WAL")
    replacement_writer.execute("PRAGMA wal_autocheckpoint = 0")
    replacement_writer.execute("UPDATE programs SET value = 999")
    replacement_writer.commit()

    handler = _handler(tmp_path)
    stage_cached_database_main = handler._stage_cached_database_main
    swapped = False

    def stage_then_swap(*args, **kwargs):
        nonlocal swapped
        stage_cached_database_main(*args, **kwargs)
        if not swapped:
            swapped = True
            os.replace(replacement_path, database_path)
            for suffix in ("-wal", "-shm"):
                os.replace(
                    os.fspath(replacement_path) + suffix,
                    os.fspath(database_path) + suffix,
                )

    monkeypatch.setattr(handler, "_stage_cached_database_main", stage_then_swap)
    try:
        with (
            pytest.raises((PathValidationError, sqlite3.OperationalError)),
            handler._connect_database_within_root(
                database_path,
                timeout=5.0,
            ),
        ):
            pass
    finally:
        replacement_writer.close()
