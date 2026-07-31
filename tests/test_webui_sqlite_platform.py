import os
import sqlite3
import stat

import pytest

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.webui.visualization import DatabaseRequestHandler


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
def test_database_view_holds_snapshot_after_final_verification(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE programs (id INTEGER)")
    writer.execute("INSERT INTO programs VALUES (1)")
    writer.commit()

    handler = _handler(tmp_path)
    verify_database_view = handler._verify_database_view
    replacement_writers = []
    verifications = 0

    def verify_then_rotate(*args, **kwargs):
        nonlocal verifications, writer
        verify_database_view(*args, **kwargs)
        verifications += 1
        if verifications == 2:
            writer.close()
            replacement = sqlite3.connect(database_path)
            replacement.execute("PRAGMA journal_mode = WAL")
            replacement.execute("PRAGMA wal_autocheckpoint = 0")
            replacement.execute("INSERT INTO programs VALUES (2)")
            replacement.commit()
            replacement_writers.append(replacement)

    monkeypatch.setattr(handler, "_verify_database_view", verify_then_rotate)
    try:
        with handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1
    finally:
        for replacement in replacement_writers:
            replacement.close()
        if not replacement_writers:
            writer.close()


def test_database_view_reads_without_descriptor_traversal(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path, program_count=1)
    handler = _handler(tmp_path)
    monkeypatch.setattr(handler, "_supports_descriptor_traversal", lambda: False)

    with handler._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1


def test_fallback_program_count_does_not_change_journal_mode(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    database = ProgramDatabase(DatabaseConfig(db_path=os.fspath(database_path)))
    database.close()
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA journal_mode = DELETE").fetchone()
    contents_before = database_path.read_bytes()
    handler = _handler(tmp_path)
    response = None
    monkeypatch.setattr(handler, "_supports_descriptor_traversal", lambda: False)

    def capture_response(data):
        nonlocal response
        response = data

    handler.send_json_response = capture_response
    handler.send_error = lambda *args: pytest.fail(f"unexpected error: {args}")

    handler.handle_get_program_count("programs.sqlite")

    with sqlite3.connect(database_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert response is not None
    assert database_path.read_bytes() == contents_before
    assert journal_mode == "delete"


@requires_descriptor_traversal
def test_database_main_snapshot_replaces_checkpointed_version(tmp_path, monkeypatch):
    database_path = tmp_path / "programs.sqlite"
    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE programs (id INTEGER)")
    writer.execute("INSERT INTO programs VALUES (1)")
    writer.commit()
    handler = _handler(tmp_path)
    copy_database_descriptor = handler._copy_database_descriptor
    main_copies = 0
    old_snapshot_directory = None

    def count_main_copies(*args, **kwargs):
        nonlocal main_copies
        if args[2] == "database.sqlite":
            main_copies += 1
            if main_copies == 2:
                assert old_snapshot_directory is not None
                assert not os.path.exists(old_snapshot_directory)
        return copy_database_descriptor(*args, **kwargs)

    monkeypatch.setattr(handler, "_copy_database_descriptor", count_main_copies)
    try:
        with handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1

        initial_stat = database_path.stat()
        source_key = (initial_stat.st_dev, initial_stat.st_ino)
        old_snapshot_directory = handler._database_main_cache[
            source_key
        ].directory.name

        writer.execute("INSERT INTO programs VALUES (2)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        with handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as reader:
            assert reader.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 2
    finally:
        writer.close()

    database_stat = database_path.stat()
    source_key = (database_stat.st_dev, database_stat.st_ino)
    assert main_copies == 2
    assert source_key in handler._database_main_cache
    assert handler._database_main_cache[
        source_key
    ].version == handler._database_cache_key(database_stat)


@pytest.mark.parametrize(
    ("path", "expected_target"),
    [
        (
            r"C:\Users\Rob Lange\programs.sqlite",
            ("file:///C:/Users/Rob%20Lange/programs.sqlite?mode=ro", True),
        ),
        (
            r"\\server\results\programs.sqlite",
            (r"\\server\results\programs.sqlite", False),
        ),
        (
            r"\\?\C:\results\programs.sqlite",
            (r"\\?\C:\results\programs.sqlite", False),
        ),
    ],
)
def test_sqlite_read_only_target_supports_windows_paths(path, expected_target):
    assert DatabaseRequestHandler._sqlite_read_only_target(path) == expected_target


@requires_descriptor_traversal
def test_staging_parent_rejects_foreign_owned_ancestor(tmp_path, monkeypatch):
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    real_stat = os.stat
    foreign_uid = os.geteuid() + 1

    def stat_with_foreign_owner(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.path.realpath(path) != os.path.realpath(tmp_path):
            return result
        values = list(result)
        values[stat.ST_UID] = foreign_uid
        return os.stat_result(values)

    monkeypatch.setattr(os, "stat", stat_with_foreign_owner)

    assert not DatabaseRequestHandler._is_secure_staging_parent(
        os.fspath(staging_parent)
    )
