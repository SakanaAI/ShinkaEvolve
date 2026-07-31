import contextlib
import errno
import os
import sqlite3
import stat

import pytest

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.webui import visualization
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
def test_program_response_cache_does_not_cross_search_roots(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    database_path = first_root / "programs.sqlite"
    database = ProgramDatabase(DatabaseConfig(db_path=os.fspath(database_path)))
    database.close()
    first_handler = _handler(first_root)
    first_response = None

    def capture_first_response(data):
        nonlocal first_response
        first_response = data

    first_handler.send_json_response = capture_first_response
    first_handler.send_error = lambda *args: pytest.fail(f"unexpected error: {args}")
    second_handler = _handler(second_root)
    second_error = None

    def capture_second_error(*args):
        nonlocal second_error
        second_error = args

    second_handler.send_json_response = lambda data: pytest.fail(
        f"unexpected cached response: {data}"
    )
    second_handler.send_error = capture_second_error

    visualization.db_cache.clear()
    try:
        first_handler.handle_get_programs("programs.sqlite")
        second_handler.handle_get_programs("programs.sqlite")
    finally:
        visualization.db_cache.clear()

    assert first_response == []
    assert second_error is not None
    assert second_error[0] == 404


@requires_descriptor_traversal
def test_database_stats_reads_path_specific_wal_for_hardlinked_prompts(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    prompts_path = tmp_path / "prompts.sqlite"
    with sqlite3.connect(database_path) as connection:
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
        connection.execute("CREATE TABLE system_prompts (metadata TEXT)")
    os.link(database_path, prompts_path)
    prompt_writer = sqlite3.connect(prompts_path)
    prompt_writer.execute("PRAGMA journal_mode = WAL")
    prompt_writer.execute("PRAGMA wal_autocheckpoint = 0")
    prompt_writer.execute("INSERT INTO system_prompts VALUES ('{}')")
    prompt_writer.commit()

    handler = _handler(tmp_path)
    connect_database = handler._connect_database_within_root
    connected_paths = []
    response = None

    @contextlib.contextmanager
    def count_connections(path, **kwargs):
        connected_paths.append(os.fspath(path))
        with connect_database(path, **kwargs) as connection:
            yield connection

    def capture_response(data):
        nonlocal response
        response = data

    monkeypatch.setattr(handler, "_connect_database_within_root", count_connections)
    handler.send_json_response = capture_response

    try:
        handler.handle_get_database_stats("programs.sqlite")

        assert response is not None
        assert response["prompt_count"] == 1
        assert connected_paths == [
            os.fspath(database_path),
            os.fspath(prompts_path),
        ]
    finally:
        prompt_writer.close()


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


@requires_descriptor_traversal
def test_database_snapshot_rejects_oversized_source_before_copy(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path)
    handler = _handler(tmp_path)
    copied = False

    def record_copy(*args, **kwargs):
        nonlocal copied
        copied = True

    monkeypatch.setattr(handler, "_database_main_cache_max_bytes", 1)
    monkeypatch.setattr(handler, "_copy_database_descriptor", record_copy)

    with (
        pytest.raises(sqlite3.OperationalError, match="snapshot limit"),
        handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ),
    ):
        pass

    assert not copied


@requires_descriptor_traversal
def test_database_snapshot_rejects_oversized_wal_before_copy(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    _create_stats_database(database_path)
    wal_path = tmp_path / "programs.sqlite-wal"
    wal_path.write_bytes(b"xx")
    handler = _handler(tmp_path)
    copy_database_descriptor = handler._copy_database_descriptor
    wal_copied = False

    def record_wal_copy(*args, **kwargs):
        nonlocal wal_copied
        if args[2].endswith("-wal"):
            wal_copied = True
        return copy_database_descriptor(*args, **kwargs)

    monkeypatch.setattr(
        handler,
        "_database_main_cache_max_bytes",
        database_path.stat().st_size + 1,
    )
    monkeypatch.setattr(handler, "_copy_database_descriptor", record_wal_copy)

    with (
        pytest.raises(sqlite3.OperationalError, match="WAL exceed"),
        handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ),
    ):
        pass

    assert not wal_copied


@requires_descriptor_traversal
def test_database_snapshot_retries_staging_parent_after_enospc(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    first_staging_parent = tmp_path / "first-staging"
    second_staging_parent = tmp_path / "second-staging"
    first_staging_parent.mkdir()
    second_staging_parent.mkdir()
    _create_stats_database(database_path)
    handler = _handler(tmp_path)
    copy_database_descriptor = handler._copy_database_descriptor
    copy_attempts = 0

    def fail_first_copy(*args, **kwargs):
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            raise OSError(errno.ENOSPC, "staging filesystem full")
        return copy_database_descriptor(*args, **kwargs)

    monkeypatch.setattr(
        handler,
        "_database_staging_parents",
        lambda path: [
            os.fspath(first_staging_parent),
            os.fspath(second_staging_parent),
        ],
    )
    monkeypatch.setattr(handler, "_copy_database_descriptor", fail_first_copy)

    with handler._connect_database_within_root(
        database_path,
        timeout=5.0,
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 0

    assert copy_attempts == 2
    assert not list(first_staging_parent.iterdir())


@requires_descriptor_traversal
def test_database_snapshot_retries_wal_staging_after_enospc(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "programs.sqlite"
    first_staging_parent = tmp_path / "first-staging"
    second_staging_parent = tmp_path / "second-staging"
    first_staging_parent.mkdir()
    second_staging_parent.mkdir()
    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE programs (id INTEGER)")
    writer.execute("INSERT INTO programs VALUES (1)")
    writer.commit()
    handler = _handler(tmp_path)
    copy_sidecar = handler._copy_optional_database_sidecar
    wal_copy_attempts = 0

    def fail_first_wal_copy(*args, **kwargs):
        nonlocal wal_copy_attempts
        wal_copy_attempts += 1
        if wal_copy_attempts == 1:
            raise OSError(errno.ENOSPC, "staging filesystem full")
        return copy_sidecar(*args, **kwargs)

    monkeypatch.setattr(
        handler,
        "_database_staging_parents",
        lambda path: [
            os.fspath(first_staging_parent),
            os.fspath(second_staging_parent),
        ],
    )
    monkeypatch.setattr(
        handler,
        "_copy_optional_database_sidecar",
        fail_first_wal_copy,
    )
    try:
        with handler._connect_database_within_root(
            database_path,
            timeout=5.0,
        ) as connection:
            assert connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1
    finally:
        writer.close()

    assert wal_copy_attempts == 2
    assert not [
        path
        for path in first_staging_parent.iterdir()
        if path.name.startswith("shinka-webui-db-")
    ]
    handler.clear_database_snapshot_cache()


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
