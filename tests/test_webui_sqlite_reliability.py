import errno
import os
import sqlite3

import pytest

from shinka.webui.visualization import DatabaseRequestHandler, DatabaseViewRaceError

requires_descriptor_traversal = pytest.mark.skipif(
    not DatabaseRequestHandler._supports_descriptor_traversal(),
    reason="database race hardening requires descriptor traversal",
)


def _handler(root):
    handler = object.__new__(DatabaseRequestHandler)
    handler.search_root = os.fspath(root)
    handler._canonical_search_root = os.path.realpath(root)
    return handler


def _create_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE programs (id INTEGER)")


@requires_descriptor_traversal
def test_database_copy_never_exceeds_captured_source_size(tmp_path, monkeypatch):
    source_path = tmp_path / "source.sqlite"
    destination_path = tmp_path / "snapshot.sqlite"
    source_path.write_bytes(b"abc")
    source_descriptor = os.open(source_path, os.O_RDONLY)
    destination_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    expected_stat = os.fstat(source_descriptor)
    real_pread = os.pread
    appended = False

    def append_before_read(descriptor, size, offset):
        nonlocal appended
        if descriptor == source_descriptor and not appended:
            appended = True
            with source_path.open("ab") as source:
                source.write(b"growth")
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(os, "pread", append_before_read)
    try:
        with pytest.raises(DatabaseViewRaceError, match="changed"):
            DatabaseRequestHandler._copy_database_descriptor(
                source_descriptor,
                destination_descriptor,
                destination_path.name,
                expected_stat=expected_stat,
            )
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)

    assert destination_path.read_bytes() == b"abc"


@requires_descriptor_traversal
def test_database_context_does_not_retry_caller_enospc(tmp_path):
    database_path = tmp_path / "programs.sqlite"
    _create_database(database_path)

    with (
        pytest.raises(OSError, match="caller filesystem full"),
        _handler(tmp_path)._connect_database_within_root(
            database_path,
            timeout=5.0,
        ),
    ):
        raise OSError(errno.ENOSPC, "caller filesystem full")
