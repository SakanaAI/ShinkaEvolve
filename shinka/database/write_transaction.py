from collections.abc import Iterator
from contextlib import contextmanager
from sqlite3 import Connection


@contextmanager
def write_transaction(conn: Connection) -> Iterator[None]:
    """Join the caller's write transaction or own it at the outermost level."""
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        yield
