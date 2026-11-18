"""Shared helpers for tests that drive the visualization server over HTTP."""

import contextlib
import http.client
import json
import sqlite3
import threading
import urllib.parse
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.dbase import Program
from shinka.webui.visualization import make_server

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
)

Response = Tuple[int, Dict[str, str], bytes]


def write_program_db(db_path: Path, program_id: str = "p0") -> None:
    db = ProgramDatabase(DatabaseConfig(db_path=str(db_path)))
    db.add(
        Program(
            id=program_id,
            code="x = 1\n",
            generation=0,
            correct=True,
            combined_score=1.0,
        )
    )
    db.close()


def log_failed_proposal(db_path: Path, generation: int, details: dict) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) "
        "VALUES (?, 'proposal', 'failed', ?, 123.0)",
        (generation, json.dumps({"node_kind": "failed_proposal", **details})),
    )
    conn.commit()
    conn.close()


@contextlib.contextmanager
def serve(
    root: Path, host: str = "127.0.0.1", allowed_hosts: Sequence[str] = ()
) -> Iterator[int]:
    """Serve ``root`` on an ephemeral port and yield the port."""
    httpd = make_server(0, str(root), host, allowed_hosts)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def request(
    port: int,
    target: str,
    method: str = "GET",
    host_header: Optional[str] = None,
) -> Response:
    """Send a request with the target exactly as given (no normalisation)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, target, skip_host=host_header is not None)
        if host_header is not None:
            conn.putheader("Host", host_header)
        conn.endheaders()
        response = conn.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, headers, response.read()
    finally:
        conn.close()


def quote(value: object) -> str:
    return urllib.parse.quote(str(value), safe="")
