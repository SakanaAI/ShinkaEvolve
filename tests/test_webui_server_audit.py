"""Regression tests for bypasses found when auditing the hardened web UI server.

Each one got past the first round of fixes: the static allowlist, the
confinement of files read beside a database, and error handling for requests
and rows the server did not expect.
"""

import http.client
import json
import sqlite3
import threading
import urllib.parse
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import pytest

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.dbase import Program
from shinka.webui.visualization import make_server
from webui_test_utils import write_prompts_db

Response = Tuple[int, Dict[str, str], bytes]


def _write_program_db(db_path: Path) -> None:
    db = ProgramDatabase(DatabaseConfig(db_path=str(db_path)))
    db.add(Program(id="p0", code="x = 1\n", generation=0, correct=True))
    db.close()


def _log_failed_proposal(db_path: Path, generation: int, details: dict) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) "
        "VALUES (?, 'proposal', 'failed', ?, 123.0)",
        (generation, json.dumps({"node_kind": "failed_proposal", **details})),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """`base/results` is the search root; `base` holds files it must not reach."""
    base = tmp_path.resolve()
    run = base / "results" / "run_a"
    run.mkdir(parents=True)
    _write_program_db(run / "programs.sqlite")
    (base / "outside.py").write_text("OUTSIDE = 1\n", encoding="utf-8")
    (base / "main.go").write_text("package outside\n", encoding="utf-8")
    write_prompts_db(base / "prompts.sqlite", "PARENT_PROMPT_SECRET")
    return base


@pytest.fixture
def server(tree: Path) -> Iterator[int]:
    httpd = make_server(0, str(tree / "results"))
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


def _request(port: int, target: str, method: str = "GET") -> Response:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, target)
        conn.endheaders()
        response = conn.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, headers, response.read()
    finally:
        conn.close()


def _raw_request(port: int, request: bytes) -> bytes:
    """Send bytes as they are, for requests http.client refuses to build."""
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def _details_target(db_path: str, generation: int) -> str:
    return "/get_program_details?" + urllib.parse.urlencode(
        {"db_path": db_path, "id": f"failed:proposal:{generation}"}
    )


def _failed_node(port: int, generation: int) -> Optional[dict]:
    status, _, body = _request(
        port, _details_target("run_a/programs.sqlite", generation)
    )
    assert status == 200, body[:300]
    return json.loads(body)


# --- static allowlist ----------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    "target",
    [
        "/index.html;x%2f..%2fsecurity.py",
        "/index.html;x%2f..%2fvisualization.py",
        "/index.html;x%2f..%2f__pycache__%2f",
        "/favicon.png;%2f..%2fsecurity.py",
    ],
)
def test_path_params_cannot_reach_other_package_files(
    server: int, target: str, method: str
):
    # urlparse drops `;params` from the path that is checked against the
    # allowlist, while the base handler used to resolve the raw target.
    status, headers, body = _request(server, target, method=method)

    assert (status, headers.get("content-type", "")) in {
        (200, "text/html"),
        (200, "image/png"),
    }, (status, headers)
    assert b"resolve_under_root" not in body
    assert b"Directory listing" not in body


# --- files read beside the database ---------------------------------------------


@pytest.mark.parametrize("db_path", [".", "./", "run_a/..", "run_a"])
def test_directory_db_path_is_refused_without_reading_or_stalling(
    server: int, db_path: str
):
    quoted = urllib.parse.quote(db_path, safe="")
    for endpoint in (
        "/get_system_prompts?db_path=",
        "/get_database_stats?db_path=",
        "/get_programs?db_path=",
        "/get_meta_files?db_path=",
    ):
        status, _, body = _request(server, endpoint + quoted)

        assert status == 404, (endpoint, status)
        assert b"PARENT_PROMPT_SECRET" not in body


def test_absolute_root_as_db_path_is_refused(server: int, tree: Path):
    quoted = urllib.parse.quote(str(tree / "results"), safe="")

    status, _, body = _request(server, "/get_system_prompts?db_path=" + quoted)

    assert status == 404
    assert b"PARENT_PROMPT_SECRET" not in body


def test_prompts_db_symlink_out_of_root_is_not_read(server: int, tree: Path):
    (tree / "results" / "run_a" / "prompts.sqlite").symlink_to(tree / "prompts.sqlite")
    quoted = urllib.parse.quote("run_a/programs.sqlite", safe="")

    status, _, body = _request(server, "/get_system_prompts?db_path=" + quoted)
    stats_status, _, stats_body = _request(
        server, "/get_database_stats?db_path=" + quoted
    )

    assert (status, json.loads(body)) == (200, [])
    assert stats_status == 200
    assert json.loads(stats_body)["prompt_count"] == 0


def test_failed_node_fallback_does_not_follow_main_symlink_out_of_root(
    server: int, tree: Path
):
    gen_dir = tree / "results" / "run_a" / "gen_3"
    gen_dir.mkdir()
    (gen_dir / "failure.json").write_text('{"failure_reason": "x"}', encoding="utf-8")
    (gen_dir / "main.py").symlink_to(tree / "outside.py")
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=3,
        details={"failure_json_path": "run_a/gen_3/failure.json"},
    )

    node = _failed_node(server, 3)

    assert node is not None
    assert node["code"] is None


def test_failed_node_fallback_still_reads_main_file_inside_root(
    server: int, tree: Path
):
    gen_dir = tree / "results" / "run_a" / "gen_4"
    gen_dir.mkdir()
    (gen_dir / "failure.json").write_text('{"failure_reason": "x"}', encoding="utf-8")
    (gen_dir / "main.py").write_text("INSIDE = 1\n", encoding="utf-8")
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=4,
        details={"failure_json_path": "run_a/gen_4/failure.json"},
    )

    node = _failed_node(server, 4)

    assert node is not None
    assert node["code"] == "INSIDE = 1\n"


def test_failure_path_naming_the_root_does_not_glob_its_parent(server: int, tree: Path):
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=5,
        details={"failure_json_path": "."},
    )

    node = _failed_node(server, 5)

    # `base/main.go` sits beside the root; its suffix must not leak.
    assert node is not None
    assert node["language"] == "python"


# --- rows and requests the server did not expect --------------------------------


@pytest.mark.parametrize("bad_path", [5, ["a"], {"k": "v"}, True])
def test_non_string_failure_path_does_not_break_the_database(
    server: int, tree: Path, bad_path: object
):
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=6,
        details={"failure_json_path": bad_path},
    )
    quoted = urllib.parse.quote("run_a/programs.sqlite", safe="")

    for endpoint in ("/get_programs", "/get_programs_summary", "/get_program_count"):
        assert _request(server, f"{endpoint}?db_path={quoted}")[0] == 200


def test_malformed_failure_json_does_not_break_the_database(server: int, tree: Path):
    gen_dir = tree / "results" / "run_a" / "gen_7"
    gen_dir.mkdir()
    (gen_dir / "failure.json").write_text('["not", "a", "dict"]', encoding="utf-8")
    (gen_dir / "odd.json").write_text('{"artifacts": "not a dict"}', encoding="utf-8")
    db = tree / "results" / "run_a" / "programs.sqlite"
    _log_failed_proposal(db, 7, {"failure_json_path": "run_a/gen_7/failure.json"})
    _log_failed_proposal(db, 8, {"failure_json_path": "run_a/gen_7/odd.json"})

    assert _failed_node(server, 7) is not None
    assert _failed_node(server, 8) is not None


@pytest.mark.parametrize(
    "target",
    [
        "/get_programs?db_path=%E2%82%AC.sqlite",
        "/get_program_details?db_path=%E2%82%AC.sqlite&id=x",
        "/plot_file/%E2%82%AC.png",
        "/plot_file/%ff.png",
    ],
)
def test_non_latin1_names_in_error_messages_get_a_response(server: int, target: str):
    # The status line is Latin-1; echoing the name used to raise and drop
    # the connection.
    assert _request(server, target)[0] == 404


@pytest.mark.parametrize(
    "target",
    [
        "/get_programs?db_path=a%1b%5d0%3bpwned%07.sqlite",
        "/get_programs?db_path=a%0d%0aFAKE-LOG-LINE",
        "/get_program_details?db_path=run_a%2Fprograms.sqlite&id=a%00b",
        "/plot_file/a%0ab.png",
    ],
)
def test_control_characters_in_request_are_refused(server: int, target: str):
    # They would otherwise reach the server's log (terminal escapes) and
    # error messages.
    assert _request(server, target)[0] == 400


@pytest.mark.parametrize(
    "request_line",
    [
        b"GET /list_databases HTTP/9.9",
        b"GET /plot_file/a b.png HTTP/1.1",
        b"GET http://[::1/list_databases HTTP/1.1",
    ],
)
def test_malformed_request_line_gets_an_error_response(
    server: int, request_line: bytes
):
    response = _raw_request(server, request_line + b"\r\nHost: localhost\r\n\r\n")

    # A request line with an unusable version is answered HTTP/0.9 style,
    # body only, so look for the error page rather than a status line.
    assert b"Error code: 400" in response or b"Error code: 505" in response
    # The server keeps serving afterwards.
    assert _request(server, "/list_databases")[0] == 200
