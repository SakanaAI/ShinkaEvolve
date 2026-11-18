"""Regression tests for the visualization server's exposure and path handling.

Each test drives a real server bound to an ephemeral loopback port.
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
from shinka.webui import security
from shinka.webui.visualization import (
    DatabaseRequestHandler,
    make_server,
    resolve_cli_paths,
)

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
)

Response = Tuple[int, Dict[str, str], bytes]


def _write_program_db(db_path: Path, program_id: str) -> None:
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
    """A search root (`base/results`) with secrets beside and above it."""
    base = tmp_path.resolve()
    root = base / "results"
    run = root / "run_a"
    plots = run / "gen_1" / "results" / "plots"
    plots.mkdir(parents=True)
    (plots / "p.png").write_bytes(PNG_BYTES)
    (run / "meta").mkdir()
    (run / "meta" / "meta_1.txt").write_text("# inside", encoding="utf-8")
    (root / ".env").write_text("API_KEY=secret", encoding="utf-8")
    _write_program_db(run / "programs.sqlite", "p0")

    # A sibling whose name extends the root's name: a string-prefix check
    # treats it as inside the root.
    private = base / "results_private"
    private.mkdir()
    (private / "secret.txt").write_text("sibling secret", encoding="utf-8")
    (private / "secret.png").write_bytes(PNG_BYTES)
    (private / "meta_1.txt").write_text("sibling meta", encoding="utf-8")
    (private / "meta_abc.txt").write_text("sibling non-int meta", encoding="utf-8")
    (base / "outside.py").write_text("OUTSIDE = 1\n", encoding="utf-8")
    (root / "link_out").symlink_to(private)
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


def _request(
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


def _quote(value: object) -> str:
    return urllib.parse.quote(str(value), safe="")


# --- finding 1: exposure -----------------------------------------------------


def test_server_binds_loopback_by_default(tree: Path):
    httpd = make_server(0, str(tree / "results"))
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


def test_plot_file_refuses_non_image_files(server: int):
    status, _, body = _request(server, "/plot_file/.env")

    assert status == 403
    assert b"secret" not in body


def test_plot_file_serves_images_inside_root(server: int):
    status, headers, body = _request(
        server, "/plot_file/" + _quote("run_a/gen_1/results/plots/p.png")
    )

    assert status == 200
    assert headers["content-type"] == "image/png"
    assert body == PNG_BYTES


def test_static_serving_is_limited_to_ui_assets(server: int):
    assert _request(server, "/")[0] == 200
    assert _request(server, "/viz_tree.html?db_path=x")[0] == 200
    assert _request(server, "/favicon.png", method="HEAD")[0] == 200
    assert _request(server, "/visualization.py")[0] == 404
    assert _request(server, "/visualization.py", method="HEAD")[0] == 404
    assert _request(server, "/__pycache__/")[0] == 404


# --- finding 2: containment --------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "/plot_file/%2e%2e/results_private/secret.png",
        "/plot_file/../results_private/secret.png",
        "/plot_file/link_out/secret.png",
    ],
)
def test_plot_file_refuses_paths_outside_root(server: int, target: str):
    status, _, body = _request(server, target)

    assert status == 403
    assert body != PNG_BYTES


def test_plot_file_refuses_absolute_path(server: int, tree: Path):
    target = "/plot_file/" + _quote(tree / "results_private" / "secret.png")

    assert _request(server, target)[0] == 403


def test_resolve_under_root_rejects_prefix_sibling_and_symlink(tree: Path):
    root = str(tree / "results")

    assert security.resolve_under_root(root, "run_a/programs.sqlite") == str(
        tree / "results" / "run_a" / "programs.sqlite"
    )
    assert security.resolve_under_root(root, str(tree / "results" / ".env"))
    assert security.resolve_under_root(root, "../results_private/secret.txt") is None
    assert security.resolve_under_root(root, "link_out/secret.txt") is None
    assert security.resolve_under_root(root, str(tree / "outside.py")) is None
    assert security.resolve_under_root(root, "") is None
    assert security.resolve_under_root(root, "a\x00b") is None


# --- finding 3: db_path confinement -----------------------------------------

DB_PATH_ENDPOINTS = [
    "/get_programs?db_path={db}",
    "/get_programs_summary?db_path={db}",
    "/get_program_count?db_path={db}",
    "/get_program_details?db_path={db}&id=p0",
    "/get_meta_files?db_path={db}",
    "/get_meta_content?db_path={db}&processed_count=1",
    "/download_meta_pdf?db_path={db}&processed_count=1",
    "/get_plots?db_path={db}&generation=1&program_id=p0",
    "/get_system_prompts?db_path={db}",
    "/get_database_stats?db_path={db}",
]


@pytest.mark.parametrize("endpoint", DB_PATH_ENDPOINTS)
@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_db_path_outside_root_is_refused(
    server: int, tree: Path, endpoint: str, escape: str
):
    db_path = {
        "relative": "../results_private/programs.sqlite",
        "absolute": str(tree / "results_private" / "programs.sqlite"),
        "symlink": "link_out/programs.sqlite",
    }[escape]

    status, _, body = _request(server, endpoint.format(db=_quote(db_path)))

    assert status == 403
    assert b"sibling" not in body


@pytest.mark.parametrize("endpoint", DB_PATH_ENDPOINTS)
def test_db_path_inside_root_still_works(server: int, endpoint: str):
    status, _, _ = _request(server, endpoint.format(db=_quote("run_a/programs.sqlite")))

    assert status == 200


@pytest.mark.parametrize(
    "endpoint",
    [
        "/get_meta_content?db_path={db}&processed_count={count}",
        "/download_meta_pdf?db_path={db}&processed_count={count}",
    ],
)
@pytest.mark.parametrize("count", ["abc", "1.txt/../../../results_private/x", "-1"])
def test_meta_endpoints_require_integer_processed_count(
    server: int, endpoint: str, count: str
):
    target = endpoint.format(db=_quote("run_a/programs.sqlite"), count=_quote(count))

    status, _, body = _request(server, target)

    assert status == 400
    assert b"sibling" not in body


def test_get_plots_requires_integer_generation(server: int):
    target = "/get_plots?db_path={db}&generation={gen}&program_id=p0".format(
        db=_quote("run_a/programs.sqlite"), gen=_quote("1/../../..")
    )

    assert _request(server, target)[0] == 400


def test_failed_node_does_not_read_artifacts_outside_root(server: int, tree: Path):
    outside_failure = tree / "outside_failure.json"
    outside_failure.write_text(
        json.dumps(
            {
                "failure_reason": "from outside",
                "artifacts": {"generated_code_path": str(tree / "outside.py")},
            }
        ),
        encoding="utf-8",
    )
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=5,
        details={"failure_json_path": str(outside_failure)},
    )

    status, _, body = _request(
        server,
        "/get_program_details?db_path={db}&id={node}".format(
            db=_quote("run_a/programs.sqlite"), node=_quote("failed:proposal:5")
        ),
    )

    assert status == 200
    assert json.loads(body)["code"] is None
    assert b"OUTSIDE" not in body


def test_failed_node_reads_artifacts_inside_root(server: int, tree: Path):
    gen_dir = tree / "results" / "run_a" / "gen_6"
    gen_dir.mkdir()
    (gen_dir / "main.py").write_text("INSIDE = 1\n", encoding="utf-8")
    failure = gen_dir / "failure.json"
    failure.write_text(
        json.dumps({"artifacts": {"generated_code_path": str(gen_dir / "main.py")}}),
        encoding="utf-8",
    )
    _log_failed_proposal(
        tree / "results" / "run_a" / "programs.sqlite",
        generation=6,
        details={"failure_json_path": str(failure)},
    )

    _, _, body = _request(
        server,
        "/get_program_details?db_path={db}&id={node}".format(
            db=_quote("run_a/programs.sqlite"), node=_quote("failed:proposal:6")
        ),
    )

    assert json.loads(body)["code"] == "INSIDE = 1\n"


# --- finding 4: cross-origin reads and DNS rebinding -------------------------


def test_json_responses_do_not_allow_cross_origin_reads(server: int):
    status, headers, _ = _request(server, "/list_databases")

    assert status == 200
    assert "access-control-allow-origin" not in headers


@pytest.mark.parametrize(
    "target", ["/list_databases", "/", "/plot_file/run_a/gen_1/results/plots/p.png"]
)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_foreign_host_header_is_refused(server: int, target: str, method: str):
    status, _, _ = _request(
        server, target, method=method, host_header=f"evil.example:{server}"
    )

    assert status == 403


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]", "10.1.2.3"])
def test_local_host_headers_are_accepted(server: int, host: str):
    status, _, _ = _request(server, "/list_databases", host_header=f"{host}:{server}")

    assert status == 200


def test_allowed_hosts_cover_bind_name_and_extras():
    allowed = security.build_allowed_hosts("node17", ["Viz.Example.com:8888"])

    assert security.is_allowed_host("node17:8000", allowed)
    assert security.is_allowed_host("viz.example.com", allowed)
    assert security.is_allowed_host("LOCALHOST:8000", allowed)
    assert security.is_allowed_host(None, allowed)
    assert not security.is_allowed_host("viz.example.com.evil.example", allowed)
    assert not security.is_allowed_host("localhost.evil.example", allowed)
    assert not security.is_allowed_host("localhost:8000/x", allowed)
    assert not security.is_allowed_host("", allowed)


def test_error_message_cannot_inject_response_headers(server: int):
    target = "/get_programs?db_path=" + "nope%0d%0aX-Injected:%20yes%0d%0a%0d%0abody"

    status, headers, _ = _request(server, target)

    assert status == 404
    assert "x-injected" not in headers


# --- finding 6: DBs written before attempt_log existed -----------------------


@pytest.fixture
def legacy_db(tree: Path) -> str:
    run = tree / "results" / "run_legacy"
    run.mkdir()
    _write_program_db(run / "programs.sqlite", "legacy0")
    conn = sqlite3.connect(run / "programs.sqlite")
    conn.execute("DROP TABLE attempt_log")
    conn.commit()
    conn.close()
    return "run_legacy/programs.sqlite"


@pytest.mark.parametrize(
    "endpoint",
    [
        "/get_programs?db_path={db}",
        "/get_programs_summary?db_path={db}",
        "/get_program_count?db_path={db}",
    ],
)
def test_db_without_attempt_log_is_served(server: int, legacy_db: str, endpoint: str):
    status, _, body = _request(server, endpoint.format(db=_quote(legacy_db)))

    assert status == 200
    assert b"legacy0" in body or json.loads(body)["count"] >= 1


def test_db_without_attempt_log_has_no_failed_nodes(
    server: int, legacy_db: str, tree: Path
):
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    handler.search_root = str(tree / "results")

    assert handler._load_failed_proposal_nodes(str(tree / "results" / legacy_db)) == []

    status, _, _ = _request(
        server,
        "/get_program_details?db_path={db}&id={node}".format(
            db=_quote(legacy_db), node=_quote("failed:proposal:1")
        ),
    )
    assert status == 404


# --- CLI path resolution ------------------------------------------------------


def test_cli_db_inside_root_becomes_root_relative(tree: Path):
    root = tree / "results"

    assert resolve_cli_paths(str(root), "run_a/programs.sqlite") == (
        str(root),
        "run_a/programs.sqlite",
    )
    assert resolve_cli_paths(str(root), str(root / "run_a" / "programs.sqlite")) == (
        str(root),
        "run_a/programs.sqlite",
    )


def test_cli_db_outside_default_root_moves_root_to_db_dir(
    tree: Path, monkeypatch: pytest.MonkeyPatch
):
    elsewhere = tree / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    db = tree / "results" / "run_a" / "programs.sqlite"

    assert resolve_cli_paths(None, str(db)) == (str(db.parent), "programs.sqlite")


def test_cli_db_outside_explicit_root_is_an_error(tree: Path):
    with pytest.raises(ValueError, match="outside the root directory"):
        resolve_cli_paths(
            str(tree / "results" / "run_a"),
            str(tree / "results_private" / "secret.txt"),
        )


def test_cli_missing_db_is_an_error(tree: Path):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_cli_paths(str(tree / "results"), "run_a/missing.sqlite")
