"""Performance regression tests for the visualization server.

Covers three fixes on the fix/performance branch:

* P9  – /get_program_count is a *lightweight* change detector: it must fold in
        the failed-proposal generations via a cheap aggregate and must NOT
        materialize failed nodes or read any failure JSON from disk.
* P10 – /get_programs_summary is TTL-cached like /get_programs, under a
        namespaced key that does not collide with the full-programs payload.
* P11 – /get_programs supports opt-in ``limit`` / ``offset`` / ``include_code``
        narrowing while the default (no params) response is unchanged.

The server harness mirrors tests/test_webui_security.py: a ThreadingTCPServer
built from create_handler_factory(root), driven with real http.client requests
that carry a valid loopback Host header.
"""

import http.client
import json
import socketserver
import threading
import urllib.parse

import pytest

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.webui import visualization
from shinka.webui.visualization import create_handler_factory


@pytest.fixture(autouse=True)
def _clear_db_cache():
    """The server's db_cache is a module global; isolate it between tests."""
    visualization.db_cache.clear()
    yield
    visualization.db_cache.clear()


N_PROGRAMS = 4
# Failed proposals recorded across these generations; the endpoint counts
# DISTINCT generations, so the duplicate gen 5 must NOT be double-counted.
FAILED_GENERATIONS = [5, 6, 5]
DISTINCT_FAILED_GENERATIONS = 2
FAILURE_REASON = "boom-marker-should-not-be-served"


def _make_db(root, n_programs=N_PROGRAMS, failed_generations=FAILED_GENERATIONS):
    """Create a real programs.sqlite under ``root/run`` and return its client path.

    Programs are inserted with raw SQL (the schema is created by ProgramDatabase)
    to keep counts deterministic — the ``add()`` path spawns island copies.
    Failed proposals go through the production ``record_attempt_event`` helper.
    """
    run_dir = root / "run"
    run_dir.mkdir()
    db_file = run_dir / "programs.sqlite"

    db = ProgramDatabase(DatabaseConfig(db_path=str(db_file)), read_only=False)
    try:
        for i in range(n_programs):
            db.cursor.execute(
                "INSERT INTO programs (id, code, language, generation, timestamp, "
                "combined_score, correct, children_count, complexity, embedding, "
                "public_metrics, private_metrics, metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"p{i}",
                    f"print({i})\n",
                    "python",
                    i,
                    100.0 + i,
                    float(i),
                    1,
                    0,
                    1.0,
                    json.dumps([0.1 * i, 0.2 * i]),
                    "{}",
                    "{}",
                    "{}",
                ),
            )
        db.conn.commit()
        for gen in failed_generations:
            db.record_attempt_event(
                gen,
                "proposal",
                "failed",
                details={
                    "node_kind": "failed_proposal",
                    "failure_reason": FAILURE_REASON,
                    "failure_json_path": "does/not/exist/failure.json",
                },
            )
    finally:
        db.close()

    return "run/programs.sqlite"


class _Server:
    """Direct ThreadingTCPServer for the handler (no start_server chdir)."""

    def __init__(self, search_root):
        self.httpd = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), create_handler_factory(str(search_root))
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, headers={"Host": f"127.0.0.1:{self.port}"})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp, body

    def get_json(self, path):
        resp, body = self.get(path)
        assert resp.status == 200, (path, resp.status, body[:200])
        return json.loads(body), body

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(tmp_path):
    db_path = _make_db(tmp_path)
    srv = _Server(tmp_path)
    srv.db_path = db_path
    yield srv
    srv.close()


def _q(db_path):
    return urllib.parse.quote(db_path)


# --------------------------------------------------------------------------- P9


def test_program_count_shape_and_counts(server):
    result, body = server.get_json(f"/get_program_count?db_path={_q(server.db_path)}")

    # Only the two keys the client actually consumes.
    assert set(result.keys()) == {"count", "max_timestamp"}
    # Programs + DISTINCT failed-proposal generations (dup gen not double-counted).
    assert result["count"] == N_PROGRAMS + DISTINCT_FAILED_GENERATIONS
    # Failed proposals are stamped with time.time(), newer than program ts 100-103.
    assert result["max_timestamp"] is not None
    assert result["max_timestamp"] > 103.0


def test_program_count_does_not_materialize_failed_nodes(server):
    _result, body = server.get_json(f"/get_program_count?db_path={_q(server.db_path)}")
    text = body.decode("utf-8")

    # No per-node failure payload leaks into this lightweight endpoint: no
    # failure_reason (would require reading the row/details), no synthetic node
    # id, and no node-only keys. Proves _read_failure_json / node build skipped.
    assert FAILURE_REASON not in text
    assert "failure_reason" not in text
    assert "failed:proposal:" not in text
    for node_only_key in ("public_metrics", "text_feedback", "embedding"):
        assert node_only_key not in text


def test_program_count_matches_full_program_list_length(server):
    """The change-detector count must equal treeData length (get_programs)."""
    count_result, _ = server.get_json(
        f"/get_program_count?db_path={_q(server.db_path)}"
    )
    programs, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    assert count_result["count"] == len(programs)


# -------------------------------------------------------------------------- P10


def test_programs_summary_is_ttl_cached(server, tmp_path):
    path = f"/get_programs_summary?db_path={_q(server.db_path)}"
    first, _ = server.get_json(path)

    # Mutate the DB after the first fetch; a fresh read would see the new row.
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / server.db_path)), read_only=False
    )
    try:
        db.cursor.execute(
            "INSERT INTO programs (id, code, language, generation, timestamp) "
            "VALUES ('extra', 'x', 'python', 99, 999.0)"
        )
        db.conn.commit()
    finally:
        db.close()

    # Within the TTL the cached (stale) payload is returned, proving the cache.
    second, _ = server.get_json(path)
    assert second == first
    assert len(second) == N_PROGRAMS + DISTINCT_FAILED_GENERATIONS


def test_summary_cache_does_not_collide_with_programs_cache(server):
    """Namespaced key: summary and full-programs payloads stay distinct."""
    summaries, _ = server.get_json(
        f"/get_programs_summary?db_path={_q(server.db_path)}"
    )
    programs, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")

    # Summaries never ship source code; full programs do.
    assert all(not s.get("code") for s in summaries)
    assert any(p.get("code") for p in programs)


# -------------------------------------------------------------------------- P11


def test_get_programs_default_ships_full_code(server):
    programs, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    assert len(programs) == N_PROGRAMS + DISTINCT_FAILED_GENERATIONS
    # At least one real program carries its full source (default = unchanged).
    assert any(p.get("code") for p in programs)


def test_get_programs_limit_returns_fewer_rows(server):
    full, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    limited, _ = server.get_json(
        f"/get_programs?db_path={_q(server.db_path)}&limit=2"
    )
    assert len(limited) == 2
    assert limited == full[:2]


def test_get_programs_offset_paginates(server):
    full, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    page, _ = server.get_json(
        f"/get_programs?db_path={_q(server.db_path)}&limit=2&offset=1"
    )
    assert page == full[1:3]


def test_get_programs_include_code_false_strips_code(server):
    full, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    stripped, _ = server.get_json(
        f"/get_programs?db_path={_q(server.db_path)}&include_code=false"
    )
    assert len(stripped) == len(full)
    assert all(item.get("code") is None for item in stripped)


def test_get_programs_ignores_garbage_pagination_params(server):
    """Non-integer limit/offset are ignored, preserving default behavior."""
    full, _ = server.get_json(f"/get_programs?db_path={_q(server.db_path)}")
    same, _ = server.get_json(
        f"/get_programs?db_path={_q(server.db_path)}&limit=abc&offset=xyz"
    )
    assert same == full
