"""Security regression tests for the visualization server.

Covers the path-traversal containment that gates every db_path/meta/plot handler,
plus the DNS-rebinding Host check, absent CORS wildcard, nosniff, and disabled
directory listing.
"""

import http.client
import os
import socketserver
import threading

import pytest

from shinka.webui.visualization import (
    DatabaseRequestHandler,
    PathValidationError,
    create_handler_factory,
)


def _handler(root):
    # Build a handler without running the socket-bound __init__.
    handler = object.__new__(DatabaseRequestHandler)
    handler.search_root = str(root)
    return handler


def test_resolve_accepts_path_within_root(tmp_path):
    (tmp_path / "run").mkdir()
    db = tmp_path / "run" / "programs.sqlite"
    db.write_text("x")
    handler = _handler(tmp_path)
    assert handler._resolve_within_root("run/programs.sqlite") == os.path.realpath(
        str(db)
    )


def test_resolve_rejects_absolute_path(tmp_path):
    handler = _handler(tmp_path)
    with pytest.raises(PathValidationError):
        handler._resolve_within_root("/etc/passwd")


def test_resolve_rejects_parent_traversal(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (tmp_path / "secret.db").write_text("secret")
    handler = _handler(served)
    with pytest.raises(PathValidationError):
        handler._resolve_within_root("../secret.db")


def test_resolve_rejects_symlink_escape(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    outside = tmp_path / "outside.db"
    outside.write_text("outside")
    os.symlink(outside, served / "link.db")
    handler = _handler(served)
    with pytest.raises(PathValidationError):
        handler._resolve_within_root("link.db")


def test_resolve_rejects_sibling_prefix_dir(tmp_path):
    # Root "<x>/served" must not allow reaching a sibling "<x>/served_bak".
    served = tmp_path / "served"
    served.mkdir()
    sibling = tmp_path / "served_bak"
    sibling.mkdir()
    (sibling / "leak.db").write_text("leak")
    handler = _handler(served)
    with pytest.raises(PathValidationError):
        handler._resolve_within_root("../served_bak/leak.db")


class _Server:
    """Direct ThreadingTCPServer for the handler (no start_server chdir)."""

    def __init__(self, search_root):
        self.httpd = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), create_handler_factory(str(search_root))
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Host": host} if host else {}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp, body

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "programs.sqlite").write_text("db")
    srv = _Server(tmp_path)
    yield srv
    srv.close()


def test_http_rejects_absolute_db_path(server):
    resp, _ = server.get(
        "/get_programs?db_path=/etc/passwd", host=f"127.0.0.1:{server.port}"
    )
    assert resp.status == 403


def test_http_rejects_foreign_host_header(server):
    # DNS-rebinding: a request whose Host is an attacker domain is refused.
    resp, _ = server.get("/list_databases", host="evil.example.com")
    assert resp.status == 403


def test_http_json_has_no_cors_wildcard_and_sets_nosniff(server):
    resp, _ = server.get("/list_databases", host=f"127.0.0.1:{server.port}")
    assert resp.status == 200
    assert resp.getheader("Access-Control-Allow-Origin") is None
    assert resp.getheader("X-Content-Type-Options") == "nosniff"


def test_list_directory_override_forbids(tmp_path):
    # The override must refuse to render an auto-generated index unconditionally.
    handler = object.__new__(DatabaseRequestHandler)
    sent = {}
    handler.send_error = lambda code, msg=None: sent.update(code=code, msg=msg)
    result = handler.list_directory(str(tmp_path))
    assert result is None
    assert sent["code"] == 403
