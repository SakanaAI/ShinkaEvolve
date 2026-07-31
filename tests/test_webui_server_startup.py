import http.client
import os
import socket
import threading

import pytest

from shinka.webui import visualization
from shinka.webui.visualization import (
    DatabaseRequestHandler,
    _bind_server,
    _bound_host_from_server_address,
    _browser_url,
    create_handler_factory,
)


def test_ipv6_host_selects_ipv6_server():
    server = _bind_server("::1", 0, DatabaseRequestHandler)
    try:
        assert server.address_family == socket.AF_INET6
    finally:
        server.server_close()


def test_empty_host_preserves_ipv4_wildcard_bind():
    server = _bind_server("", 0, DatabaseRequestHandler)
    try:
        assert server.address_family == socket.AF_INET
        assert server.server_address[0] == "0.0.0.0"
    finally:
        server.server_close()


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 is unavailable")
@pytest.mark.parametrize("host_header", ["[::1]", "[::1]:{port}"])
def test_ipv6_server_accepts_bracketed_host_headers(tmp_path, host_header):
    try:
        server = _bind_server(
            "::1",
            0,
            create_handler_factory(os.fspath(tmp_path)),
        )
    except OSError as exc:
        pytest.skip(f"IPv6 loopback bind unavailable: {exc}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    connection = http.client.HTTPConnection("::1", port, timeout=5)
    try:
        connection.putrequest("GET", "/list_databases", skip_host=True)
        connection.putheader("Host", host_header.format(port=port))
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        assert response.status == 200
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("host_header", ["[::1]", "[::1]:8000"])
def test_ipv6_host_header_is_accepted_with_or_without_port(host_header):
    handler = object.__new__(DatabaseRequestHandler)
    handler.allowed_hosts = frozenset({"[::1]"})
    handler.headers = {"Host": host_header}

    assert handler._host_allowed()


def test_handler_factory_configures_database_snapshot_limit(tmp_path, monkeypatch):
    captured = {}
    factory = create_handler_factory(
        os.fspath(tmp_path),
        max_database_snapshot_bytes=1234,
    )

    def capture_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(DatabaseRequestHandler, "__init__", capture_init)

    factory()

    assert captured["max_database_snapshot_bytes"] == 1234


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, float("inf")])
def test_handler_factory_rejects_invalid_database_snapshot_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="snapshot size"):
        create_handler_factory(
            os.fspath(tmp_path),
            max_database_snapshot_bytes=limit,
        )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, float("inf")])
def test_start_server_rejects_invalid_database_snapshot_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="snapshot size"):
        visualization.start_server(
            0,
            os.fspath(tmp_path),
            max_database_snapshot_bytes=limit,
        )


@pytest.mark.parametrize(
    ("bound_host", "expected"),
    [
        ("::1", "http://[::1]:8765/"),
        ("192.168.1.20", "http://192.168.1.20:8765/"),
        ("127.0.0.1", "http://127.0.0.1:8765/"),
        ("0.0.0.0", "http://127.0.0.1:8765/"),
        ("::", "http://[::1]:8765/"),
    ],
)
def test_browser_url_uses_reachable_bound_address(bound_host, expected):
    assert _browser_url(bound_host, 8765) == expected


def test_browser_url_encodes_database_path():
    url = _browser_url("127.0.0.2", 8765, "run/a b.sqlite")

    assert url == (
        "http://127.0.0.2:8765/viz_tree.html?db_path=run%2Fa+b.sqlite"
    )


def test_scoped_ipv6_browser_url_preserves_interface(monkeypatch):
    monkeypatch.setattr(socket, "if_indextoname", lambda scope_id: "eth0")

    bound_host = _bound_host_from_server_address(("fe80::1", 8765, 0, 3))

    assert bound_host == "fe80::1%eth0"
    assert _browser_url(bound_host, 8765) == "http://[fe80::1%25eth0]:8765/"


def test_bind_server_falls_back_to_next_resolved_address(monkeypatch):
    addresses = [
        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 8000, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 8000)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: addresses)

    class FailingServer:
        def __init__(self, address, request_handler):
            raise OSError("IPv6 unavailable")

    expected_server = object()
    monkeypatch.setattr(
        visualization,
        "_server_class_for_family",
        lambda family: FailingServer if family == socket.AF_INET6 else (
            lambda address, request_handler: expected_server
        ),
    )

    assert _bind_server("dual-stack.example", 8000, DatabaseRequestHandler) is (
        expected_server
    )


def test_readiness_callback_can_request_server(tmp_path):
    callback_complete = threading.Event()
    errors = []

    def probe(httpd):
        connection = http.client.HTTPConnection(
            str(httpd.server_address[0]),
            int(httpd.server_address[1]),
            timeout=5,
        )
        try:
            connection.request("GET", "/list_databases")
            response = connection.getresponse()
            response.read()
            assert response.status == 200
        except (AssertionError, OSError, http.client.HTTPException) as exc:
            errors.append(exc)
        finally:
            connection.close()
            httpd.shutdown()
            callback_complete.set()

    original_directory = os.getcwd()
    try:
        visualization.start_server(
            0,
            os.fspath(tmp_path),
            host="127.0.0.1",
            on_ready=probe,
        )
    finally:
        os.chdir(original_directory)

    assert callback_complete.wait(timeout=5)
    assert errors == []


def test_main_propagates_server_startup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["shinka_visualize", os.fspath(tmp_path)],
    )
    monkeypatch.setattr(
        visualization,
        "start_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("bind failed")),
    )
    with pytest.raises(OSError, match="bind failed"):
        visualization.main()


def test_main_configures_database_snapshot_limit(tmp_path, monkeypatch):
    captured = {}

    def capture_start_server(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "sys.argv",
        [
            "shinka_visualize",
            os.fspath(tmp_path),
            "--max-database-snapshot-gib",
            "4.5",
        ],
    )
    monkeypatch.setattr(visualization, "start_server", capture_start_server)

    visualization.main()

    assert captured["max_database_snapshot_bytes"] == int(4.5 * 1024**3)


@pytest.mark.parametrize("limit", ["0", "1e-300", "1e300", "nan", "inf"])
def test_main_rejects_unrepresentable_database_snapshot_limit(
    tmp_path, monkeypatch, limit
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "shinka_visualize",
            os.fspath(tmp_path),
            "--max-database-snapshot-gib",
            limit,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        visualization.main()
