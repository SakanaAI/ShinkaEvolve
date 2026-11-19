"""Tests for the ``shinka_visualize`` command line and the Host names it accepts."""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shinka.webui import security, visualization
from webui_test_utils import serve, write_program_db


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """A root directory holding one run."""
    root = tmp_path.resolve() / "results"
    (root / "run_a").mkdir(parents=True)
    write_program_db(root / "run_a" / "programs.sqlite")
    return root


# --- command line ---------------------------------------------------------------


class _IdleServer:
    """Stands in for a bound server: ``main`` returns once serving ends."""

    server_address = ("127.0.0.1", 4321)
    allowed_hosts = frozenset({"localhost"})

    def serve_forever(self) -> None:
        return None


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: List[str]) -> Dict[str, Any]:
    """Run ``main`` without binding; return what it bound and what it opened."""
    seen: Dict[str, Any] = {}

    def fake_make_server(port, search_root, host, allowed_hosts):
        seen.update(
            port=port, root=search_root, host=host, allowed_hosts=list(allowed_hosts)
        )
        return _IdleServer()

    monkeypatch.setattr(visualization, "make_server", fake_make_server)
    monkeypatch.setattr(
        visualization.webbrowser,
        "open_new_tab",
        lambda url: seen.update(opened=url),
    )
    monkeypatch.setattr(sys, "argv", ["shinka_visualize", *argv])
    visualization.main()
    return seen


def test_cli_listens_on_loopback_unless_told_otherwise(
    results: Path, monkeypatch: pytest.MonkeyPatch
):
    seen = _run_cli(monkeypatch, [str(results)])

    assert seen["host"] == "127.0.0.1"
    assert seen["allowed_hosts"] == []
    assert seen["root"] == str(results)


def test_cli_passes_host_and_allowed_hosts_to_the_server(
    results: Path, monkeypatch: pytest.MonkeyPatch
):
    seen = _run_cli(
        monkeypatch,
        [
            str(results),
            "--host",
            "0.0.0.0",
            "--allowed-host",
            "viz.example.org",
            "--allowed-host",
            "alias",
        ],
    )

    assert seen["host"] == "0.0.0.0"
    assert seen["allowed_hosts"] == ["viz.example.org", "alias"]


def test_cli_opens_the_tree_view_with_a_root_relative_db_path(
    results: Path, monkeypatch: pytest.MonkeyPatch
):
    db = results / "run_a" / "programs.sqlite"

    seen = _run_cli(monkeypatch, [str(results), "--db", str(db), "--open"])

    assert seen["opened"] == (
        "http://localhost:4321/viz_tree.html?db_path=run_a%2Fprograms.sqlite"
    )


def test_cli_exits_when_the_port_is_taken(
    results: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with serve(results) as taken_port:
        monkeypatch.setattr(
            sys,
            "argv",
            ["shinka_visualize", str(results), "--port", str(taken_port)],
        )

        with pytest.raises(SystemExit) as exit_info:
            visualization.main()

    assert exit_info.value.code == 1
    assert f"cannot listen on 127.0.0.1 port {taken_port}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "host, url",
    [
        ("127.0.0.1", "http://localhost:8888/"),
        ("0.0.0.0", "http://localhost:8888/"),
        ("::", "http://localhost:8888/"),
        ("192.168.0.75", "http://192.168.0.75:8888/"),
        ("node17", "http://node17:8888/"),
        ("fd00::17", "http://[fd00::17]:8888/"),
    ],
)
def test_printed_url_names_an_address_the_server_answers_on(host: str, url: str):
    assert visualization.browser_url(host, 8888) == url


# --- names a server on the network answers to ----------------------------------


@pytest.fixture
def node17(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine called ``node17`` whose resolver knows its full name."""
    monkeypatch.setattr(security.socket, "gethostname", lambda: "node17")
    monkeypatch.setattr(
        security.socket, "getfqdn", lambda name="": "node17.cluster.example.org"
    )


@pytest.mark.parametrize(
    "host",
    [
        "node17:8888",
        "node17.cluster.example.org:8888",
        "node17.local:8888",
        "NODE17.Local.",
        "localhost:8888",
    ],
)
def test_network_server_answers_to_the_machine_names(node17: None, host: str):
    allowed = security.build_allowed_hosts("0.0.0.0")

    assert security.is_allowed_host(host, allowed)


@pytest.mark.parametrize(
    "host", ["node17.evil.example", "cluster.example.org", "local", "node170.local"]
)
def test_network_server_refuses_other_names(node17: None, host: str):
    allowed = security.build_allowed_hosts("0.0.0.0")

    assert not security.is_allowed_host(host, allowed)


def test_full_name_of_the_machine_also_covers_its_short_names(
    monkeypatch: pytest.MonkeyPatch,
):
    full_name = "node17.cluster.example.org"
    monkeypatch.setattr(security.socket, "gethostname", lambda: full_name)
    monkeypatch.setattr(security.socket, "getfqdn", lambda name="": full_name)

    allowed = security.build_allowed_hosts("0.0.0.0")

    assert {"node17", "node17.local", full_name} <= allowed


@pytest.mark.parametrize("bind_host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_server_does_not_ask_the_resolver_for_names(
    monkeypatch: pytest.MonkeyPatch, bind_host: str
):
    # The full name comes from the network's resolver. A server only this
    # machine can reach has no use for it, and trusting it would let that
    # resolver choose a name the Host check accepts.
    def no_lookup(name: str = "") -> str:
        raise AssertionError("resolver consulted for a loopback server")

    monkeypatch.setattr(security.socket, "gethostname", lambda: "node17")
    monkeypatch.setattr(security.socket, "getfqdn", no_lookup)

    allowed = security.build_allowed_hosts(bind_host)

    assert security.is_allowed_host("node17:8000", allowed)
    assert not security.is_allowed_host("node17.local:8000", allowed)


def test_start_up_lists_the_names_the_server_answers_to(
    results: Path, capsys: pytest.CaptureFixture[str]
):
    httpd = visualization.make_server(0, str(results), "127.0.0.1", ["viz.example.org"])
    try:
        visualization._print_banner(httpd, "127.0.0.1", str(results))
    finally:
        httpd.server_close()

    names_line = next(
        line for line in capsys.readouterr().out.splitlines() if "Answering" in line
    )
    assert "localhost" in names_line
    assert "viz.example.org" in names_line
    assert "--allowed-host" in names_line


def test_allowed_names_leave_out_addresses_and_reverse_lookup_names(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(security.socket, "gethostname", lambda: "node17")
    monkeypatch.setattr(
        security.socket, "getfqdn", lambda name="": "75.0.168.192.in-addr.arpa"
    )

    allowed = security.build_allowed_hosts("0.0.0.0", ["10.1.2.3", "alias"])

    assert allowed == {"localhost", "node17", "node17.local", "alias"}
    assert security.is_allowed_host("10.1.2.3:8000", allowed)


# --- paths given on the command line --------------------------------------------


def test_root_that_is_a_file_is_reported_as_such(results: Path):
    db = results / "run_a" / "programs.sqlite"

    with pytest.raises(ValueError, match="is not a directory"):
        visualization.resolve_cli_paths(str(db), None)


def test_missing_root_is_reported_as_missing(results: Path):
    with pytest.raises(ValueError, match="does not exist"):
        visualization.resolve_cli_paths(str(results / "nowhere"), None)
