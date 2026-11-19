"""Regression tests for what an independent review of the hardened server found.

Each test drives a real server bound to an ephemeral loopback port.
"""

import json
import os
import socket
from pathlib import Path
from typing import Iterator, List

import pytest

from shinka.webui import security, visualization
from webui_test_utils import (
    PNG_BYTES,
    log_failed_proposal,
    quote,
    request,
    serve,
    write_program_db,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """`base/results` is the search root; `base/private` must stay unreachable."""
    base = tmp_path.resolve()
    run = base / "results" / "run_a"
    plots = run / "gen_1" / "results" / "plots"
    plots.mkdir(parents=True)
    (plots / "p.png").write_bytes(PNG_BYTES)
    (run / "meta").mkdir()
    (run / "meta" / "meta_1.txt").write_text("# inside", encoding="utf-8")
    write_program_db(run / "programs.sqlite", "p0")

    private = base / "private"
    (private / "meta").mkdir(parents=True)
    (private / "meta" / "meta_1.txt").write_text("OUTSIDE NOTE", encoding="utf-8")
    (private / "secret.png").write_bytes(PNG_BYTES)
    (private / "secret.txt").write_text("OUTSIDE TEXT", encoding="utf-8")
    write_program_db(private / "programs.sqlite", "outside_program")
    return base


@pytest.fixture
def server(tree: Path) -> Iterator[int]:
    with serve(tree / "results") as port:
        yield port


# --- symlink loops -------------------------------------------------------------


@pytest.fixture
def looped_tree(tree: Path) -> Path:
    """A root holding a symlink loop next to a symlink that leaves the root.

    Before Python 3.13, ``os.path.realpath`` stops at the loop and returns the
    rest of the path unresolved, so ``loop/../out`` came back as ``<root>/out``.
    """
    root = tree / "results"
    (root / "loop").symlink_to("loop")
    (root / "out").symlink_to(tree / "private")
    return tree


def test_symlink_loop_does_not_hide_a_link_leaving_the_root(looped_tree: Path):
    root = str(looped_tree / "results")

    assert security.resolve_under_root(root, "loop/../out/secret.png") is None
    assert security.resolve_under_root(root, "loop/../out/missing.png") is None
    assert security.resolve_under_root(root, "loop/x.png") is None


def test_paths_inside_the_root_still_resolve(looped_tree: Path):
    root = looped_tree / "results"
    (root / "inside_link").symlink_to(root / "run_a")

    assert security.resolve_under_root(str(root), "run_a/not_yet.sqlite") == str(
        root / "run_a" / "not_yet.sqlite"
    )
    assert security.resolve_under_root(str(root), "inside_link/programs.sqlite") == str(
        root / "run_a" / "programs.sqlite"
    )


@pytest.mark.parametrize(
    "target",
    [
        "/plot_file/loop/../out/secret.png",
        "/get_programs?db_path=" + quote("loop/../out/programs.sqlite"),
        "/get_meta_content?processed_count=1&db_path="
        + quote("loop/../out/programs.sqlite"),
    ],
)
def test_symlink_loop_cannot_reach_files_outside_root(looped_tree: Path, target: str):
    with serve(looped_tree / "results") as port:
        status, _, body = request(port, target)

    assert status == 403
    assert body != PNG_BYTES
    assert b"outside_program" not in body
    assert b"OUTSIDE" not in body


def test_failure_json_cannot_reach_code_outside_root_through_a_loop(
    looped_tree: Path,
):
    run = looped_tree / "results" / "run_a"
    (run / "failure.json").write_text(
        json.dumps({"artifacts": {"generated_code_path": "loop/../out/secret.txt"}}),
        encoding="utf-8",
    )
    log_failed_proposal(
        run / "programs.sqlite", 5, {"failure_json_path": "run_a/failure.json"}
    )
    target = "/get_program_details?db_path={db}&id={node}".format(
        db=quote("run_a/programs.sqlite"), node=quote("failed:proposal:5")
    )

    with serve(looped_tree / "results") as port:
        status, _, body = request(port, target)

    assert status == 200
    assert json.loads(body)["code"] is None


# --- names the request check must not refuse ------------------------------------


@pytest.mark.parametrize(
    "name",
    ["run　wide", "run nbsp", "run narrow", "run‍joiner"],
    ids=["ideographic-space", "no-break-space", "narrow-no-break-space", "zwj"],
)
def test_run_directories_with_unicode_spaces_are_served(tree: Path, name: str):
    run = tree / "results" / name
    run.mkdir()
    write_program_db(run / "programs.sqlite", "p_unicode")
    db_path = quote(f"{name}/programs.sqlite")

    with serve(tree / "results") as port:
        page_status, _, _ = request(port, "/viz_tree.html?db_path=" + db_path)
        status, _, body = request(port, "/get_programs?db_path=" + db_path)

    assert page_status == 200
    assert status == 200
    assert b"p_unicode" in body


@pytest.mark.parametrize("control", ["%C2%85", "%C2%9B", "%7F", "%09"])
def test_control_characters_beyond_ascii_are_still_refused(server: int, control: str):
    target = f"/get_programs?db_path=run_a{control}/programs.sqlite"

    assert request(server, target)[0] == 400


# --- controls the first round of tests did not pin ------------------------------


@pytest.mark.parametrize(
    "name",
    ["note.svg", "page.html", "run_a/programs.sqlite", "run_a/meta/meta_1.txt"],
)
def test_plot_file_refuses_everything_but_images(tree: Path, name: str):
    root = tree / "results"
    (root / "note.svg").write_text("<svg onload='MARKER'/>", encoding="utf-8")
    (root / "page.html").write_text("<script>MARKER</script>", encoding="utf-8")

    with serve(root) as port:
        status, _, body = request(port, "/plot_file/" + name)

    assert status == 403
    assert b"MARKER" not in body
    assert b"# inside" not in body


def test_plot_file_checks_the_type_of_the_file_a_link_points_to(tree: Path):
    root = tree / "results"
    (root / "fake.png").symlink_to(root / "run_a" / "meta" / "meta_1.txt")

    with serve(root) as port:
        status, _, body = request(port, "/plot_file/fake.png")

    assert status == 403
    assert b"# inside" not in body


def _run_with_db(tree: Path, name: str) -> Path:
    run = tree / "results" / name
    run.mkdir()
    write_program_db(run / "programs.sqlite", "p0")
    return run


def test_meta_directory_linked_out_of_root_is_not_listed_or_read(tree: Path):
    run = _run_with_db(tree, "run_meta_dir")
    (run / "meta").symlink_to(tree / "private" / "meta")
    db_path = quote("run_meta_dir/programs.sqlite")

    with serve(tree / "results") as port:
        listing = request(port, "/get_meta_files?db_path=" + db_path)
        content = request(
            port, "/get_meta_content?processed_count=1&db_path=" + db_path
        )

    assert listing[0] == 404
    assert content[0] == 404
    assert b"OUTSIDE" not in listing[2] + content[2]


@pytest.mark.parametrize("endpoint", ["/get_meta_content", "/download_meta_pdf"])
def test_meta_file_linked_out_of_root_is_not_read(tree: Path, endpoint: str):
    run = _run_with_db(tree, "run_meta_file")
    (run / "meta").mkdir()
    (run / "meta" / "meta_1.txt").symlink_to(tree / "private" / "meta" / "meta_1.txt")
    target = f"{endpoint}?processed_count=1&db_path=" + quote(
        "run_meta_file/programs.sqlite"
    )

    with serve(tree / "results") as port:
        status, _, body = request(port, target)

    assert status == 404
    assert b"OUTSIDE" not in body


def test_plots_directory_linked_out_of_root_is_not_listed_or_served(tree: Path):
    outside_plots = tree / "private" / "gen_1" / "results" / "plots"
    outside_plots.mkdir(parents=True)
    (outside_plots / "o.png").write_bytes(PNG_BYTES)
    run = _run_with_db(tree, "run_plots")
    (run / "gen_1").symlink_to(tree / "private" / "gen_1")

    with serve(tree / "results") as port:
        listing = request(
            port,
            "/get_plots?generation=1&program_id=p0&db_path="
            + quote("run_plots/programs.sqlite"),
        )
        plot = request(port, "/plot_file/run_plots/gen_1/results/plots/o.png")

    assert (listing[0], json.loads(listing[2])) == (200, [])
    assert plot[0] == 403
    assert plot[2] != PNG_BYTES


def test_allowed_hosts_reach_the_request_handler(tree: Path):
    with serve(tree / "results", allowed_hosts=["viz.example.org"]) as port:
        allowed = request(port, "/list_databases", host_header="viz.example.org")
        machine = request(
            port, "/list_databases", host_header=f"{socket.gethostname()}:{port}"
        )
        other = request(port, "/list_databases", host_header="other.example.org")

    assert allowed[0] == 200
    assert machine[0] == 200
    assert other[0] == 403


@pytest.mark.parametrize(
    "host", ["evillocalhost", "xlocalhost:8000", "localhost.evil.example", "node170"]
)
def test_host_names_that_only_resemble_an_allowed_one_are_refused(host: str):
    allowed = security.build_allowed_hosts("node17")

    assert not security.is_allowed_host(host, allowed)


# --- the list of databases ------------------------------------------------------


def test_database_linked_out_of_root_is_left_out_of_the_list(
    tree: Path, capsys: pytest.CaptureFixture[str]
):
    # Every other endpoint refuses such a file, so listing it only leads to
    # a row that never loads.
    root = tree / "results"
    (root / "run_out").mkdir()
    (root / "run_out" / "programs.sqlite").symlink_to(
        tree / "private" / "programs.sqlite"
    )
    (root / "run_in").mkdir()
    (root / "run_in" / "programs.sqlite").symlink_to(root / "run_a" / "programs.sqlite")

    with serve(root) as port:
        status, _, body = request(port, "/list_databases")

    assert status == 200
    assert sorted(db["path"] for db in json.loads(body)) == [
        "run_a/programs.sqlite",
        "run_in/programs.sqlite",
    ]
    assert "run_out/programs.sqlite" in capsys.readouterr().out


# --- cost of confinement --------------------------------------------------------


def test_confining_many_paths_does_not_resolve_the_root_each_time(
    tree: Path, monkeypatch: pytest.MonkeyPatch
):
    # A poll confines one path per failed proposal; resolving the root again
    # for each of them doubled its file-system calls.
    root = str(tree / "results")
    handler = visualization.DatabaseRequestHandler.__new__(
        visualization.DatabaseRequestHandler
    )
    handler.search_root = root
    handler._real_root = root
    resolved: List[str] = []
    realpath = os.path.realpath

    def recording_realpath(path, **kwargs):
        resolved.append(str(path))
        return realpath(path, **kwargs)

    monkeypatch.setattr(security.os.path, "realpath", recording_realpath)

    for _ in range(3):
        assert handler._confine("run_a/programs.sqlite") is not None

    assert resolved == [root + "/run_a/programs.sqlite"] * 3


# --- the programs cache ---------------------------------------------------------


def test_cached_programs_belong_to_the_database_that_was_asked_for(tmp_path: Path):
    # The cache used the request's relative path as its key, so two servers
    # in one process (as in this test suite) answered from each other's data.
    roots = {}
    for name in ("first", "second"):
        roots[name] = tmp_path.resolve() / name
        (roots[name] / "run_a").mkdir(parents=True)
        write_program_db(roots[name] / "run_a" / "programs.sqlite", f"from_{name}")
    target = "/get_programs?db_path=" + quote("run_a/programs.sqlite")

    with serve(roots["first"]) as port:
        first_body = request(port, target)[2]
    with serve(roots["second"]) as port:
        second_body = request(port, target)[2]

    assert b"from_first" in first_body
    assert b"from_second" in second_body
    assert b"from_first" not in second_body


# --- failed proposals of a run opened from another directory --------------------


def _failed_node(port: int, generation: int) -> dict:
    target = "/get_program_details?db_path={db}&id={node}".format(
        db=quote("run_a/programs.sqlite"), node=quote(f"failed:proposal:{generation}")
    )
    status, _, body = request(port, target)
    assert status == 200, body[:300]
    return json.loads(body)


def _write_failed_generation(gen_dir: Path, recorded_dir: str) -> None:
    """A failed generation whose files record paths under ``recorded_dir``."""
    gen_dir.mkdir(parents=True)
    (gen_dir / "main.py").write_text("FAILED_CODE = 1\n", encoding="utf-8")
    (gen_dir / "failure.json").write_text(
        json.dumps(
            {
                "language": "python",
                "attempts": 3,
                "artifacts": {"generated_code_path": f"{recorded_dir}/main.py"},
            }
        ),
        encoding="utf-8",
    )


def test_failed_proposal_is_found_when_the_run_was_started_elsewhere(tree: Path):
    # The run records these paths relative to the directory it was started
    # in (here the parent of the root), so they do not resolve under the root.
    run = tree / "results" / "run_a"
    _write_failed_generation(run / "gen_9", "results/run_a/gen_9")
    log_failed_proposal(
        run / "programs.sqlite",
        9,
        {"failure_json_path": "results/run_a/gen_9/failure.json"},
    )

    with serve(tree / "results") as port:
        node = _failed_node(port, 9)

    assert node["code"] == "FAILED_CODE = 1\n"
    assert node["metadata"]["attempts"] == 3


def test_failed_proposal_fallback_stays_inside_the_root(tree: Path):
    run = tree / "results" / "run_a"
    _write_failed_generation(tree / "private" / "gen_10", "results/run_a/gen_10")
    (run / "gen_10").symlink_to(tree / "private" / "gen_10")
    log_failed_proposal(
        run / "programs.sqlite",
        10,
        {"failure_json_path": "results/run_a/gen_10/failure.json"},
    )

    with serve(tree / "results") as port:
        node = _failed_node(port, 10)

    assert node["code"] is None
    assert "attempts" not in node["metadata"]
