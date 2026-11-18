"""Regression tests for what an independent review of the hardened server found.

Each test drives a real server bound to an ephemeral loopback port.
"""

import json
from pathlib import Path
from typing import Iterator

import pytest

from shinka.webui import security
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
