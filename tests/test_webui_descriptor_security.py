import os

import pytest

from shinka.webui.visualization import DatabaseRequestHandler, PathValidationError

requires_descriptor_traversal = pytest.mark.skipif(
    not (
        os.open in os.supports_dir_fd
        and os.listdir in os.supports_fd
        and os.stat in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    ),
    reason="race-safe descriptor traversal requires Unix openat support",
)


def _handler(root):
    handler = object.__new__(DatabaseRequestHandler)
    handler.search_root = str(root)
    handler._canonical_search_root = os.path.realpath(root)
    return handler


@requires_descriptor_traversal
def test_text_read_rejects_file_symlink_swap_after_resolution(tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    target = served / "artifact.txt"
    target.write_text("safe", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    handler = _handler(served)
    resolve_path = handler._resolve_within_root

    def resolve_then_swap(path):
        resolved = resolve_path(path)
        target.unlink()
        target.symlink_to(outside)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)

    with pytest.raises(PathValidationError):
        handler._read_text_within_root("artifact.txt")


@requires_descriptor_traversal
def test_text_read_rejects_parent_symlink_swap_after_resolution(
    tmp_path, monkeypatch
):
    served = tmp_path / "served"
    artifact_dir = served / "run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact.txt").write_text("safe", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "artifact.txt").write_text("secret", encoding="utf-8")
    handler = _handler(served)
    resolve_path = handler._resolve_within_root

    def resolve_then_swap(path):
        resolved = resolve_path(path)
        artifact_dir.rename(served / "parked")
        artifact_dir.symlink_to(outside_dir)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)

    with pytest.raises(PathValidationError):
        handler._read_text_within_root("run/artifact.txt")


@requires_descriptor_traversal
def test_text_read_accepts_symlink_resolving_within_search_root(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("safe", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(target)
    handler = _handler(tmp_path)

    assert handler._read_text_within_root("link.txt") == "safe"


def test_text_read_preserves_fallback_without_descriptor_traversal(tmp_path, monkeypatch):
    target = tmp_path / "artifact.txt"
    target.write_text("safe", encoding="utf-8")
    handler = _handler(tmp_path)
    monkeypatch.setattr(handler, "_supports_descriptor_traversal", lambda: False)

    assert handler._read_text_within_root("artifact.txt") == "safe"


@requires_descriptor_traversal
def test_text_read_uses_pinned_root_after_ancestor_swap(tmp_path):
    parent = tmp_path / "parent"
    served = parent / "served"
    served.mkdir(parents=True)
    (served / "artifact.txt").write_text("safe", encoding="utf-8")
    outside_parent = tmp_path / "outside-parent"
    outside_served = outside_parent / "served"
    outside_served.mkdir(parents=True)
    (outside_served / "artifact.txt").write_text("secret", encoding="utf-8")
    handler = _handler(served)
    handler._search_root_descriptor = handler._open_search_root_descriptor(served)

    parent.rename(tmp_path / "parked-parent")
    parent.symlink_to(outside_parent)

    try:
        with pytest.raises(PathValidationError):
            handler._read_text_within_root("artifact.txt")
    finally:
        os.close(handler._search_root_descriptor)


@requires_descriptor_traversal
def test_text_read_rejects_parent_component_from_changed_resolution(
    tmp_path, monkeypatch
):
    served = tmp_path / "served"
    served.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "artifact.txt"
    secret.write_text("secret", encoding="utf-8")
    handler = _handler(served)
    handler._search_root_descriptor = handler._open_search_root_descriptor(served)
    monkeypatch.setattr(handler, "_resolve_within_root", lambda path: str(secret))

    try:
        with pytest.raises(PathValidationError):
            handler._read_text_within_root("artifact.txt")
    finally:
        os.close(handler._search_root_descriptor)


@requires_descriptor_traversal
def test_database_listing_uses_pinned_root_after_ancestor_swap(tmp_path):
    parent = tmp_path / "parent"
    served = parent / "served"
    served.mkdir(parents=True)
    (served / "safe.sqlite").write_text("", encoding="utf-8")
    outside_parent = tmp_path / "outside-parent"
    outside_served = outside_parent / "served"
    outside_served.mkdir(parents=True)
    (outside_served / "secret.sqlite").write_text("", encoding="utf-8")
    handler = _handler(served)
    handler._search_root_descriptor = handler._open_search_root_descriptor(served)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)

    parent.rename(tmp_path / "parked-parent")
    parent.symlink_to(outside_parent)

    try:
        handler.handle_list_databases()
        assert [database["path"] for database in sent["data"]] == ["safe.sqlite"]
    finally:
        os.close(handler._search_root_descriptor)


@requires_descriptor_traversal
def test_database_listing_skips_unreadable_subdirectory(tmp_path):
    (tmp_path / "safe.sqlite").write_text("", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "secret.sqlite").write_text("", encoding="utf-8")
    blocked.chmod(0)
    handler = _handler(tmp_path)

    try:
        assert handler._walk_files_within_root() == ["safe.sqlite"]
    finally:
        blocked.chmod(0o700)


@requires_descriptor_traversal
def test_database_listing_handles_tree_deeper_than_fd_limit(tmp_path):
    resource = pytest.importorskip("resource")
    directory = tmp_path
    relative_parts = []
    for depth in range(48):
        name = f"d{depth}"
        relative_parts.append(name)
        directory /= name
        directory.mkdir()
    (directory / "programs.sqlite").write_text("", encoding="utf-8")
    handler = _handler(tmp_path)

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard_limit < 32:
        pytest.skip("file descriptor hard limit is below test threshold")
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, hard_limit))
    try:
        expected = os.path.join(*relative_parts, "programs.sqlite")
        assert handler._walk_files_within_root() == [expected]
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


@requires_descriptor_traversal
def test_database_listing_walk_ignores_non_database_files(tmp_path):
    (tmp_path / "programs.sqlite").write_text("", encoding="utf-8")
    for index in range(20):
        (tmp_path / f"artifact-{index}.txt").write_text("x", encoding="utf-8")

    assert _handler(tmp_path)._walk_files_within_root() == ["programs.sqlite"]


@requires_descriptor_traversal
def test_database_listing_preserves_os_walk_sibling_order(tmp_path):
    for name in ("gamma", "beta", "alpha"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "programs.sqlite").write_text("", encoding="utf-8")

    expected = [
        os.path.relpath(os.path.join(root, filename), tmp_path)
        for root, _, files in os.walk(tmp_path)
        for filename in files
    ]

    assert _handler(tmp_path)._walk_files_within_root() == expected


@requires_descriptor_traversal
def test_directory_listing_rejects_symlink_swap_after_resolution(
    tmp_path, monkeypatch
):
    served = tmp_path / "served"
    meta_dir = served / "run" / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta_1.txt").write_text("safe", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "meta_2.txt").write_text("secret", encoding="utf-8")
    handler = _handler(served)
    resolve_path = handler._resolve_within_root

    def resolve_then_swap(path):
        resolved = resolve_path(path)
        meta_dir.rename(served / "parked")
        meta_dir.symlink_to(outside_dir)
        return resolved

    monkeypatch.setattr(handler, "_resolve_within_root", resolve_then_swap)

    with pytest.raises(PathValidationError):
        handler._list_directory_within_root("run/meta")


def test_meta_listing_propagates_directory_race_rejection(tmp_path):
    run_dir = tmp_path / "run"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)
    (run_dir / "programs.sqlite").write_text("", encoding="utf-8")
    handler = _handler(tmp_path)

    def reject_listing(path):
        raise PathValidationError("path changed during access")

    handler._list_directory_within_root = reject_listing

    with pytest.raises(PathValidationError):
        handler.handle_get_meta_files("run/programs.sqlite")
