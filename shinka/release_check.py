from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


def _archive_members(archive_path: Path) -> list[str]:
    if archive_path.suffix == ".whl":
        with zipfile.ZipFile(archive_path) as archive:
            return sorted(
                name
                for name in archive.namelist()
                if name and not name.endswith("/")
            )

    if archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as archive:
            return sorted(
                member.name
                for member in archive.getmembers()
                if member.isfile()
            )

    raise ValueError(f"Unsupported archive format: {archive_path}")


def _is_ignored(repo_root: Path, relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relpath],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def find_ignored_archive_members(
    repo_root: Path, archive_paths: Iterable[Path]
) -> dict[str, list[str]]:
    offending: dict[str, list[str]] = {}
    for archive_path in archive_paths:
        ignored_members = [
            member
            for member in _archive_members(archive_path)
            if _is_ignored(repo_root, member)
        ]
        if ignored_members:
            offending[archive_path.name] = ignored_members
    return offending


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path.cwd()
    archive_paths = [Path(arg) for arg in args]
    offending = find_ignored_archive_members(repo_root, archive_paths)
    if not offending:
        return 0

    for archive_name, members in offending.items():
        print(f"{archive_name}:")
        for member in members:
            print(f"  {member}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
