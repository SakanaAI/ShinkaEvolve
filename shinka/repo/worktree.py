from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class MutabilityViolation(ValueError):
    """Raised when a candidate worktree edits immutable paths."""


@dataclass
class RepoWorktree:
    individual_id: str
    generation: int
    branch_name: str
    path: Path
    parent_commit: str
    policy_fingerprints: dict[str, str] = field(default_factory=dict)
    hidden_paths: list[str] = field(default_factory=list)


@dataclass
class WorktreeSnapshot:
    commit_sha: Optional[str]
    parent_commit: str
    diff: str
    changed_files: list[str] = field(default_factory=list)
    status: str = ""
    diff_stat: str = ""


def _run_git(
    repo_path: Path,
    args: list[str],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    completed = subprocess.run(
        command,
        cwd=str(cwd or repo_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize_repo_path(path)
    for pattern in patterns:
        normalized_pattern = _normalize_repo_path(pattern)
        if not normalized_pattern:
            continue
        if fnmatch.fnmatch(normalized, normalized_pattern):
            return True
        if normalized == normalized_pattern or normalized.startswith(
            f"{normalized_pattern}/"
        ):
            return True
    return False


def _has_glob_magic(path: str) -> bool:
    return any(char in path for char in "*?[")


def _tracked_paths_for_patterns(
    worktree_path: Path,
    patterns: Iterable[str],
) -> list[str]:
    matches: set[str] = set()
    for pattern in patterns:
        normalized = _normalize_repo_path(pattern)
        if not normalized:
            continue
        output = _run_git(
            worktree_path,
            ["ls-files", "--", normalized],
            cwd=worktree_path,
            check=False,
        ).stdout
        matches.update(
            _normalize_repo_path(line)
            for line in output.splitlines()
            if line.strip()
        )
    return sorted(matches)


def _visible_paths_for_patterns(root: Path, patterns: Iterable[str]) -> list[str]:
    matches: set[str] = set()
    for pattern in patterns:
        normalized = _normalize_repo_path(pattern)
        if not normalized:
            continue
        if _has_glob_magic(normalized):
            for path in root.glob(normalized):
                try:
                    matches.add(path.relative_to(root).as_posix())
                except ValueError:
                    continue
        else:
            path = root / normalized
            if path.exists() or path.is_symlink():
                matches.add(normalized)
    return sorted(matches)


def _existing_paths_for_patterns(root: Path, patterns: Iterable[str]) -> list[str]:
    matches = set(_tracked_paths_for_patterns(root, patterns))
    matches.update(_visible_paths_for_patterns(root, patterns))
    return sorted(matches, key=lambda item: len(Path(item).parts), reverse=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _make_readonly(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    if path.is_dir():
        for child in path.iterdir():
            _make_readonly(child)
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _copy_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        os.symlink(os.readlink(source), destination)
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(8192)
    except OSError:
        return False


def _gitlink_paths(worktree_path: Path) -> set[str]:
    output = _run_git(
        worktree_path,
        ["ls-files", "-s"],
        cwd=worktree_path,
        check=False,
    ).stdout
    paths: set[str] = set()
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[0] == "160000":
            paths.add(_normalize_repo_path(parts[3]))
    return paths


class WorktreeManager:
    LOCKFILE_NAMES = {
        "Cargo.lock",
        "Pipfile.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }

    def __init__(
        self,
        *,
        seed_repo_path: str,
        worktree_root: str,
        mutable_paths: Optional[list[str]] = None,
        immutable_paths: Optional[list[str]] = None,
        hidden_paths: Optional[list[str]] = None,
        ignore_paths: Optional[list[str]] = None,
        base_ref: str = "HEAD",
        max_file_bytes: Optional[int] = None,
        allow_binary_files: bool = True,
        allow_deletions: bool = True,
        allow_lockfile_changes: bool = True,
    ):
        self.seed_repo_path = Path(seed_repo_path).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.mutable_paths = mutable_paths or []
        self.immutable_paths = immutable_paths or []
        self.hidden_paths = hidden_paths or []
        self.ignore_paths = ignore_paths or [".git", ".shinka"]
        self.base_ref = base_ref
        self.max_file_bytes = max_file_bytes
        self.allow_binary_files = allow_binary_files
        self.allow_deletions = allow_deletions
        self.allow_lockfile_changes = allow_lockfile_changes

    def initialize_seed_repo(self) -> str:
        if not self.seed_repo_path.exists():
            raise FileNotFoundError(f"Seed repo does not exist: {self.seed_repo_path}")
        _run_git(self.seed_repo_path, ["rev-parse", "--is-inside-work-tree"]) # TODO: output goes nowhere
        self.worktree_root.mkdir(parents=True, exist_ok=True) 
        return self.resolve_ref(self.base_ref)

    def resolve_ref(self, ref: str) -> str:
        return _run_git(
            self.seed_repo_path, ["rev-parse", ref], check=True
        ).stdout.strip()

    def create_child_worktree(
        self,
        *,
        parent_commit: str,
        generation: int,
        individual_id: str,
    ) -> RepoWorktree:
        short_id = individual_id.replace("-", "")[:12]
        branch_name = f"shinka/gen-{generation}-{short_id}"
        path = self.worktree_root / f"gen_{generation}_{short_id}"
        if path.exists():
            raise FileExistsError(f"Worktree path already exists: {path}")

        _run_git(
            self.seed_repo_path,
            ["worktree", "add", "-b", branch_name, str(path), parent_commit],
        )
        return RepoWorktree(
            individual_id=individual_id,
            generation=generation,
            branch_name=branch_name,
            path=path,
            parent_commit=parent_commit,
        )

    def create_agent_worktree_view(
        self,
        worktree: RepoWorktree,
        *,
        hidden_paths: Optional[list[str]] = None,
    ) -> RepoWorktree:
        agent_path = worktree.path.parent / f"{worktree.path.name}__agent"
        if agent_path.exists():
            raise FileExistsError(f"Agent worktree path already exists: {agent_path}")

        _run_git(
            self.seed_repo_path,
            ["worktree", "add", "--detach", str(agent_path), worktree.parent_commit],
        )
        view_hidden_paths = list(
            dict.fromkeys([*self.hidden_paths, *(hidden_paths or [])])
        )
        agent_view = RepoWorktree(
            individual_id=worktree.individual_id,
            generation=worktree.generation,
            branch_name=f"{worktree.branch_name}-agent",
            path=agent_path,
            parent_commit=worktree.parent_commit,
            hidden_paths=view_hidden_paths,
        )
        self.hide_paths(agent_view, view_hidden_paths)
        self.freeze_immutable_paths(agent_view)
        return agent_view

    def write_policy_files(
        self,
        worktree: RepoWorktree,
        *,
        prompt_text: Optional[str] = None,
    ) -> Path:
        shinka_dir = worktree.path / ".shinka"
        shinka_dir.mkdir(parents=True, exist_ok=True)
        (shinka_dir / "mutable_paths.txt").write_text(
            "\n".join(self.mutable_paths) + ("\n" if self.mutable_paths else ""),
            encoding="utf-8",
        )
        (shinka_dir / "immutable_paths.txt").write_text(
            "\n".join(self.immutable_paths) + ("\n" if self.immutable_paths else ""),
            encoding="utf-8",
        )
        prompt_path = shinka_dir / "goal.md"
        if prompt_text is not None:
            prompt_path.write_text(prompt_text, encoding="utf-8")
        worktree.policy_fingerprints = {
            str(path.relative_to(worktree.path)): _sha256_file(path)
            for path in [
                shinka_dir / "mutable_paths.txt",
                shinka_dir / "immutable_paths.txt",
                prompt_path,
            ]
            if path.exists()
        }
        return prompt_path

    def hide_paths(self, worktree: RepoWorktree, hidden_paths: Iterable[str]) -> None:
        hidden = [_normalize_repo_path(path) for path in hidden_paths if path.strip()]
        if not hidden:
            return

        tracked_paths = _tracked_paths_for_patterns(worktree.path, hidden)
        if tracked_paths:
            _run_git(
                worktree.path,
                ["update-index", "--skip-worktree", "--", *tracked_paths],
                cwd=worktree.path,
            )

        for rel_path in _existing_paths_for_patterns(worktree.path, hidden):
            if _matches_any(rel_path, [".git"]):
                continue
            _remove_path(worktree.path / rel_path)

    def freeze_immutable_paths(self, worktree: RepoWorktree) -> None:
        for rel_path in _existing_paths_for_patterns(
            worktree.path,
            self.immutable_paths,
        ):
            path = worktree.path / rel_path
            _make_readonly(path)

    def enforce_hidden_paths_absent(self, worktree: RepoWorktree) -> None:
        if not worktree.hidden_paths:
            return
        visible_hidden_paths = _visible_paths_for_patterns(
            worktree.path,
            worktree.hidden_paths,
        )
        if visible_hidden_paths:
            raise MutabilityViolation(
                "; ".join(
                    f"{path}: hidden evaluation path is visible during generation"
                    for path in visible_hidden_paths
                )
            )

    def write_parent_id(self, worktree: RepoWorktree, parent_id: Optional[str]) -> None:
        """Write the parent individual ID into `.shinka/parent_id`.

        Because `.shinka/` is in `ignore_paths`, the agent cannot alter or
        commit this file, making parent-ID validation tamper-proof.
        """
        shinka_dir = worktree.path / ".shinka"
        shinka_dir.mkdir(parents=True, exist_ok=True)
        (shinka_dir / "parent_id").write_text(
            parent_id or "", encoding="utf-8",
        )

    @staticmethod
    def read_parent_id(worktree_path: Path) -> Optional[str]:
        """Read the stashed parent ID from `.shinka/parent_id`."""
        parent_id_path = worktree_path / ".shinka" / "parent_id"
        if not parent_id_path.exists():
            return None
        value = parent_id_path.read_text(encoding="utf-8").strip()
        return value or None

    def changed_files(self, worktree_path: Path, parent_commit: str) -> list[str]:
        output = _run_git(
            worktree_path,
            ["diff", "--name-only", parent_commit, "--"],
            cwd=worktree_path,
        ).stdout
        untracked = _run_git(
            worktree_path,
            ["ls-files", "--others", "--exclude-standard"],
            cwd=worktree_path,
        ).stdout
        paths = {
            _normalize_repo_path(line)
            for line in f"{output}\n{untracked}".splitlines()
            if line.strip()
        }
        return sorted(
            path for path in paths if not _matches_any(path, self.ignore_paths)
        )

    def diff_parent(self, worktree_path: Path, parent_commit: str) -> WorktreeSnapshot:
        diff = _run_git(
            worktree_path, ["diff", parent_commit, "--"], cwd=worktree_path
        ).stdout
        status = _run_git(
            worktree_path, ["status", "--porcelain"], cwd=worktree_path
        ).stdout
        diff_stat = _run_git(
            worktree_path, ["diff", "--stat", parent_commit, "--"], cwd=worktree_path
        ).stdout
        return WorktreeSnapshot(
            commit_sha=None,
            parent_commit=parent_commit,
            diff=diff,
            changed_files=self.changed_files(worktree_path, parent_commit),
            status=status,
            diff_stat=diff_stat,
        )

    def enforce_mutability(self, changed_files: Iterable[str]) -> None:
        violations: list[str] = []
        for path in changed_files:
            normalized = _normalize_repo_path(path)
            if (
                Path(path).is_absolute()
                or normalized.startswith("../")
                or normalized == ".."
            ):
                violations.append(f"{path}: path traversal is not allowed")
                continue
            if _matches_any(normalized, [".git", ".shinka"]):
                violations.append(f"{path}: protected path")
                continue
            if self.immutable_paths and _matches_any(normalized, self.immutable_paths):
                violations.append(f"{path}: immutable path")
                continue
            if self.mutable_paths and not _matches_any(normalized, self.mutable_paths):
                violations.append(f"{path}: outside mutable paths")
        if violations:
            raise MutabilityViolation("; ".join(violations))

    def verify_policy_files(self, worktree: RepoWorktree) -> None:
        violations = []
        for rel_path, expected_hash in worktree.policy_fingerprints.items():
            path = worktree.path / rel_path
            if not path.exists():
                violations.append(f"{rel_path}: policy file deleted")
            elif _sha256_file(path) != expected_hash:
                violations.append(f"{rel_path}: policy file modified")
        if violations:
            raise MutabilityViolation("; ".join(violations))

    def validate_snapshot(
        self,
        worktree: RepoWorktree,
        snapshot: WorktreeSnapshot,
    ) -> None:
        self.verify_policy_files(worktree)
        self.enforce_hidden_paths_absent(worktree)
        self.enforce_mutability(snapshot.changed_files)

        violations: list[str] = []
        deleted_files = set(
            _run_git(
                worktree.path,
                ["diff", "--name-only", "--diff-filter=D", worktree.parent_commit, "--"],
                cwd=worktree.path,
            ).stdout.splitlines()
        )
        gitlink_files = _gitlink_paths(worktree.path)

        for rel_path in snapshot.changed_files:
            normalized = _normalize_repo_path(rel_path)
            path = worktree.path / normalized

            if normalized in deleted_files and not self.allow_deletions:
                violations.append(f"{rel_path}: deletions are not allowed")
                continue
            if normalized in gitlink_files:
                violations.append(f"{rel_path}: submodule changes are not allowed")
                continue
            if (
                Path(normalized).name in self.LOCKFILE_NAMES
                and not self.allow_lockfile_changes
            ):
                violations.append(f"{rel_path}: dependency lockfile changes are not allowed")
                continue
            if not path.exists() or path.is_dir():
                continue
            if path.is_symlink():
                target = os.readlink(path)
                target_path = Path(target)
                resolved_target = (
                    target_path if target_path.is_absolute() else path.parent / target_path
                ).resolve()
                try:
                    resolved_target.relative_to(worktree.path.resolve())
                except ValueError:
                    violations.append(f"{rel_path}: symlink target escapes worktree")
                continue
            size = path.stat().st_size
            if self.max_file_bytes is not None and size > self.max_file_bytes:
                violations.append(f"{rel_path}: file exceeds max size {self.max_file_bytes}")
                continue
            if not self.allow_binary_files and _looks_binary(path):
                violations.append(f"{rel_path}: binary files are not allowed")

        if violations:
            raise MutabilityViolation("; ".join(violations))

    def apply_agent_view_changes(
        self,
        agent_view: RepoWorktree,
        target_worktree: RepoWorktree,
    ) -> WorktreeSnapshot:
        snapshot = self.diff_parent(agent_view.path, agent_view.parent_commit)
        self.validate_snapshot(agent_view, snapshot)

        for rel_path in snapshot.changed_files:
            source = agent_view.path / rel_path
            destination = target_worktree.path / rel_path
            if source.exists() or source.is_symlink():
                _copy_path(source, destination)
            elif self.allow_deletions:
                _remove_path(destination)
            else:
                raise MutabilityViolation(f"{rel_path}: deletions are not allowed")

        return snapshot

    def commit_child(
        self,
        worktree: RepoWorktree,
        *,
        message: Optional[str] = None,
    ) -> WorktreeSnapshot:
        snapshot = self.diff_parent(worktree.path, worktree.parent_commit)
        self.validate_snapshot(worktree, snapshot)
        if not snapshot.changed_files:
            return snapshot

        _run_git(worktree.path, ["add", "-A", "--", *snapshot.changed_files], cwd=worktree.path)
        _run_git(
            worktree.path,
            [
                "-c",
                "user.name=Shinka",
                "-c",
                "user.email=shinka@example.invalid",
                "commit",
                "-m",
                message
                or f"Shinka individual gen {worktree.generation} {worktree.individual_id}",
            ],
            cwd=worktree.path,
        )
        snapshot.commit_sha = _run_git(
            worktree.path, ["rev-parse", "HEAD"], cwd=worktree.path
        ).stdout.strip()
        return snapshot

    def cleanup_worktree(self, worktree: RepoWorktree, *, remove: bool = True) -> None:
        if not remove:
            return
        try:
            _run_git(
                self.seed_repo_path,
                ["worktree", "remove", "--force", str(worktree.path)],
                check=False,
            )
        finally:
            if worktree.path.exists():
                shutil.rmtree(worktree.path, ignore_errors=True)
