from __future__ import annotations

import fnmatch
import logging
import shutil
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


class WorktreeManager:
    def __init__(
        self,
        *,
        seed_repo_path: str,
        worktree_root: str,
        mutable_paths: Optional[list[str]] = None,
        immutable_paths: Optional[list[str]] = None,
        ignore_paths: Optional[list[str]] = None,
        base_ref: str = "HEAD",
    ):
        self.seed_repo_path = Path(seed_repo_path).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.mutable_paths = mutable_paths or []
        self.immutable_paths = immutable_paths or []
        self.ignore_paths = ignore_paths or [".git", ".shinka"]
        self.base_ref = base_ref

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
        return prompt_path

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
        return sorted(path for path in paths if not _matches_any(path, self.ignore_paths))

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
            if normalized.startswith("../") or normalized == "..":
                violations.append(f"{path}: path traversal is not allowed")
                continue
            if _matches_any(normalized, self.ignore_paths):
                continue
            if self.immutable_paths and _matches_any(normalized, self.immutable_paths):
                violations.append(f"{path}: immutable path")
                continue
            if self.mutable_paths and not _matches_any(normalized, self.mutable_paths):
                violations.append(f"{path}: outside mutable paths")
        if violations:
            raise MutabilityViolation("; ".join(violations))

    def commit_child(
        self,
        worktree: RepoWorktree,
        *,
        message: Optional[str] = None,
    ) -> WorktreeSnapshot:
        snapshot = self.diff_parent(worktree.path, worktree.parent_commit)
        self.enforce_mutability(snapshot.changed_files)
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

