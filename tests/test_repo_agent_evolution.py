from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shinka.database import DatabaseConfig, Repo, RepoDatabase
from shinka.launch import JobScheduler, LocalJobConfig
from shinka.repo import (
    WorktreeManager,
    build_initial_summary,
    validate_summary,
)


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _make_seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "seed"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "seed")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("immutable\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "add files")
    return repo


def test_summary_schema_validates():
    summary = build_initial_summary(
        individual_id="child",
        generation=1,
        commit_sha="abc",
    )

    result = validate_summary(
        summary,
        max_chars=12000,
    )

    assert result.valid
    assert result.schema_version == "repo-individual-v1"


def test_worktree_manager_preserves_seed_and_excludes_summary_from_children(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    seed_commit = manager.initialize_seed_repo()
    initial = manager.create_child_worktree(
        parent_commit=seed_commit,
        generation=0,
        individual_id="root123",
    )

    (initial.path / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (initial.path / ".shinka").mkdir()
    (initial.path / ".shinka" / "individual.md").write_text(
        build_initial_summary(
            individual_id="root123",
            generation=0,
            commit_sha="pending",
            changed_files=["src/app.py"],
        ),
        encoding="utf-8",
    )
    initial_snapshot = manager.commit_child(initial)

    assert initial_snapshot.commit_sha
    assert initial_snapshot.changed_files == ["src/app.py"]
    assert (seed_repo / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert ".shinka/individual.md" not in _git(
        initial.path, "ls-tree", "-r", "--name-only", "HEAD"
    )

    child = manager.create_child_worktree(
        parent_commit=initial_snapshot.commit_sha,
        generation=1,
        individual_id="child123",
    )
    assert (child.path / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert not (child.path / ".shinka" / "individual.md").exists()

    (child.path / "src" / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    (child.path / ".shinka").mkdir()
    (child.path / ".shinka" / "individual.md").write_text(
        build_initial_summary(
            individual_id="child123",
            generation=1,
            commit_sha="pending",
            changed_files=["src/app.py"],
        ),
        encoding="utf-8",
    )
    child_snapshot = manager.commit_child(child)

    assert child_snapshot.commit_sha
    assert child_snapshot.changed_files == ["src/app.py"]
    assert ".shinka/individual.md" not in _git(
        child.path, "ls-tree", "-r", "--name-only", "HEAD"
    )


def test_worktree_manager_enforces_immutable_paths(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    parent = manager.initialize_seed_repo()
    bad_worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=2,
        individual_id="bad123",
    )
    (bad_worktree.path / "README.md").write_text("changed\n", encoding="utf-8")
    bad_snapshot = manager.diff_parent(bad_worktree.path, parent)

    try:
        manager.enforce_mutability(bad_snapshot.changed_files)
    except Exception as exc:
        assert "immutable path" in str(exc)
    else:
        raise AssertionError("expected immutable path violation")


def test_repo_database_fields_roundtrip(tmp_path):
    db = RepoDatabase(
        DatabaseConfig(db_path=str(tmp_path / "repos.sqlite")),
        embedding_model=None,
    )
    repo = Repo(
        id="repo-1",
        code="# summary\n",
        language="repo",
        individual_type="repo",
        generation=0,
        repo_commit="abc",
        repo_parent_commit="parent",
        repo_diff="diff --git a/src/app.py b/src/app.py\n",
        repo_summary="# summary\n",
        summary_version="repo-individual-v1",
        changed_files=["src/app.py"],
        artifact_uri="/tmp/worktree",
        mutable_paths=["src"],
        immutable_paths=["README.md"],
        correct=True,
    )
    db.add(repo)

    loaded = db.get("repo-1")

    assert loaded is not None
    assert loaded.individual_type == "repo"
    assert loaded.repo_commit == "abc"
    assert loaded.changed_files == ["src/app.py"]
    db.cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert "repos" in {row["name"] for row in db.cursor.fetchall()}


def test_scheduler_passes_repo_path_and_cwd(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        "\n".join(
            [
                "import argparse, json, os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--repo_path', required=True)",
                "parser.add_argument('--results_dir', required=True)",
                "args = parser.parse_args()",
                "Path(args.results_dir).mkdir(parents=True, exist_ok=True)",
                "ok = Path(os.getcwd()).resolve() == Path(args.repo_path).resolve()",
                "Path(args.results_dir, 'metrics.json').write_text(json.dumps({'combined_score': 1.0 if ok else 0.0, 'public': {}, 'private': {}}))",
                "Path(args.results_dir, 'correct.json').write_text(json.dumps({'correct': ok, 'error': '' if ok else os.getcwd()}))",
            ]
        ),
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(eval_program_path=str(evaluator)),
    )

    results, _ = scheduler.run(str(repo), str(tmp_path / "results"), str(repo))

    assert results["correct"]["correct"] is True


def test_worktree_manager_parent_id_file(tmp_path):
    """Test the .shinka/parent_id tamper-proof lineage mechanism."""
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    seed_commit = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=seed_commit,
        generation=1,
        individual_id="child-abc",
    )

    # Write parent ID
    manager.write_parent_id(worktree, "parent-xyz")

    # Read it back
    read_back = WorktreeManager.read_parent_id(worktree.path)
    assert read_back == "parent-xyz"

    # Verify the file is in the .shinka directory
    parent_id_path = worktree.path / ".shinka" / "parent_id"
    assert parent_id_path.exists()
    assert parent_id_path.read_text(encoding="utf-8") == "parent-xyz"

    # None parent_id should read back as None
    manager.write_parent_id(worktree, None)
    assert WorktreeManager.read_parent_id(worktree.path) is None

    # Non-existent path should return None
    assert WorktreeManager.read_parent_id(tmp_path / "nonexistent") is None
