from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.launch import JobScheduler, LocalJobConfig
from shinka.repo import (
    WorktreeManager,
    build_initial_summary,
    build_summary_template,
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


def _add_seed_evaluator(seed_repo: Path) -> None:
    (seed_repo / "evaluate.py").write_text("print('hidden')\n", encoding="utf-8")
    _git(seed_repo, "add", "evaluate.py")
    _git(
        seed_repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "add evaluator",
    )


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


def test_summary_template_requires_agent_rewrite():
    summary = build_summary_template(
        individual_id="child",
        generation=1,
        parent_id="parent",
        parent_commit="abc",
    )

    result = validate_summary(
        summary,
        max_chars=12000,
    )

    assert not result.valid
    assert any("unresolved placeholder" in error for error in result.errors)


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

    with pytest.raises(Exception, match="immutable path"):
        manager.validate_snapshot(bad_worktree, bad_snapshot)


def test_agent_worktree_view_hides_evaluator_and_freezes_immutable_paths(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    _add_seed_evaluator(seed_repo)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
        hidden_paths=["evaluate.py"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="view123",
    )

    agent_view = manager.create_agent_worktree_view(
        worktree,
        hidden_paths=["evaluate.py"],
    )

    assert not (agent_view.path / "evaluate.py").exists()
    assert (agent_view.path / "README.md").exists()
    assert (agent_view.path / "README.md").stat().st_mode & stat.S_IWUSR == 0
    assert _git(agent_view.path, "status", "--short") == ""

    (agent_view.path / "evaluate.py").write_text("print('touch')\n", encoding="utf-8")
    snapshot = manager.diff_parent(agent_view.path, parent)
    with pytest.raises(Exception, match="hidden evaluation path is visible"):
        manager.validate_snapshot(agent_view, snapshot)


def test_agent_worktree_view_imports_only_mutable_changes(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    _add_seed_evaluator(seed_repo)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
        hidden_paths=["evaluate.py"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="import123",
    )
    agent_view = manager.create_agent_worktree_view(
        worktree,
        hidden_paths=["evaluate.py"],
    )

    (agent_view.path / "src" / "app.py").write_text("VALUE = 4\n", encoding="utf-8")
    view_snapshot = manager.apply_agent_view_changes(agent_view, worktree)
    canonical_snapshot = manager.diff_parent(worktree.path, parent)

    assert view_snapshot.changed_files == ["src/app.py"]
    assert canonical_snapshot.changed_files == ["src/app.py"]
    assert (worktree.path / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 4\n"
    assert (worktree.path / "README.md").read_text(encoding="utf-8") == "immutable\n"
    assert (worktree.path / "evaluate.py").read_text(encoding="utf-8") == "print('hidden')\n"


def test_worktree_manager_rejects_policy_tampering(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="tamper123",
    )

    manager.write_policy_files(worktree, prompt_text="change src only")
    (worktree.path / ".shinka" / "mutable_paths.txt").write_text(
        "src\nREADME.md\n",
        encoding="utf-8",
    )
    (worktree.path / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    snapshot = manager.diff_parent(worktree.path, parent)

    with pytest.raises(Exception, match="policy file modified"):
        manager.validate_snapshot(worktree, snapshot)


def test_worktree_manager_rejects_deleted_files(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="delete123",
    )
    (worktree.path / "src" / "app.py").unlink()
    snapshot = manager.diff_parent(worktree.path, parent)

    with pytest.raises(Exception, match="deletions are not allowed"):
        manager.validate_snapshot(worktree, snapshot)


def test_worktree_manager_rejects_binary_files(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="binary123",
    )
    (worktree.path / "src" / "blob.bin").write_bytes(b"abc\0def")
    snapshot = manager.diff_parent(worktree.path, parent)

    with pytest.raises(Exception, match="binary files are not allowed"):
        manager.validate_snapshot(worktree, snapshot)


def test_worktree_manager_rejects_symlink_escape(tmp_path):
    seed_repo = _make_seed_repo(tmp_path)
    manager = WorktreeManager(
        seed_repo_path=str(seed_repo),
        worktree_root=str(tmp_path / "worktrees"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
    )
    parent = manager.initialize_seed_repo()
    worktree = manager.create_child_worktree(
        parent_commit=parent,
        generation=1,
        individual_id="link123",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (worktree.path / "src" / "outside").symlink_to(outside)
    snapshot = manager.diff_parent(worktree.path, parent)

    with pytest.raises(Exception, match="symlink target escapes worktree"):
        manager.validate_snapshot(worktree, snapshot)


def test_repo_database_fields_roundtrip(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")),
        embedding_model=None,
    )
    repo = Program(
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
        agent_session_id="shinka-gen-0-repo1",
        agent_session_name="shinka-gen-0-repo1",
        agent_provider="codex",
        agent_model="gpt-test",
        correct=True,
    )
    db.add(repo)

    loaded = db.get("repo-1")

    assert loaded is not None
    assert loaded.individual_type == "repo"
    assert loaded.repo_commit == "abc"
    assert loaded.changed_files == ["src/app.py"]
    assert loaded.agent_session_id == "shinka-gen-0-repo1"
    assert loaded.agent_session_name == "shinka-gen-0-repo1"
    assert loaded.agent_provider == "codex"
    assert loaded.agent_model == "gpt-test"
    db.cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert "programs" in {row["name"] for row in db.cursor.fetchall()}


def test_repo_island_copies_preserve_repo_fields(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=3),
        embedding_model=None,
    )
    repo = Program(
        id="repo-root",
        code="# summary\n",
        language="repo",
        individual_type="repo",
        generation=0,
        code_diff="diff --git a/src/app.py b/src/app.py\n",
        repo_commit="abc",
        repo_parent_commit="parent",
        repo_diff="diff --git a/src/app.py b/src/app.py\n",
        repo_summary="# summary\n",
        summary_version="repo-individual-v1",
        changed_files=["src/app.py"],
        artifact_uri=str(tmp_path / "worktree"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
        agent_session_id="shinka-gen-0-root",
        agent_session_name="shinka-gen-0-root",
        agent_provider="codex",
        agent_model="gpt-test",
        combined_score=1.25,
        public_metrics={"valid": True},
        private_metrics={"lengths": [1.0, 2.0]},
        text_feedback=["line one", "line two"],
        complexity=1.0,
        correct=True,
        metadata={"source_job_id": "job-1"},
        system_prompt_id="prompt-1",
    )

    db.add(repo)

    db.cursor.execute(
        "SELECT * FROM programs WHERE id != ? ORDER BY island_idx",
        (repo.id,),
    )
    copies = db.cursor.fetchall()
    assert len(copies) == 2

    for copy in copies:
        assert copy["individual_type"] == "repo"
        assert copy["parent_id"] is None
        assert copy["repo_commit"] == "abc"
        assert copy["repo_parent_commit"] == "parent"
        assert copy["repo_diff"] == repo.repo_diff
        assert copy["repo_summary"] == "# summary\n"
        assert copy["summary_version"] == "repo-individual-v1"
        assert json.loads(copy["changed_files"]) == ["src/app.py"]
        assert copy["artifact_uri"] == str(tmp_path / "worktree")
        assert json.loads(copy["mutable_paths"]) == ["src"]
        assert json.loads(copy["immutable_paths"]) == ["README.md"]
        assert copy["agent_session_id"] == "shinka-gen-0-root"
        assert copy["agent_session_name"] == "shinka-gen-0-root"
        assert copy["agent_provider"] == "codex"
        assert copy["agent_model"] == "gpt-test"
        assert copy["system_prompt_id"] == "prompt-1"
        assert copy["code_diff"] == repo.code_diff
        assert copy["combined_score"] == 1.25
        assert json.loads(copy["public_metrics"]) == {"valid": True}
        assert json.loads(copy["private_metrics"]) == {"lengths": [1.0, 2.0]}
        assert copy["text_feedback"] == "line one\nline two"
        metadata = json.loads(copy["metadata"])
        assert metadata["source_job_id"] == "job-1"
        assert metadata["_is_island_copy"] is True
        assert metadata["_original_program_id"] == "repo-root"
        assert "_needs_island_copies" not in metadata


def test_dynamic_spawned_island_copy_preserves_repo_fields(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"),
            num_islands=1,
            enable_dynamic_islands=True,
            island_spawn_strategy="initial",
        ),
        embedding_model=None,
    )
    repo = Program(
        id="repo-root",
        code="# summary\n",
        language="repo",
        individual_type="repo",
        generation=0,
        repo_commit="abc",
        repo_parent_commit="parent",
        repo_diff="repo-diff",
        repo_summary="# summary\n",
        summary_version="repo-individual-v1",
        changed_files=["src/app.py"],
        artifact_uri=str(tmp_path / "worktree"),
        mutable_paths=["src"],
        immutable_paths=["README.md"],
        agent_session_id="session",
        agent_session_name="session-name",
        agent_provider="codex",
        agent_model="gpt-test",
        combined_score=1.25,
        complexity=1.0,
        correct=True,
        metadata={"source_job_id": "job-1"},
        system_prompt_id="prompt-1",
    )
    db.add(repo)

    assert db.island_manager.spawn_new_island()

    db.cursor.execute(
        "SELECT * FROM programs WHERE id != ? ORDER BY island_idx",
        (repo.id,),
    )
    spawned = db.cursor.fetchone()
    assert spawned is not None
    assert spawned["parent_id"] is None
    assert spawned["repo_commit"] == "abc"
    assert spawned["repo_parent_commit"] == "parent"
    assert spawned["repo_diff"] == "repo-diff"
    assert spawned["repo_summary"] == "# summary\n"
    assert spawned["summary_version"] == "repo-individual-v1"
    assert json.loads(spawned["changed_files"]) == ["src/app.py"]
    assert spawned["artifact_uri"] == str(tmp_path / "worktree")
    assert json.loads(spawned["mutable_paths"]) == ["src"]
    assert json.loads(spawned["immutable_paths"]) == ["README.md"]
    assert spawned["agent_session_id"] == "session"
    assert spawned["agent_session_name"] == "session-name"
    assert spawned["agent_provider"] == "codex"
    assert spawned["agent_model"] == "gpt-test"
    assert spawned["system_prompt_id"] == "prompt-1"
    metadata = json.loads(spawned["metadata"])
    assert metadata["source_job_id"] == "job-1"
    assert metadata["_spawned_island"] is True
    assert metadata["_spawned_from_program_id"] == "repo-root"


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
