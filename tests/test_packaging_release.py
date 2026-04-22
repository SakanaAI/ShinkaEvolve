from __future__ import annotations

import subprocess
from pathlib import Path
import tarfile
import tomllib
import zipfile

import pytest

import shinka
import shinka.configs
from shinka.cli import launch as cli_launch
from shinka.release_check import find_ignored_archive_members


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _make_git_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / ".gitignore").write_text("ignored.txt\n*.tmp\n", encoding="utf-8")
    return repo_root


def _make_tar_gz(archive_path: Path, members: dict[str, str]) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for relpath, content in members.items():
            file_path = archive_path.parent / relpath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            archive.add(file_path, arcname=relpath)


def _make_whl(archive_path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relpath, content in members.items():
            archive.writestr(relpath, content)


def test_project_metadata_targets_pypi_release():
    pyproject = _read_pyproject()
    project = pyproject["project"]

    assert project["name"] == "shinka-evolve"
    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert project["scripts"] == {
        "shinka_launch": "shinka.cli.launch:main",
        "shinka_models": "shinka.cli.models:main",
        "shinka_run": "shinka.cli.run:main",
        "shinka_visualize": "shinka.webui.visualization:main",
    }

    setuptools_cfg = pyproject["tool"]["setuptools"]
    assert "script-files" not in setuptools_cfg
    assert setuptools_cfg["include-package-data"] is False
    assert pyproject["tool"]["setuptools"]["package-data"] == {
        "shinka": [
            "configs/*.yaml",
            "configs/*/*.yaml",
            "favicon.png",
            "embed/providers/*.csv",
            "llm/providers/*.csv",
            "webui/*.html",
            "webui/*.png",
            "webui/*.jpg",
        ]
    }


def test_readme_documents_package_install():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install shinka-evolve" in readme
    assert "uv pip install shinka-evolve" in readme
    assert "CHANGELOG.md" in readme
    assert "CONTRIBUTING.md" in readme
    assert "release_notes.md" not in readme


def test_changelog_tracks_current_package_version():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert shinka.__version__ == "0.0.5"
    assert "# Changelog" in changelog
    assert f"## {shinka.__version__} -" in changelog


def test_packaged_hydra_configs_live_inside_package():
    config_path = Path(shinka.__file__).resolve().parent / "configs" / "config.yaml"
    assert config_path.exists()


def test_packaged_hydra_configs_are_importable():
    assert Path(shinka.configs.__file__).resolve().name == "__init__.py"


def test_shinka_launch_preprocesses_global_args():
    processed = cli_launch.preprocess_args(
        ["task=circle_packing", "variant=default", "verbose=true"]
    )
    assert processed == [
        "task@_global_=circle_packing",
        "variant@_global_=default",
        "verbose=true",
    ]


def test_release_check_flags_gitignored_members(tmp_path: Path):
    repo_root = _make_git_repo(tmp_path)
    archive_path = tmp_path / "artifact.tar.gz"
    _make_tar_gz(
        archive_path,
        {
            "ok.txt": "ok\n",
            "ignored.txt": "nope\n",
            "nested/file.tmp": "nope\n",
        },
    )

    offending = find_ignored_archive_members(repo_root, [archive_path])
    assert offending == {
        archive_path.name: ["ignored.txt", "nested/file.tmp"],
    }


def test_release_check_handles_wheels(tmp_path: Path):
    repo_root = _make_git_repo(tmp_path)
    archive_path = tmp_path / "artifact.whl"
    _make_whl(
        archive_path,
        {
            "ok.py": "print('ok')\n",
            "ignored.txt": "nope\n",
        },
    )

    offending = find_ignored_archive_members(repo_root, [archive_path])
    assert offending == {archive_path.name: ["ignored.txt"]}


@pytest.mark.parametrize("archive_name", ["artifact.tar.gz", "artifact.whl"])
def test_release_check_accepts_clean_archives(tmp_path: Path, archive_name: str):
    repo_root = _make_git_repo(tmp_path)
    archive_path = tmp_path / archive_name
    members = {"pkg/module.py": "x = 1\n", "README.md": "ok\n"}

    if archive_name.endswith(".whl"):
        _make_whl(archive_path, members)
    else:
        _make_tar_gz(archive_path, members)

    assert find_ignored_archive_members(repo_root, [archive_path]) == {}
