"""The committed lockfile must match pyproject.toml and CI must install from it."""

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def _key(name: str) -> str:
    return str(canonicalize_name(name))


def _declared_requirements() -> dict[str, Requirement]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = map(Requirement, pyproject["project"]["dependencies"])
    return {_key(req.name): req for req in requirements}


def _lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _locked_versions(lock: dict, name: str) -> list[str]:
    return [
        package["version"] for package in lock["package"] if package["name"] == name
    ]


def test_lockfile_records_the_declared_dependencies():
    """`uv lock --check` in miniature: a stale lock hides a broken fresh install."""
    lock = _lock()
    project = next(
        package for package in lock["package"] if package["name"] == "shinka-evolve"
    )
    locked = {
        _key(entry["name"]): Requirement(
            entry["name"] + entry.get("specifier", "")
        )
        for entry in project["metadata"]["requires-dist"]
        if "extra ==" not in entry.get("marker", "")
    }

    declared = _declared_requirements()

    assert sorted(locked) == sorted(declared)
    for name, requirement in declared.items():
        assert locked[name].specifier == requirement.specifier, name


def test_locked_versions_satisfy_the_declared_bounds():
    lock = _lock()

    for name, requirement in _declared_requirements().items():
        versions = _locked_versions(lock, name)
        assert versions, name
        for version in versions:
            assert requirement.specifier.contains(version), (name, version)


def test_openai_floor_accepts_the_admin_api_key_argument():
    """llm/client.py always passes admin_api_key, which openai 2.34.0 introduced."""
    openai_requirement = _declared_requirements()["openai"]

    assert not openai_requirement.specifier.contains("2.33.0")
    assert openai_requirement.specifier.contains("2.34.0")


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_workflows_install_from_the_lockfile(workflow: Path):
    """Without --locked, `uv sync` re-resolves and CI stops testing the lock."""
    sync_commands = re.findall(r"uv sync[^\n]*", workflow.read_text(encoding="utf-8"))

    for command in sync_commands:
        assert "--locked" in command, command


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_workflows_with_oidc_tokens_pin_actions_by_commit(workflow: Path):
    """A moved tag must not run new code in a job that can mint an OIDC token."""
    text = workflow.read_text(encoding="utf-8")
    if "id-token: write" not in text:
        pytest.skip("workflow does not request an OIDC token")

    actions = re.findall(r"uses:\s*(\S+)", text)

    assert actions
    for action in actions:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action), action
