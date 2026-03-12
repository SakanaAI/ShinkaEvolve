from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_public_ci_excludes_secret_backed_tests() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert 'pytest -q -m "not requires_secrets"' in workflow


def test_integration_workflow_exists_for_secret_backed_tests() -> None:
    workflow = _read(".github/workflows/integration.yml")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "push:" in workflow
    assert 'pytest -q -m "requires_secrets"' in workflow
    assert "OPENAI_API_KEY" in workflow


def test_pytest_markers_are_registered() -> None:
    pyproject = _read("pyproject.toml")

    assert 'addopts = "--strict-markers"' in pyproject
    assert 'integration: live external/provider integration coverage' in pyproject
    assert 'requires_secrets: tests that need CI secrets or private credentials' in pyproject
