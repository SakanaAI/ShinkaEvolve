from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_contributing_guide_documents_repo_checks_and_pipeline_evidence() -> None:
    contributing = _read("CONTRIBUTING.md")

    assert "# Contributing to `shinka`" in contributing
    assert "uv sync --dev" in contributing
    assert 'uv run ruff check tests --exclude tests/file.py' in contributing
    assert (
        'uv run --with pytest-cov pytest -q -m "not requires_secrets"'
        in contributing
    )
    assert "core program evolution pipeline" in contributing
    assert "comparison against a baseline" in contributing
    assert "examples/circle_packing" in contributing


def test_issue_templates_enforce_expected_structure() -> None:
    bug_template = _read(".github/ISSUE_TEMPLATE/bug_report.yml")
    feature_template = _read(".github/ISSUE_TEMPLATE/feature_request.yml")

    for required in [
        "label: Summary",
        "label: Steps to reproduce",
        "label: Expected behavior",
        "label: Actual behavior",
        "label: Environment",
    ]:
        assert required in bug_template

    for required in [
        "label: Summary",
        "label: Motivation",
        "label: Proposed change",
        "label: Alternatives considered",
        "label: Validation plan or example",
    ]:
        assert required in feature_template


def test_pull_request_template_requires_testing_and_baseline_context() -> None:
    pr_template = _read(".github/pull_request_template.md")

    for required in [
        "## Summary",
        "## Why",
        "## Testing",
        "## Core evolution pipeline evidence",
        "Baseline used for comparison",
    ]:
        assert required in pr_template
