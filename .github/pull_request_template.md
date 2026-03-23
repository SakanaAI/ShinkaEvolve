## Summary

- What changed?

## Why

- Why is this change needed?

## Linked issue or context

- Issue number, discussion, or background.

## Testing

- Commands run:
  - `uv run ruff check tests --exclude tests/file.py`
  - `uv run mypy --follow-imports=skip --ignore-missing-imports tests/test_*.py tests/conftest.py`
  - `uv run --with pytest-cov pytest -q -m "not requires_secrets" --cov=shinka --cov-report=term-missing --cov-report=xml:coverage.xml`
- Optional secret-backed tests:
  - `uv run pytest -q -m "requires_secrets"`
- Results:

## Risks and compatibility

- Breaking changes, migrations, limitations, or follow-up work.

## Core evolution pipeline evidence

If this PR changes the core program evolution pipeline, provide:

- Example or benchmark used
- Baseline used for comparison
- Exact command/config used
- Metric comparison
- Short interpretation

## Docs and UI

- Docs updated if needed
- Screenshots attached if UI/WebUI behavior changed
