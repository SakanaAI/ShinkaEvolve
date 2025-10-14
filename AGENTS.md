# Repository Guidelines

## Project Structure & Module Organization
- `shinka/`: core Python package for evolution runners, job configs, and database utilities.
- `configs/`: Hydra configuration presets; start new experiments by extending an existing YAML here.
- `tests/`: pytest suite covering edit strategies and evaluation tooling; mirror source layout when adding modules.
- `examples/`: runnable task templates, including notebooks, to demonstrate common workflows.
- `webui-react/` and `website/`: front-end clients; keep UI changes isolated from the core engine.

## Build, Test, and Development Commands
- `uv venv --python 3.12 && source .venv/bin/activate`: create and activate the recommended environment.
- `uv pip install -e .[dev]`: install the package with developer tooling.
- `genesis_launch variant=circle_packing_example`: run a reference experiment via the CLI entrypoint.
- `pytest`: execute the full unit test suite; use `pytest tests/test_edit_circle.py -k "smoke"` while iterating.
- `black shinka tests && isort shinka tests && flake8 shinka tests`: format and lint before submitting.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, and type hints for new public APIs.
- Modules, files, and functions use `snake_case`; classes follow `PascalCase`.
- Keep experiment configs declarative; prefer Hydra overrides (`genesis_launch +foo=bar`) over ad-hoc scripts.
- Document non-obvious behaviors with concise docstrings referencing evaluation assumptions.

## Testing Guidelines
- Favor pytest parameterization to cover candidate mutation edge cases.
- Place new tests alongside related modules under `tests/` using the `test_<feature>.py` pattern.
- Ensure evolution runs include deterministic seeds when feasible; capture fixtures for expensive evaluators.
- Add regression tests whenever logic affects scoring, database migrations, or patch synthesis.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes observed in history (`feat:`, `fix:`, `docs:`); keep the subject under 72 chars.
- Squash noise commits locally; PRs should include a one-paragraph summary plus CLI or screenshot evidence for UI changes.
- Link tracking issues and note required credentials or configs in the PR body.
- Request review once CI (formatting + pytest) is green; highlight remaining risks or TODOs inline.

## Configuration & Secrets
- Store API keys in `.env` (see `docs/getting_started.md`); never commit secrets or experiment artifacts.
- For multi-node or Slurm runs, duplicate configs into `configs/custom/` and document cluster dependencies in the PR.
