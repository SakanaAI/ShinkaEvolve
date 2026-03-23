# Contributing to `shinka`

Thank you for your interest in contributing to `shinka` and ShinkaEvolve. We welcome code contributions, bug reports, feature requests, documentation improvements, and representative benchmark results.

## Contributing code using pull requests

We use git for development, and contributions are expected to come in through pull requests.

Follow these steps to contribute code:

1. Fork the repository on GitHub.
2. Install Python >= 3.10 and `uv`.
3. Clone your fork and install the project with development dependencies:

```bash
git clone https://github.com/SakanaAI/ShinkaEvolve.git
cd ShinkaEvolve
uv sync --dev
```

4. Add the main repository as an upstream remote:

```bash
git remote add upstream https://github.com/SakanaAI/ShinkaEvolve.git
```

5. Create a branch for your change:

```bash
git checkout -b name-of-change
```

6. Implement your change. If the change fixes a bug or guards against a regression, add or update tests when it fits.
7. Run the built-in checks from the repository root before opening a pull request:

```bash
uv run ruff check tests --exclude tests/file.py
uv run mypy --follow-imports=skip --ignore-missing-imports tests/test_*.py tests/conftest.py
uv run --with pytest-cov pytest -q -m "not requires_secrets" --cov=shinka --cov-report=term-missing --cov-report=xml:coverage.xml
```

If your change touches provider integrations or other secret-backed paths and you have the required credentials configured, also run:

```bash
uv run pytest -q -m "requires_secrets"
```

8. Commit your change with a clear message, sync with the latest `main`, then push your branch:

```bash
git add path/to/file.py
git commit -m "feat: describe the change"
git fetch upstream
git rebase upstream/main
git push --set-upstream origin name-of-change
```

9. Open a pull request using the repository template and include the requested context.

## Issue structure

Please use the GitHub issue templates. Good issues are specific and reproducible.

Bug reports should include:

- A short summary
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Logs, stack traces, screenshots, or a minimal script when available

Feature requests should include:

- The problem or motivation
- The proposed change
- Alternatives you considered
- A validation plan or example use case

## Pull request structure

Please use the pull request template. Every PR should include:

- A concise summary of the change
- Why the change is needed
- Linked issues or related context
- Tests run and their outcomes
- Risks, limitations, or backward-compatibility notes when relevant

If you change user-facing behavior, docs, CLI output, or WebUI behavior, include updated documentation or screenshots where appropriate.

## Core program evolution pipeline changes

Changes to the core program evolution pipeline require extra evidence. If your PR changes parent selection, mutation/edit generation, archive updates, prompt evolution, novelty logic, evaluation scheduling, proposal oversubscription, island behavior, or other central evolution logic, include:

- Results on at least one representative runnable example
- A comparison against a baseline
- The exact commands or config overrides used
- The metric(s) you are comparing
- A short interpretation of the result

Do not add random benchmark tasks or examples just to justify a PR. Use a representative task that highlights the capability you are changing, for example a new programming language backend, a new execution mode, or a new core optimization behavior.

The baseline can be the current `main` branch, a tagged release, or an explicitly named prior configuration. Make the comparison easy for reviewers to reproduce.

## AI-assisted contributions

AI-assisted issue reports and pull requests are welcome, but not AI slop. Every submission should be human-verified, technically understood by the author, and edited into a clean, minimal, reviewable change.

We will spend roughly as much time reviewing a PR as the human author spent creating it. Tid for tad. If a contributor did not invest the time to validate, simplify, benchmark, and explain a change, we will not invest that time on review either.

Please take Occam's razor seriously. Prefer the smallest change that solves the problem clearly. A 1000-line implementation for a niche feature will not be merged.

## License

By contributing, you agree that your contributions will be licensed under the repository's Apache 2.0 license.
