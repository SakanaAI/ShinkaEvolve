# Scripts

This directory contains utility scripts for the Genesis project.

## check_pinned_deps.py

**Purpose**: Pre-commit hook to enforce dependency pinning in `pyproject.toml`.

**What it does**:
- Checks that all dependencies use `==` for version pinning
- Validates dependencies in:
  - `[project] dependencies`
  - `[tool.uv] dev-dependencies`
  - `[build-system] requires`
- Ensures `requires-python` uses `==` operator
- Fails the commit if any dependency is unpinned or uses `>=`, `~=`, etc.

**Usage**:
```bash
# Run manually
python scripts/check_pinned_deps.py

# Automatically runs via pre-commit hook when pyproject.toml is changed
```

**Why pin dependencies?**
- **Reproducibility**: Ensures everyone gets the exact same versions
- **Stability**: Prevents unexpected breaking changes from new versions
- **Security**: Makes it easier to audit and track dependency vulnerabilities
- **CI/CD**: Guarantees consistent builds across environments

## Pre-commit Hooks

### Hadolint (Dockerfile Linting)

**Purpose**: Lint Dockerfiles to ensure best practices and catch common mistakes.

**What it does**:
- Checks Dockerfiles for best practices
- Validates instruction syntax
- Warns about security issues
- Suggests improvements for caching and build optimization
- Runs automatically on any file named `Dockerfile` or matching `*.dockerfile`

**Common checks**:
- `DL3008`: Pin versions in apt-get install
- `DL3009`: Delete apt-get lists after installing
- `DL3015`: Avoid additional packages by specifying `--no-install-recommends`
- `DL3059`: Multiple consecutive RUN instructions should be consolidated
- And many more...

**Usage**:
```bash
# Run manually on a Dockerfile
docker run --rm -i hadolint/hadolint:v2.12.0 hadolint < Dockerfile

# Automatically runs via pre-commit hook when Dockerfiles are changed
```

**Documentation**: https://github.com/hadolint/hadolint

### Installing Pre-commit Hooks

To enable the pre-commit hooks:

```bash
# Install pre-commit (already in dev dependencies)
pip install pre-commit

# Install the git hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

Once installed, the hooks will automatically run when you commit changes to `pyproject.toml`.
