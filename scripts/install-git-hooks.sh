#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hooks_path="$repo_root/.githooks"

if [[ ! -d "$hooks_path" ]]; then
  echo "Missing hooks directory: $hooks_path" >&2
  exit 1
fi

chmod +x "$hooks_path/pre-push"
git config core.hooksPath "$hooks_path"

echo "Git hooks installed. core.hooksPath=$hooks_path"
