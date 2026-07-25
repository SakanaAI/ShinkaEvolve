"""Launch the Headless CLI in Docker with a minimal provider auth seed.

Headless mounts each agent's auth seed paths read-only under
``/tmp/headless-host-home`` and its container bootstrap then runs
``cp -R /tmp/headless-host-home/. "$HOME"/`` into a tmpfs home. When a seed
path is a directory the whole tree is mounted and copied, so agents that keep
conversation logs next to their credentials (antigravity keeps hundreds of
megabytes under ``~/.gemini/antigravity-cli``) push that entire tree into
container RAM on every proposal.

This module stages a throwaway ``HOME`` holding only the credential files the
agent needs, then points Headless at it, so the container copy stays in the
kilobyte range.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BASE_COMMAND = "npx -y @roberttlange/headless"
BASE_COMMAND_ENV = "SHINKA_HEADLESS_DOCKER_BASE_COMMAND"
IMAGE_ENV = "SHINKA_HEADLESS_DOCKER_IMAGE"
PLATFORM_ENV = "SHINKA_HEADLESS_DOCKER_PLATFORM"
EXTRA_ARGS_ENV = "SHINKA_HEADLESS_DOCKER_ARGS"
EXTRA_SEED_ENV = "SHINKA_HEADLESS_DOCKER_SEED_EXTRA"

# Mirrors the ``seedPaths`` table of @roberttlange/headless. Kept local so the
# wrapper does not pay an extra CLI round trip per proposal;
# ``tests/test_headless_docker.py`` checks it against ``--show-config``.
AGENT_SEED_PATHS: dict[str, tuple[str, ...]] = {
    "acp": (".config/acp",),
    "antigravity": (".gemini/antigravity-cli", ".gemini/config"),
    "claude": (
        ".claude.json",
        ".claude/settings.json",
        ".claude/.credentials.json",
        ".claude/auth.json",
    ),
    "codex": (".codex/auth.json", ".codex/config.toml"),
    "cursor": (".cursor/cli-config.json",),
    "gemini": (
        ".gemini/google_accounts.json",
        ".gemini/settings.json",
        ".gemini/state.json",
        ".gemini/trustedFolders.json",
        ".gemini/installation_id",
    ),
    "opencode": (".config/opencode",),
    "pi": (".pi/agent/auth.json", ".pi/agent/settings.json"),
}

# Directory seed paths are never copied wholesale. Where the credential files
# are known, only those are staged; otherwise the generic rule in
# ``_stage_directory_seed`` keeps top-level regular files and drops subtrees.
SEED_DIRECTORY_ALLOWLIST: dict[str, tuple[str, ...]] = {
    ".gemini/antigravity-cli": (
        "antigravity-oauth-token",
        "settings.json",
        "installation_id",
        "jetski_state.pbtxt",
    ),
}

# Codex writes desktop themes, plugin marketplaces and MCP server commands into
# the same config as its model preferences. Those entries point at host-only
# paths, so only the keys that still mean something inside a container are
# carried over.
CODEX_CONFIG_KEYS = (
    "model",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
    "preferred_auth_method",
    "service_tier",
)
CODEX_CONFIG_DEFAULTS = ('service_tier = "fast"',)
CODEX_SERVICE_TIERS = frozenset({"fast", "flex"})

MAX_SEED_FILE_BYTES = 4 * 1024 * 1024
MAX_SEED_TOTAL_BYTES = 16 * 1024 * 1024

# Subcommands and flags that never reach an agent container, so they run
# against the real home instead of a staged one.
PASSTHROUGH_FLAGS = frozenset(
    {"--check", "--help", "-h", "--list", "--show-config", "--version", "-v"}
)
PASSTHROUGH_SUBCOMMANDS = frozenset(
    {"attach", "cron", "docker", "rename", "run", "send"}
)

DOCKER_DESKTOP_BIN = "/Applications/Docker.app/Contents/Resources/bin"


def base_command() -> list[str]:
    raw_command = os.getenv(BASE_COMMAND_ENV, DEFAULT_BASE_COMMAND).strip()
    if not raw_command:
        raise ValueError(f"{BASE_COMMAND_ENV} cannot be empty.")
    return shlex.split(raw_command)


def seed_paths(agent: str) -> tuple[str, ...]:
    paths = AGENT_SEED_PATHS.get(agent, ())
    extra = os.getenv(EXTRA_SEED_ENV, "")
    additional = tuple(part for part in extra.split(os.pathsep) if part.strip())
    return paths + additional


class SeedBudget:
    def __init__(self, *, total_bytes: int = MAX_SEED_TOTAL_BYTES) -> None:
        self.remaining = total_bytes
        self.skipped: list[str] = []

    def copy(self, source: Path, destination: Path, label: str) -> None:
        size = source.stat().st_size
        if size > MAX_SEED_FILE_BYTES:
            self.skipped.append(f"{label} ({size} bytes exceeds per-file limit)")
            return
        if size > self.remaining:
            self.skipped.append(f"{label} ({size} bytes exceeds remaining budget)")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
        self.remaining -= size


def _stage_directory_seed(
    *,
    source: Path,
    destination: Path,
    relative: str,
    budget: SeedBudget,
) -> None:
    allowlist = SEED_DIRECTORY_ALLOWLIST.get(relative)
    if allowlist is not None:
        candidates = [source / name for name in allowlist]
    else:
        candidates = sorted(source.iterdir())
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        budget.copy(
            candidate,
            destination / candidate.name,
            f"{relative}/{candidate.name}",
        )


def _minimal_codex_config(source: Path) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    if source.is_file():
        for raw_line in source.read_text(errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("["):
                break
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or key not in CODEX_CONFIG_KEYS or key in seen:
                continue
            value = value.strip()
            if not value or value.startswith(("[", "{")):
                continue
            if key == "service_tier":
                normalized_value = value.strip('"\'')
                if normalized_value not in CODEX_SERVICE_TIERS:
                    continue
            seen.add(key)
            lines.append(f"{key} = {value}")
    for default in CODEX_CONFIG_DEFAULTS:
        key = default.split("=", 1)[0].strip()
        if key not in seen:
            lines.append(default)
    return "\n".join(lines) + "\n"


def stage_auth_home(*, agent: str, host_home: Path, stage_home: Path) -> SeedBudget:
    budget = SeedBudget()
    for relative in seed_paths(agent):
        source = host_home / relative
        if source.is_symlink() or not source.exists():
            continue
        destination = stage_home / relative
        if source.is_dir():
            _stage_directory_seed(
                source=source,
                destination=destination,
                relative=relative,
                budget=budget,
            )
        else:
            budget.copy(source, destination, relative)

    if agent == "codex":
        config = stage_home / ".codex" / "config.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(_minimal_codex_config(host_home / ".codex" / "config.toml"))
        config.chmod(0o600)
    return budget


def strip_session_args(args: list[str]) -> list[str]:
    """Drop ``--session``; Headless rejects it together with ``--docker``.

    Every Shinka proposal already runs in its own worktree, so there is no
    session state to resume.
    """
    stripped: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--session":
            index += 2
            continue
        if argument.startswith("--session="):
            index += 1
            continue
        stripped.append(argument)
        index += 1
    return stripped


def docker_flags() -> list[str]:
    flags: list[str] = ["--docker"]
    image = os.getenv(IMAGE_ENV, "").strip()
    if image:
        flags.extend(["--docker-image", image])
    platform = os.getenv(PLATFORM_ENV, "").strip()
    if platform:
        flags.extend(["--docker-arg", f"--platform={platform}"])
    for extra in shlex.split(os.getenv(EXTRA_ARGS_ENV, "")):
        flags.extend(["--docker-arg", extra])
    return flags


def build_command(*, agent: str, args: list[str]) -> list[str]:
    return [*base_command(), agent, *docker_flags(), *strip_session_args(args)]


def child_env(*, host_home: Path, stage_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(stage_home)
    # Keep npm/npx pointed at the real cache. Otherwise every proposal
    # re-downloads the Headless package (~50 MB) into the staged home.
    env.setdefault("npm_config_cache", str(host_home / ".npm"))
    if shutil.which("docker") is None and Path(DOCKER_DESKTOP_BIN).is_dir():
        env["PATH"] = os.pathsep.join([DOCKER_DESKTOP_BIN, env.get("PATH", "")])
    return env


def _is_passthrough(args: list[str]) -> bool:
    if not args:
        return True
    if args[0] in PASSTHROUGH_SUBCOMMANDS:
        return True
    return any(argument in PASSTHROUGH_FLAGS for argument in args)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if _is_passthrough(args):
        return subprocess.call([*base_command(), *args])

    agent = args[0]
    if agent not in AGENT_SEED_PATHS:
        # Still containerize. Falling back to a plain invocation would silently
        # run the agent on the host, which is the opposite of what the caller
        # asked for; an unauthenticated container fails loudly instead.
        print(
            f"headless-docker: no auth seed paths known for agent {agent!r}; "
            f"set {EXTRA_SEED_ENV} if it needs credentials",
            file=sys.stderr,
        )

    host_home = Path(os.path.expanduser("~"))
    stage_home = Path(tempfile.mkdtemp(prefix="shinka-headless-home."))
    try:
        budget = stage_auth_home(
            agent=agent, host_home=host_home, stage_home=stage_home
        )
        for entry in budget.skipped:
            print(f"headless-docker: skipped auth seed {entry}", file=sys.stderr)
        return subprocess.call(
            build_command(agent=agent, args=args[1:]),
            env=child_env(host_home=host_home, stage_home=stage_home),
        )
    finally:
        shutil.rmtree(stage_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
