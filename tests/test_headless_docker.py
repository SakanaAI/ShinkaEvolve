from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from shinka.llm.providers import headless_docker as hd


def _staged_files(stage_home: Path) -> dict[str, int]:
    return {
        str(path.relative_to(stage_home)): path.stat().st_size
        for path in sorted(stage_home.rglob("*"))
        if path.is_file()
    }


def _make_antigravity_home(root: Path, *, bulk_files: int = 400) -> Path:
    home = root / "home"
    cli_dir = home / ".gemini" / "antigravity-cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "antigravity-oauth-token").write_text("token")
    (cli_dir / "settings.json").write_text('{"useG1Credits": true}')
    (cli_dir / "installation_id").write_text("abc123")
    (cli_dir / "jetski_state.pbtxt").write_text("post_onboarding: true")
    (cli_dir / "conversation_summaries.db").write_bytes(b"\0" * 400_000)
    for name in ("conversations", "brain", "log"):
        bulk = cli_dir / name
        bulk.mkdir()
        for index in range(bulk_files):
            (bulk / f"{index}.bin").write_bytes(b"\0" * 4096)
    return home


def test_directory_seed_never_copies_the_whole_tree(tmp_path):
    host_home = _make_antigravity_home(tmp_path)
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    hd.stage_auth_home(agent="antigravity", host_home=host_home, stage_home=stage_home)

    staged = _staged_files(stage_home)
    assert set(staged) == {
        ".gemini/antigravity-cli/antigravity-oauth-token",
        ".gemini/antigravity-cli/settings.json",
        ".gemini/antigravity-cli/installation_id",
        ".gemini/antigravity-cli/jetski_state.pbtxt",
    }
    assert sum(staged.values()) < 64 * 1024


def test_directory_seed_without_allowlist_keeps_only_top_level_files(tmp_path):
    host_home = tmp_path / "home"
    config_dir = host_home / ".config" / "opencode"
    (config_dir / "cache").mkdir(parents=True)
    (config_dir / "auth.json").write_text("{}")
    (config_dir / "cache" / "blob.bin").write_bytes(b"\0" * 200_000)
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    hd.stage_auth_home(agent="opencode", host_home=host_home, stage_home=stage_home)

    assert set(_staged_files(stage_home)) == {".config/opencode/auth.json"}


def test_symlinked_seed_paths_are_skipped(tmp_path):
    host_home = tmp_path / "home"
    (host_home / ".cursor").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("secret")
    (host_home / ".cursor" / "cli-config.json").symlink_to(outside)
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    hd.stage_auth_home(agent="cursor", host_home=host_home, stage_home=stage_home)

    assert _staged_files(stage_home) == {}


def test_oversized_seed_files_are_skipped_and_reported(tmp_path):
    host_home = tmp_path / "home"
    (host_home / ".claude").mkdir(parents=True)
    (host_home / ".claude.json").write_bytes(b"\0" * (hd.MAX_SEED_FILE_BYTES + 1))
    (host_home / ".claude" / "auth.json").write_text("{}")
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    budget = hd.stage_auth_home(
        agent="claude", host_home=host_home, stage_home=stage_home
    )

    assert set(_staged_files(stage_home)) == {".claude/auth.json"}
    assert any("per-file limit" in entry for entry in budget.skipped)
    assert any(entry.startswith(".claude.json") for entry in budget.skipped)


def test_total_seed_budget_is_enforced(tmp_path):
    host_home = tmp_path / "home"
    gemini_dir = host_home / ".gemini"
    gemini_dir.mkdir(parents=True)
    for name in hd.AGENT_SEED_PATHS["gemini"]:
        (host_home / name).write_bytes(b"\0" * (hd.MAX_SEED_FILE_BYTES - 1))
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    budget = hd.stage_auth_home(
        agent="gemini", host_home=host_home, stage_home=stage_home
    )

    assert sum(_staged_files(stage_home).values()) <= hd.MAX_SEED_TOTAL_BYTES
    assert any("remaining budget" in entry for entry in budget.skipped)


def test_codex_config_keeps_model_keys_and_drops_host_sections(tmp_path):
    host_home = tmp_path / "home"
    (host_home / ".codex").mkdir(parents=True)
    (host_home / ".codex" / "auth.json").write_text("{}")
    (host_home / ".codex" / "config.toml").write_text(
        "\n".join(
            [
                'model = "gpt-5.6-luna"',
                'model_reasoning_effort = "xhigh"',
                'service_tier = "default"',
                "notify = [",
                '  "/Users/someone/notify.sh",',
                "]",
                "[desktop]",
                'theme = "dark"',
                "[mcp_servers.node_repl]",
                'command = "/Applications/ChatGPT.app/node_repl"',
            ]
        )
    )
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    hd.stage_auth_home(agent="codex", host_home=host_home, stage_home=stage_home)

    config = (stage_home / ".codex" / "config.toml").read_text()
    assert 'model = "gpt-5.6-luna"' in config
    assert 'model_reasoning_effort = "xhigh"' in config
    assert 'service_tier = "fast"' in config
    assert "mcp_servers" not in config
    assert "desktop" not in config
    assert "notify" not in config
    assert "/Users/someone" not in config


def test_codex_config_falls_back_to_defaults_without_host_config(tmp_path):
    host_home = tmp_path / "home"
    (host_home / ".codex").mkdir(parents=True)
    (host_home / ".codex" / "auth.json").write_text("{}")
    stage_home = tmp_path / "stage"
    stage_home.mkdir()

    hd.stage_auth_home(agent="codex", host_home=host_home, stage_home=stage_home)

    assert (stage_home / ".codex" / "config.toml").read_text() == (
        'service_tier = "fast"\n'
    )


def test_extra_seed_paths_can_be_added(tmp_path, monkeypatch):
    host_home = tmp_path / "home"
    (host_home / ".gemini").mkdir(parents=True)
    (host_home / ".gemini" / "config.json").write_text("{}")
    stage_home = tmp_path / "stage"
    stage_home.mkdir()
    monkeypatch.setenv(hd.EXTRA_SEED_ENV, ".gemini/config.json")

    hd.stage_auth_home(agent="cursor", host_home=host_home, stage_home=stage_home)

    assert set(_staged_files(stage_home)) == {".gemini/config.json"}


def test_session_args_are_stripped():
    assert hd.strip_session_args(
        ["--session", "gen1", "--model", "composer-2.5", "--session=gen2", "--json"]
    ) == ["--model", "composer-2.5", "--json"]


def test_build_command_requests_docker_and_honours_env(monkeypatch):
    monkeypatch.setenv(hd.IMAGE_ENV, "example/image:tag")
    monkeypatch.setenv(hd.PLATFORM_ENV, "linux/arm64")
    monkeypatch.setenv(hd.EXTRA_ARGS_ENV, "--memory=4g --cpus=2")

    command = hd.build_command(agent="codex", args=["--json"])

    assert command == [
        "npx",
        "-y",
        "@roberttlange/headless",
        "codex",
        "--docker",
        "--docker-image",
        "example/image:tag",
        "--docker-arg",
        "--platform=linux/arm64",
        "--docker-arg",
        "--memory=4g",
        "--docker-arg",
        "--cpus=2",
        "--json",
    ]


def test_child_env_stages_home_but_keeps_the_host_npm_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("npm_config_cache", raising=False)
    host_home = tmp_path / "home"
    stage_home = tmp_path / "stage"

    env = hd.child_env(host_home=host_home, stage_home=stage_home)

    assert env["HOME"] == str(stage_home)
    assert env["npm_config_cache"] == str(host_home / ".npm")


@pytest.mark.parametrize(
    "args",
    [[], ["--check"], ["codex", "--check"], ["docker", "doctor"], ["run", "list"]],
)
def test_informational_invocations_bypass_staging(args, monkeypatch):
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        hd.subprocess, "call", lambda cmd, **kwargs: recorded.append(cmd) or 0
    )

    assert hd.main(args) == 0
    assert recorded == [[*hd.base_command(), *args]]
    assert all("--docker" not in cmd for cmd in recorded)


def test_unknown_agents_still_run_in_docker(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        hd.subprocess, "call", lambda cmd, **kwargs: recorded.append(cmd) or 0
    )

    assert hd.main(["brand-new-agent", "--json"]) == 0

    assert "--docker" in recorded[0]
    assert "no auth seed paths known" in capsys.readouterr().err


def test_main_stages_a_minimal_home_and_cleans_it_up(tmp_path, monkeypatch):
    host_home = _make_antigravity_home(tmp_path, bulk_files=50)
    monkeypatch.setenv("HOME", str(host_home))
    observed: dict[str, object] = {}

    def fake_call(cmd, **kwargs):
        stage_home = Path(kwargs["env"]["HOME"])
        observed["command"] = cmd
        observed["stage_home"] = stage_home
        observed["staged"] = _staged_files(stage_home)
        return 0

    monkeypatch.setattr(hd.subprocess, "call", fake_call)

    assert hd.main(["antigravity", "--json"]) == 0

    assert "--docker" in observed["command"]
    assert sum(observed["staged"].values()) < 64 * 1024
    assert not Path(observed["stage_home"]).exists()


@pytest.mark.integration
def test_seed_paths_match_the_installed_headless_cli():
    if shutil.which("npx") is None:
        pytest.skip("npx is unavailable")
    for agent, expected in hd.AGENT_SEED_PATHS.items():
        completed = subprocess.run(
            ["npx", "-y", "@roberttlange/headless", agent, "--show-config"],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "HOME": os.path.expanduser("~")},
        )
        if completed.returncode != 0:
            pytest.skip(f"headless --show-config unavailable: {completed.stderr}")
        reported = tuple(
            line.split("|")[2].strip()
            for line in completed.stdout.splitlines()
            if "| Seed path" in line
        )
        assert reported == expected, agent
