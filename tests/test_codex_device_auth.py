import subprocess
from pathlib import Path

import pytest

from shinka.tools.codex_device_auth import CodexAuthError, ensure_codex_authenticated


def test_ensure_codex_authenticated_noop_when_logged_in(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, stdout="Logged in", stderr="")
        raise AssertionError(f"Unexpected call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    method = ensure_codex_authenticated(Path("/bin/codex"), allow_interactive=False)
    assert method == "status"
    assert [args for args, _ in calls] == [[str(Path("/bin/codex")), "login", "status"]]


def test_ensure_codex_authenticated_uses_api_key_login(monkeypatch):
    calls = []
    status_calls = {"count": 0}

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[1:] == ["login", "status"]:
            status_calls["count"] += 1
            if status_calls["count"] == 1:
                return subprocess.CompletedProcess(
                    args, 1, stdout="", stderr="Not logged in"
                )
            return subprocess.CompletedProcess(args, 0, stdout="Logged in", stderr="")

        if args[1:] == ["login", "--with-api-key"]:
            assert kwargs.get("input", "").startswith("sk-test")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    method = ensure_codex_authenticated(
        Path("/bin/codex"),
        api_key="sk-test",
        allow_interactive=False,
    )
    assert method == "api_key"

    called = [a for a, _ in calls]
    assert called[0][1:] == ["login", "status"]
    assert called[1][1:] == ["login", "--with-api-key"]
    assert called[2][1:] == ["login", "status"]


def test_ensure_codex_authenticated_raises_when_noninteractive(monkeypatch):
    def fake_run(args, **kwargs):
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(
                args, 1, stdout="", stderr="Not logged in"
            )
        raise AssertionError(f"Unexpected call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexAuthError):
        ensure_codex_authenticated(Path("/bin/codex"), allow_interactive=False)
