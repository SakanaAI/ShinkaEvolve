from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from shinka.llm.providers.headless import parse_headless_model, query_headless


def _run(command: list[str], *, timeout: float = 120) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout.strip()


def _native_auth_check(agent: str, requested_model: str | None) -> str:
    commands = {
        "antigravity": ["agy", "models"],
        "cursor": ["agent", "models"],
        "codex": ["codex", "--version"],
    }
    command = commands.get(agent)
    if command is None:
        return "no native authentication check registered"
    output = _run(command)
    if requested_model and agent in {"antigravity", "cursor"}:
        normalized_requested = requested_model.lower().replace("-xhigh", "")
        normalized_output = output.lower()
        if normalized_requested not in normalized_output:
            raise RuntimeError(
                f"native {agent} model list does not include {requested_model!r}"
            )
    return output.splitlines()[0] if output else "ok"


def _initialize_repo(path: Path) -> None:
    _run(["git", "init", str(path)])
    (path / "README.md").write_text("headless canary\n", encoding="utf-8")
    _run(["git", "-C", str(path), "add", "README.md"])
    _run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Shinka Canary",
            "-c",
            "user.email=canary@example.invalid",
            "commit",
            "-m",
            "seed",
        ]
    )


def run_canary(
    model: str,
    *,
    timeout_seconds: float,
    cleanup_grace: float,
    artifacts_dir: Path,
) -> dict:
    parsed = parse_headless_model(model)
    native_check = _native_auth_check(parsed.agent, parsed.agent_model)
    session_name = f"shinka-canary-{uuid.uuid4().hex[:12]}"

    route_dir = artifacts_dir / f"{parsed.agent}-{uuid.uuid4().hex[:12]}"
    repo = route_dir / "repo"
    repo.mkdir(parents=True)
    _initialize_repo(repo)
    marker = repo / "canary.txt"

    try:
        query_headless(
            None,
            model,
            f"In the repository `{repo}`, create canary.txt containing exactly FIRST and finish the task.",
            "Edit the active repository directly. Do not only describe the change.",
            [],
            output_model=None,
            headless_work_dir=str(repo),
            headless_session_name=session_name,
            headless_timeout_seconds=timeout_seconds,
            headless_cleanup_grace_seconds=cleanup_grace,
            headless_output_mode="json",
        )
        if not marker.is_file() or "FIRST" not in marker.read_text(encoding="utf-8"):
            raise RuntimeError("first Headless call did not mutate the requested worktree")

        query_headless(
            None,
            model,
            f"Resume this task in `{repo}` and add a second line containing exactly SECOND to canary.txt.",
            "Continue editing the active repository directly.",
            [{"role": "assistant", "content": "Created canary.txt with FIRST."}],
            output_model=None,
            headless_work_dir=str(repo),
            headless_session_name=session_name,
            headless_timeout_seconds=timeout_seconds,
            headless_cleanup_grace_seconds=cleanup_grace,
            headless_output_mode="json",
        )
        marker_text = marker.read_text(encoding="utf-8")
        if "FIRST" not in marker_text or "SECOND" not in marker_text:
            raise RuntimeError("named Headless session did not resume and mutate worktree")

    except Exception as exc:
        raise RuntimeError(f"{exc}; canary artifacts retained at {route_dir}") from exc

    status = _run(["git", "-C", str(repo), "status", "--short"])
    return {
        "model": model,
        "agent": parsed.agent,
        "native_check": native_check,
        "session_name": session_name,
        "worktree_mutated": True,
        "session_resumed": True,
        "git_status": status.splitlines(),
        "artifacts_dir": str(route_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a real authenticated Headless workdir/session canary."
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=60)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("results/headless_canaries"),
    )
    args = parser.parse_args(argv)

    results = [
        run_canary(
            model,
            timeout_seconds=args.timeout_seconds,
            cleanup_grace=args.cleanup_grace_seconds,
            artifacts_dir=args.artifacts_dir.resolve(),
        )
        for model in args.model
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
