from __future__ import annotations

import json
import os
import shlex
import stat
import sys
import asyncio
import subprocess
from pathlib import Path

import pytest

from shinka.cli import run as cli_run
from shinka.llm.client import get_async_client_llm, get_client_llm
from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.headless import (
    parse_headless_model,
    query_headless,
    query_headless_async,
)
from shinka.llm.providers import LLMAuthenticationError, LLMTimeoutError
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.model_availability import validate_model_env_access


def _make_fake_headless(tmp_path: Path) -> Path:
    script = tmp_path / "fake_headless.py"
    script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "from pathlib import Path",
                "",
                "if '--check' in sys.argv:",
                "    raise SystemExit(0)",
                "",
                "prompt_path = Path(sys.argv[sys.argv.index('--prompt-file') + 1])",
                "work_dir = Path(sys.argv[sys.argv.index('--work-dir') + 1])",
                "allow_mode = sys.argv[sys.argv.index('--allow') + 1]",
                "timeout_value = sys.argv[sys.argv.index('--timeout') + 1]",
                "assert prompt_path.exists(), prompt_path",
                "assert prompt_path.parent == work_dir / '.shinka', prompt_path",
                "assert work_dir.exists(), work_dir",
                "assert allow_mode == 'yolo', allow_mode",
                "assert timeout_value == '10', timeout_value",
                "assert '--json' in sys.argv",
                "assert '--usage' not in sys.argv",
                "if not (work_dir / '.git').exists():",
                "    (work_dir / 'generated.txt').write_text('mutated by headless\\n')",
                "src_file = work_dir / 'src' / 'app.py'",
                "if src_file.exists():",
                "    src_file.write_text('VALUE = 2\\n')",
                "    summary = '''# Individual Summary",
                "",
                "- Schema-Version: repo-individual-v1",
                "- Individual: fake",
                "- Generation: 1",
                "- Commit: pending",
                "",
                "## Parent",
                "",
                "Fake parent.",
                "",
                "## Core Idea",
                "",
                "Change VALUE to improve the fake score.",
                "",
                "## Lineage Context",
                "",
                "Fake lineage.",
                "",
                "## Changed Files",
                "",
                "- src/app.py",
                "",
                "## Validation Performed",
                "",
                "Fake validation.",
                "",
                "## Performance Hypothesis",
                "",
                "VALUE = 2 should score one.",
                "",
                "## Risks and Followups",
                "",
                "- None.",
                "",
                "## Minimal Snippets",
                "",
                "- VALUE = 2",
                "'''",
                "    (work_dir / '.shinka' / 'individual.md').write_text(summary + '\\n')",
                "print('fake headless completed')",
            ]
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _fake_headless_command(script: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _make_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "headless_task"
    task_dir.mkdir()
    seed_repo = task_dir / "seed_repo"
    seed_repo.mkdir()
    subprocess.run(["git", "init"], cwd=seed_repo, check=True, capture_output=True)
    (seed_repo / "src").mkdir()
    (seed_repo / "src" / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=seed_repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "seed",
        ],
        cwd=seed_repo,
        check=True,
        capture_output=True,
    )
    (task_dir / "evaluate.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import argparse",
                "import json",
                "from pathlib import Path",
                "",
                "def main(repo_path: str, results_dir: str):",
                "    value = int((Path(repo_path) / 'src' / 'app.py').read_text().split('=')[1])",
                "    score = 1.0 if value == 2 else 0.0",
                "    Path(results_dir).mkdir(parents=True, exist_ok=True)",
                "    Path(results_dir, 'metrics.json').write_text(json.dumps({'combined_score': score, 'public': {'score': score}, 'private': {}}))",
                "    Path(results_dir, 'correct.json').write_text(json.dumps({'correct': score == 1.0, 'error': ''}))",
                "",
                "if __name__ == '__main__':",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--repo_path', required=True)",
                "    parser.add_argument('--results_dir', required=True)",
                "    args = parser.parse_args()",
                "    main(args.repo_path, args.results_dir)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return task_dir


def test_parse_headless_model_with_model_and_effort():
    parsed = parse_headless_model("headless/opencode@openai/gpt-5.4?effort=high")

    assert parsed.agent == "opencode"
    assert parsed.agent_model == "openai/gpt-5.4"
    assert parsed.effort == "high"


def test_resolve_headless_model_backend():
    resolved = resolve_model_backend("headless/codex@gpt-5.5?effort=high")

    assert resolved.provider == "headless"
    assert resolved.api_model_name == "headless/codex@gpt-5.5?effort=high"
    assert resolved.base_url is None


def test_get_client_allows_headless_without_api_client():
    client, model_name, provider = get_client_llm("headless/codex")
    async_client, async_model_name, async_provider = get_async_client_llm(
        "headless/codex"
    )

    assert client is None
    assert model_name == "headless/codex"
    assert provider == "headless"
    assert async_client is None
    assert async_model_name == "headless/codex"
    assert async_provider == "headless"


def test_headless_kwargs_skip_api_only_parameters():
    kwargs = sample_model_kwargs(
        model_names=["headless/codex@gpt-5.5?effort=high"],
        temperatures=[0.0, 1.0],
        max_tokens=[128],
        reasoning_efforts=["high"],
    )

    assert kwargs == {"model_name": "headless/codex@gpt-5.5?effort=high"}


def test_query_headless_invokes_command_and_mutates_worktree(tmp_path, monkeypatch):
    fake_headless = _make_fake_headless(tmp_path)
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(fake_headless))
    monkeypatch.setenv("SHINKA_HEADLESS_TIMEOUT", "10")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    result = query_headless(
        None,
        "headless/codex@test-model?effort=low",
        "user request",
        "system instructions",
        [],
        output_model=None,
        headless_work_dir=str(work_dir),
    )

    assert "Headless agent completed" in result.content
    assert result.model_name == "headless/codex@test-model?effort=low"
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert (work_dir / "generated.txt").read_text(encoding="utf-8") == "mutated by headless\n"
    assert result.kwargs["headless_work_dir"] == str(work_dir)
    assert Path(result.kwargs["headless_prompt_path"]).exists()
    assert Path(result.kwargs["headless_stdout_path"]).exists()


def test_query_headless_invokes_claude_through_shell(tmp_path, monkeypatch):
    fake_headless = _make_fake_headless(tmp_path)
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(fake_headless))
    monkeypatch.setenv("SHINKA_HEADLESS_TIMEOUT", "10")

    result = query_headless(
        None,
        "headless/claude",
        "user request",
        "system instructions",
        [],
        output_model=None,
        headless_work_dir=str(tmp_path),
    )

    assert "Headless agent completed" in result.content
    assert result.model_name == "headless/claude"


def test_query_headless_parses_appended_usage(tmp_path, monkeypatch):
    script = tmp_path / "stdout_headless.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "if '--check' in sys.argv:",
                "    raise SystemExit(0)",
                "work_dir = Path(sys.argv[sys.argv.index('--work-dir') + 1])",
                "Path(sys.argv[sys.argv.index('--prompt-file') + 1]).exists() or sys.exit(2)",
                "(work_dir / 'generated.txt').write_text('mutated')",
                "usage = {",
                "    'agent': 'codex',",
                "    'provider': 'openai',",
                "    'model': 'gpt-5',",
                "    'inputTokens': 1,",
                "    'cacheReadTokens': 2,",
                "    'cacheWriteTokens': 3,",
                "    'outputTokens': 4,",
                "    'reasoningOutputTokens': 5,",
                "    'totalTokens': 15,",
                "    'cost': {",
                "        'input': 0.01,",
                "        'cacheRead': 0.02,",
                "        'cacheWrite': 0.03,",
                "        'output': 0.04,",
                "        'total': 0.10,",
                "    },",
                "    'pricingSource': 'models.dev',",
                "    'pricingStatus': 'priced',",
                "}",
                "print('final assistant message')",
                "print(json.dumps({'usage': usage}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(script))

    result = query_headless(
        None,
        "headless/codex",
        "user request",
        "system instructions",
        [],
        output_model=None,
        headless_work_dir=str(tmp_path),
    )

    assert result.cost == pytest.approx(0.10)
    assert result.input_cost == pytest.approx(0.06)
    assert result.output_cost == pytest.approx(0.04)
    assert result.input_tokens == 6
    assert result.output_tokens == 4
    assert result.thinking_tokens == 5
    assert result.kwargs["headless_usage_unknown"] is False
    assert result.kwargs["headless_usage"]["totalTokens"] == 15
    stdout_path = Path(result.kwargs["headless_stdout_path"])
    assert '"usage"' in stdout_path.read_text(encoding="utf-8")


def test_query_headless_reuses_named_session_in_json_mode(tmp_path, monkeypatch):
    script = tmp_path / "session_headless.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "work_dir = Path(sys.argv[sys.argv.index('--work-dir') + 1])",
                "session = sys.argv[sys.argv.index('--session') + 1]",
                "assert '--json' in sys.argv and '--usage' not in sys.argv",
                "with (work_dir / 'sessions.txt').open('a') as handle:",
                "    handle.write(session + '\\n')",
                "print('{}')",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(script))

    for _ in range(2):
        query_headless(
            None,
            "headless/cursor@test",
            "request",
            "system",
            [],
            output_model=None,
            headless_work_dir=str(tmp_path),
            headless_session_name="proposal-session",
            headless_timeout_seconds=10,
            headless_cleanup_grace_seconds=0.1,
        )

    assert (tmp_path / "sessions.txt").read_text().splitlines() == [
        "proposal-session",
        "proposal-session",
    ]


def test_headless_authentication_failure_is_typed(tmp_path, monkeypatch):
    script = tmp_path / "auth_headless.py"
    script.write_text(
        "import sys\nprint('Authentication required. Run agent login', file=sys.stderr)\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(script))

    with pytest.raises(LLMAuthenticationError):
        query_headless(
            None,
            "headless/cursor@test",
            "request",
            "system",
            [],
            output_model=None,
            headless_work_dir=str(tmp_path),
            headless_timeout_seconds=10,
            headless_cleanup_grace_seconds=0.1,
        )


def test_headless_timeout_kills_only_owned_process_group(tmp_path, monkeypatch):
    script = tmp_path / "timeout_headless.py"
    script.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import time",
                "from pathlib import Path",
                "work_dir = Path(sys.argv[sys.argv.index('--work-dir') + 1])",
                "child = subprocess.Popen(['sleep', '30'])",
                "(work_dir / 'child.pid').write_text(str(child.pid))",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(script))
    unrelated = subprocess.Popen(["sleep", "30"])
    try:
        with pytest.raises(LLMTimeoutError):
            query_headless(
                None,
                "headless/cursor@test",
                "request",
                "system",
                [],
                output_model=None,
                headless_work_dir=str(tmp_path),
                headless_timeout_seconds=0.1,
                headless_cleanup_grace_seconds=0.1,
            )

        child_pid = int((tmp_path / "child.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_query_headless_serializes_claude_async_calls(tmp_path, monkeypatch):
    active = 0
    max_active = 0

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            nonlocal active
            await asyncio.sleep(0.01)
            active -= 1
            return (
                b"content\n"
                b'{"usage":{"inputTokens":1,"outputTokens":1,"cost":{"total":0}}}',
                b"",
            )

        def kill(self):
            raise AssertionError("fake process should not time out")

    async def fake_create_subprocess_shell(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        return FakeProcess()

    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", "headless")
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    async def run_queries():
        await asyncio.gather(
            query_headless_async(
                None,
                "headless/claude",
                "user request",
                "system instructions",
                [],
                output_model=None,
                headless_work_dir=str(tmp_path),
            ),
            query_headless_async(
                None,
                "headless/claude",
                "user request",
                "system instructions",
                [],
                output_model=None,
                headless_work_dir=str(tmp_path),
            ),
        )

    asyncio.run(run_queries())

    assert max_active == 1


def test_validate_model_env_access_runs_headless_check(tmp_path, monkeypatch):
    fake_headless = _make_fake_headless(tmp_path)
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(fake_headless))
    monkeypatch.setenv("SHINKA_HEADLESS_TIMEOUT", "10")

    validate_model_env_access(llm_models=["headless/codex"])


@pytest.mark.integration
def test_shinka_run_full_headless_cli_mutation_succeeds(tmp_path, monkeypatch):
    fake_headless = _make_fake_headless(tmp_path)
    task_dir = _make_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(fake_headless))
    monkeypatch.setenv("SHINKA_HEADLESS_TIMEOUT", "10")

    exit_code = cli_run.main(
        [
            "--task-dir",
            str(task_dir),
            "--results_dir",
            str(results_dir),
            "--num_generations",
            "2",
            "--max-evaluation-jobs",
            "1",
            "--max-proposal-jobs",
            "1",
            "--max-db-workers",
            "1",
            "--no-verbose",
            "--set",
            'evo.llm_models=["headless/codex@test-model?effort=low"]',
            "--set",
            "evo.llm_dynamic_selection=null",
            "--set",
            "evo.embedding_model=null",
            "--set",
            'evo.mutable_paths=["src"]',
            "--set",
            'evo.patch_types=["full"]',
            "--set",
            "evo.patch_type_probs=[1.0]",
            "--set",
            "evo.max_patch_resamples=1",
            "--set",
            "evo.max_novelty_attempts=1",
            "--set",
            "evo.max_patch_attempts=1",
            "--set",
            "evo.headless_proposal_timeout_seconds=10",
            "--set",
            "evo.generation_target_mode=proposal_ids",
            "--set",
            "db.num_islands=1",
            "--set",
            "db.archive_size=4",
        ]
    )

    assert exit_code == 0
    attempt_prompts = list(results_dir.glob("gen_1/attempts/**/headless_prompt.md"))
    assert attempt_prompts, sorted(str(path) for path in results_dir.rglob("*"))
    prompt_text = attempt_prompts[0].read_text(encoding="utf-8")
    assert "Repository Mode Contract" in prompt_text
    assert ".shinka/individual.md" in prompt_text

    metrics_files = list(results_dir.glob("gen_1/**/metrics.json"))
    assert metrics_files
    best_score = max(
        json.loads(path.read_text(encoding="utf-8"))["combined_score"]
        for path in metrics_files
    )
    assert best_score == pytest.approx(1.0)
