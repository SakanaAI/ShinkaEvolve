from __future__ import annotations

import json
import shlex
import sqlite3
import stat
import sys
import asyncio
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
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.model_availability import validate_model_env_access


def _make_fake_headless(tmp_path: Path, *, vary_outputs: bool = False) -> Path:
    script = tmp_path / "fake_headless.py"
    score_lines = (
        [
            "active_generations = [int(path.parent.name.removeprefix('gen_')) "
            "for path in work_dir.glob('gen_*/.generation_lock')]",
            "generation = max(active_generations, default=1)",
            "model = sys.argv[sys.argv.index('--model') + 1] "
            "if '--model' in sys.argv else 'default'",
            "print(f'# selected-model: {model}')",
            "print('def score():')",
            "print(f'    return {float(generation)!r}')",
        ]
        if vary_outputs
        else [
            "print('def score():')",
            "print('    return 1.0')",
        ]
    )
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
                "assert prompt_path.exists(), prompt_path",
                "assert work_dir.exists(), work_dir",
                "print('<NAME>')",
                "print('raise_score')",
                "print('</NAME>')",
                "print('<DESCRIPTION>')",
                "print('Deterministic fake headless mutation.')",
                "print('</DESCRIPTION>')",
                "print('<CODE>')",
                "print('```python')",
                "print('# EVOLVE-BLOCK-START')",
                *score_lines,
                "print('# EVOLVE-BLOCK-END')",
                "print('```')",
                "print('</CODE>')",
                "print(json.dumps({'usage': {'input_tokens': 11, 'output_tokens': 13, 'thinking_tokens': 0, 'cost': 0.0}}))",
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
    (task_dir / "initial.py").write_text(
        "\n".join(
            [
                "# EVOLVE-BLOCK-START",
                "def score():",
                "    return 0.0",
                "# EVOLVE-BLOCK-END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "evaluate.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import argparse",
                "import importlib.util",
                "import json",
                "from pathlib import Path",
                "",
                "def _load(path):",
                "    spec = importlib.util.spec_from_file_location('program', path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    spec.loader.exec_module(module)",
                "    return module",
                "",
                "def main(program_path: str, results_dir: str):",
                "    score = float(_load(program_path).score())",
                "    Path(results_dir).mkdir(parents=True, exist_ok=True)",
                "    Path(results_dir, 'metrics.json').write_text(json.dumps({'combined_score': score, 'public': {'score': score}, 'private': {}}))",
                "    Path(results_dir, 'correct.json').write_text(json.dumps({'correct': True, 'error': ''}))",
                "",
                "if __name__ == '__main__':",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--program_path', required=True)",
                "    parser.add_argument('--results_dir', required=True)",
                "    args = parser.parse_args()",
                "    main(args.program_path, args.results_dir)",
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


def test_query_headless_invokes_command_and_parses_usage(tmp_path, monkeypatch):
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

    assert "raise_score" in result.content
    assert result.model_name == "headless/codex@test-model?effort=low"
    assert result.input_tokens == 11
    assert result.output_tokens == 13
    assert Path(result.kwargs["headless_prompt_path"]).exists()


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

    assert "raise_score" in result.content
    assert result.model_name == "headless/claude"


def test_query_headless_accepts_nested_cost_usage(tmp_path, monkeypatch):
    script = tmp_path / "nested_cost_headless.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "if '--check' in sys.argv:",
                "    raise SystemExit(0)",
                "Path(sys.argv[sys.argv.index('--prompt-file') + 1]).exists() or sys.exit(2)",
                "print('content')",
                "print(json.dumps({'usage': {'inputTokens': 1, 'outputTokens': 2, 'reasoningOutputTokens': 3, 'cost': {'input': 0.01, 'output': 0.02, 'total': 0.03}}}))",
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

    assert result.cost == pytest.approx(0.03)
    assert result.input_cost == pytest.approx(0.01)
    assert result.output_cost == pytest.approx(0.02)
    assert result.input_tokens == 1
    assert result.output_tokens == 2
    assert result.thinking_tokens == 3


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
            "db.num_islands=1",
            "--set",
            "db.archive_size=4",
        ]
    )

    assert exit_code == 0
    attempt_prompts = list(results_dir.glob("gen_1/attempts/**/headless_prompt.md"))
    assert attempt_prompts, sorted(str(path) for path in results_dir.rglob("*"))
    assert "Current program" in attempt_prompts[0].read_text(encoding="utf-8")

    metrics_files = list(results_dir.glob("gen_1/**/metrics.json"))
    assert metrics_files
    best_score = max(
        json.loads(path.read_text(encoding="utf-8"))["combined_score"]
        for path in metrics_files
    )
    assert best_score == pytest.approx(1.0)


def _checkpoint_trace(results_dir: Path) -> list[dict[str, object]]:
    with sqlite3.connect(results_dir / "programs.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, generation, code, parent_id, archive_inspiration_ids,
                   top_k_inspiration_ids, combined_score, correct, island_idx,
                   metadata
            FROM programs
            ORDER BY generation, id
            """
        ).fetchall()

    generation_by_id = {row["id"]: row["generation"] for row in rows}

    def generations(raw_ids: str) -> list[int]:
        return sorted(generation_by_id[item] for item in json.loads(raw_ids))

    return [
        {
            "generation": row["generation"],
            "code": row["code"],
            "parent_generation": generation_by_id.get(row["parent_id"]),
            "archive_generations": generations(row["archive_inspiration_ids"]),
            "top_k_generations": generations(row["top_k_inspiration_ids"]),
            "score": row["combined_score"],
            "correct": row["correct"],
            "island": row["island_idx"],
            "model": json.loads(row["metadata"] or "{}").get("model_name"),
        }
        for row in rows
    ]


@pytest.mark.integration
def test_seeded_checkpoint_resume_matches_uninterrupted_evolution(
    tmp_path, monkeypatch
):
    fake_headless = _make_fake_headless(tmp_path, vary_outputs=True)
    task_dir = _make_task_dir(tmp_path)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", _fake_headless_command(fake_headless))
    monkeypatch.setenv("SHINKA_HEADLESS_TIMEOUT", "10")

    original_build_runner = cli_run._build_runner

    def build_runner_with_checkpoint(**kwargs):
        runner = original_build_runner(**kwargs)
        checkpoint_path = Path(runner.results_dir) / "checkpoint.pkl"
        if Path(runner.results_dir) == resumed_dir and not checkpoint_path.exists():
            original_update = runner._update_completed_generations

            async def update_and_request_checkpoint():
                await original_update()
                if (
                    runner.completed_generations >= 4
                    and not runner.checkpoint_requested.is_set()
                ):
                    runner.request_checkpoint_and_exit()
                    await asyncio.sleep(0)

            runner._update_completed_generations = update_and_request_checkpoint
        return runner

    monkeypatch.setattr(cli_run, "_build_runner", build_runner_with_checkpoint)

    def run(results_dir: Path) -> int:
        return cli_run.main(
            [
                "--task-dir",
                str(task_dir),
                "--results_dir",
                str(results_dir),
                "--num_generations",
                "8",
                "--random-seed",
                "20260830",
                "--checkpoint-resume-mode",
                "strict",
                "--max-evaluation-jobs",
                "1",
                "--max-proposal-jobs",
                "1",
                "--max-db-workers",
                "1",
                "--no-verbose",
                "--set",
                'evo.llm_models=["headless/codex@model-a","headless/codex@model-b"]',
                "--set",
                "evo.llm_dynamic_selection=fixed",
                "--set",
                "evo.embedding_model=null",
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
                "db.num_islands=1",
                "--set",
                "db.archive_size=4",
            ]
        )

    assert run(uninterrupted_dir) == 0
    assert run(resumed_dir) == 0
    assert (resumed_dir / "checkpoint.pkl").is_file()
    assert len(_checkpoint_trace(resumed_dir)) == 4

    assert run(resumed_dir) == 0
    resume_audit = json.loads(
        (resumed_dir / "checkpoint_resume.json").read_text(encoding="utf-8")
    )
    assert resume_audit["deterministic_resume"] is True
    assert _checkpoint_trace(resumed_dir) == _checkpoint_trace(uninterrupted_dir)
