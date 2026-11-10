"""Regression coverage for permanent local-endpoint failures (issue #196)."""

import asyncio
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from rich.console import Console

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.llm.llm import AsyncLLMClient, LLMClient
from shinka.llm.providers.errors import NonRetryableLLMError


@pytest.mark.parametrize("status", [401, 403, 404])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_permanent_endpoint_error_crosses_outer_retry_without_retries(
    status, asynchronous
):
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {"message": "DeploymentNotFound", "code": "DeploymentNotFound"}
            },
        )

    model = "local/fake-model@http://127.0.0.1:18404/v1"
    transport = httpx.MockTransport(reply)

    async def run():
        async with openai.AsyncOpenAI(
            api_key="local",
            base_url="http://127.0.0.1:18404/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            with patch(
                "shinka.llm.query.get_async_client_llm",
                return_value=(client, "fake-model", "local_openai"),
            ):
                await AsyncLLMClient(model_names=model, verbose=False).query(
                    "msg", "sys", llm_kwargs={"model_name": model}
                )

    def invoke():
        if asynchronous:
            asyncio.run(run())
        else:
            with openai.OpenAI(
                api_key="local",
                base_url="http://127.0.0.1:18404/v1",
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            ) as client:
                with patch(
                    "shinka.llm.query.get_client_llm",
                    return_value=(client, "fake-model", "local_openai"),
                ):
                    LLMClient(model_names=model, verbose=False).query(
                        "msg", "sys", llm_kwargs={"model_name": model}
                    )

    with pytest.raises(NonRetryableLLMError, match=str(status)):
        invoke()
    assert len(calls) == 1


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize(
    "generations,programs,failed",
    [
        (2, [Program(id="seed", code="seed", generation=0, correct=True)], True),
        (1, [Program(id="seed", code="seed", generation=0, correct=True)], False),
        (
            2,
            [
                Program(
                    id="failed",
                    code="",
                    generation=1,
                    correct=False,
                    metadata={"node_kind": "failed_proposal"},
                )
            ],
            True,
        ),
        (
            3,
            [Program(id="candidate", code="candidate", generation=1, correct=False)],
            False,
        ),
        (
            2,
            [
                Program(
                    id="copy",
                    code="seed",
                    generation=1,
                    correct=True,
                    metadata={"_is_island_copy": True},
                )
            ],
            True,
        ),
    ],
)
def test_summary_marks_no_evolved_program_as_failure(
    verbose, generations, programs, failed, monkeypatch
):
    runner: Any = ShinkaEvolveRunner.__new__(ShinkaEvolveRunner)
    runner.verbose = verbose
    runner.db = SimpleNamespace(
        get_all_programs=lambda: programs, print_summary=lambda **kwargs: None
    )
    runner.evo_config = SimpleNamespace(num_generations=generations, max_api_costs=None)
    runner.start_time = None
    runner.completed_generations = len(programs)
    runner.total_proposals_generated = 0
    runner.total_api_cost = 0
    runner.embedding_client = None
    runner.meta_summarizer = None
    runner.console = Console(file=StringIO())
    monkeypatch.setattr(
        runner, "_get_generations_without_program_due_to_proposal_failure", lambda: []
    )
    monkeypatch.setattr(runner, "_log_timing_bottleneck_summary", lambda: None)
    asyncio.run(runner._print_final_summary())
    assert runner.evolution_failed is failed


@pytest.mark.parametrize("copy_marker", ["_is_island_copy", "_spawned_island"])
def test_database_summary_separates_seed_copies_from_evolution(tmp_path, copy_marker):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1),
        embedding_model="",
    )
    try:
        db.add(Program(id="seed", code="seed", generation=0, correct=True))
        db.add(
            Program(
                id="copy",
                code="seed",
                generation=0,
                correct=True,
                metadata={copy_marker: True},
            )
        )
        output = StringIO()
        db.print_summary(
            console=Console(file=output, width=120, color_system=None),
            total_program_target=2,
        )
        text = output.getvalue()
        assert "1 / 2" in text
        assert "Correct DB Rows" in text
        assert "Island Copies" in text
        assert "Evolved Programs" in text
        assert "Correct Evolved" in text
    finally:
        db.close()


@pytest.mark.parametrize("status", [429, 500, 503])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_transient_endpoint_status_still_retries(status, asynchronous):
    from shinka.llm.providers.local_openai import (
        query_local_openai,
        query_local_openai_async,
    )

    calls = []

    def reply(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, json={"error": {"message": "temporary"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(reply)
    if asynchronous:

        async def run():
            async with openai.AsyncOpenAI(
                api_key="local",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=transport),
            ) as client:
                return await query_local_openai_async(
                    client, "fake-model", "msg", "sys", [], None
                )

        result = asyncio.run(run())
    else:
        with openai.OpenAI(
            api_key="local",
            max_retries=0,
            http_client=httpx.Client(transport=transport),
        ) as client:
            result = query_local_openai(client, "fake-model", "msg", "sys", [], None)
    assert result.content == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize("failed,expected", [(False, 0), (True, 1)])
def test_cli_returns_evolution_outcome(tmp_path, monkeypatch, failed, expected):
    import shinka.cli.run as cli

    class Runner:
        evolution_failed = failed

        def __init__(self, **kwargs):
            pass

        def run(self):
            pass

    monkeypatch.setattr(cli, "ShinkaEvolveRunner", Runner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "initial.py").write_text(
        "# EVOLVE-BLOCK-START\nx=1\n# EVOLVE-BLOCK-END\n", encoding="utf-8"
    )
    (task / "evaluate.py").write_text("def main(): pass\n", encoding="utf-8")
    assert (
        cli.main(
            [
                "--task-dir",
                str(task),
                "--results_dir",
                str(tmp_path / "results"),
                "--num_generations",
                "2",
            ]
        )
        == expected
    )


@pytest.mark.parametrize("status", [401, 403, 404])
def test_endpoint_error_survives_multiprocessing_serialization(status):
    import pickle
    from shinka.llm.providers.errors import LLMEndpointError

    error = LLMEndpointError(status, "DeploymentNotFound")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, NonRetryableLLMError)
    assert restored.status_code == status
    assert restored.code == "DeploymentNotFound"
    assert str(restored) == str(error)
