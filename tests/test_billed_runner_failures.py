"""Output exhaustion stops all proposal retry layers and retains its bill."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import EmptyResponseError


@pytest.mark.parametrize("fix_mode", [False, True])
@pytest.mark.parametrize("patch_attempts", [1, 2])
@pytest.mark.parametrize("save_fails", [False, True])
def test_output_exhaustion_stops_proposal_retries(
    monkeypatch, fix_mode, patch_attempts, save_fails
):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.evo_config = EvolutionConfig(
        max_patch_attempts=patch_attempts, max_api_costs=5
    )
    runner.db_config = DatabaseConfig()
    runner.llm_selection = None
    runner.verbose = False
    runner.total_api_cost = 0.0
    parent = Program(id="parent", code="pass", generation=0, correct=not fix_mode)

    async def sample(**kwargs):
        return parent, [], [], fix_mode

    runner.async_db = SimpleNamespace(sample_with_fix_mode_async=sample)
    runner.prompt_sampler = SimpleNamespace(
        task_sys_msg="task",
        sample=lambda **kwargs: ("sys", "patch", "diff"),
        sample_fix=lambda **kwargs: ("sys", "patch", "diff"),
    )
    runner._get_current_system_prompt = lambda: (None, None)

    async def save(**kwargs):
        if save_fails:
            raise OSError("attempt log unavailable")

    failures = []

    async def record(**kwargs):
        failures.append(kwargs)
        runner.total_api_cost += kwargs["api_costs"]

    runner._save_patch_attempt_async = save
    runner._record_terminal_failed_proposal = record
    runner.llm = AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
    requests = []

    async def exhausted(**kwargs):
        requests.append(kwargs)
        raise EmptyResponseError(
            "no text",
            "max_output_tokens",
            {"cost": 3.55, "input_tokens": 1000, "thinking_tokens": 65536},
        )

    monkeypatch.setattr(llm_module, "query_async", exhausted)
    asyncio.run(
        runner._generate_evolved_proposal(
            1, "task", "unused", "unused", None, None, None, 0, None, 1
        )
    )
    assert len(requests) == 1
    assert runner.total_api_cost == pytest.approx(3.55)
    assert len(failures) == 1
    assert failures[0]["meta_patch_data"]["non_retryable"]
    assert failures[0]["meta_patch_data"]["patch_attempt"] == 1
    assert "max_output_tokens" in failures[0]["failure_reason"]
