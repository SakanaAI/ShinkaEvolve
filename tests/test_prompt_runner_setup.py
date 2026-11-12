"""Runner prompt settings reach provider requests through the client sampler."""

import asyncio
import importlib
import json

import openai
from openai import _base_client
import pytest

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)
from shinka.llm.llm import AsyncLLMClient

httpx = getattr(_base_client, "httpx", None) or getattr(_base_client, "httpx2")
query_module = importlib.import_module("shinka.llm.query")
PROMPT_TEXT = "Improve packing while preserving valid geometry, the program API, and immutable regions."


@pytest.fixture
def runner(tmp_path):
    instance = object.__new__(ShinkaEvolveRunner)
    instance.results_dir = str(tmp_path)
    instance.verbose = False
    instance.prompt_db = None
    instance.evo_config = EvolutionConfig(
        evolve_prompts=True,
        task_sys_msg=PROMPT_TEXT,
        prompt_llm_models=["gpt-5-mini"],
    )
    yield instance
    if instance.prompt_db is not None:
        instance.prompt_db.close()


@pytest.mark.parametrize("patch_type", ["diff", "full"])
@pytest.mark.parametrize(
    "settings,expected_tokens",
    [
        (
            {"temperatures": [0.0], "reasoning_efforts": ["low"], "max_tokens": 8192},
            8192,
        ),
        ({"max_tokens": 2048}, 2048),
        ({}, 4096),
    ],
    ids=["custom-sampler-settings", "token-limit-only", "defaults"],
)
def test_prompt_settings_serialize_through_sdk(
    runner, monkeypatch, patch_type, settings, expected_tokens
):
    captured = []
    content = f"<NAME>improved</NAME><DESCRIPTION>Sharper guidance</DESCRIPTION><PROMPT>{PROMPT_TEXT}</PROMPT>"

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_prompt",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-5-mini",
                "output": [
                    {
                        "id": "msg_prompt",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": content, "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async def run():
        async with openai.AsyncOpenAI(
            api_key="test-only",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as sdk:
            monkeypatch.setattr(
                query_module,
                "get_async_client_llm",
                lambda model, **kwargs: (sdk, model, "openai"),
            )
            monkeypatch.setattr(llm_module, "MAX_RETRIES", 1)
            runner.evo_config.prompt_llm_kwargs = settings
            runner.evo_config.prompt_patch_types = [patch_type]
            runner.evo_config.prompt_patch_type_probs = [1.0]
            runner.prompt_llm = AsyncLLMClient(
                model_names="gpt-5-mini", verbose=False, **settings
            )
            await runner._setup_prompt_evolution()
            parent = runner.prompt_db.get_all_prompts()[0]
            return await runner.prompt_evolver.evolve(parent, next_generation=1)

    prompt, mutation, cost = asyncio.run(run())
    assert prompt is not None
    assert prompt.prompt_text == PROMPT_TEXT
    assert mutation == patch_type
    assert cost > 0
    assert len(captured) == 1
    assert captured[0]["max_output_tokens"] == expected_tokens
    assert captured[0]["reasoning"]["effort"] == "low"
    assert not {"temperatures", "reasoning_efforts", "max_tokens"}.intersection(
        captured[0]
    )
    assert prompt.metadata["llm"]["llm_kwargs"]["max_output_tokens"] == expected_tokens


@pytest.mark.parametrize(
    "database_state", ["fresh", "empty", "seed", "evolved", "duplicate-seeds"]
)
def test_prompt_setup_reuses_persisted_prompts(runner, database_state):
    existing_generations = {
        "fresh": [],
        "empty": [],
        "seed": [0],
        "evolved": [0, 1],
        "duplicate-seeds": [0, 0],
    }[database_state]
    expected = {}
    if database_state != "fresh":
        db = SystemPromptDatabase(
            SystemPromptConfig(db_path=runner.results_dir + "/prompts.sqlite")
        )
        try:
            parent_id = None
            for generation in existing_generations:
                prompt = create_system_prompt(
                    prompt_text=PROMPT_TEXT,
                    generation=generation,
                    parent_id=parent_id if generation else None,
                    patch_type="full" if generation else "init",
                    metadata={"llm": {"cost": 0.5 if generation else 0.0}},
                )
                db.add(prompt)
                db.update_fitness(
                    prompt.id,
                    percentile=0.75,
                    program_id="existing-program-" + prompt.id,
                    correct=True,
                    improvement=0.1,
                    program_score=2.0,
                )
                parent_id = prompt.id
            expected = {prompt.id: prompt.to_dict() for prompt in db.get_all_prompts()}
        finally:
            db.close()

    expected_cost = 0.5 if database_state == "evolved" else 0.0
    runner.prompt_llm = AsyncLLMClient(model_names="gpt-5-mini", verbose=False)

    async def run():
        nonlocal expected
        for _ in range(2):
            await runner._setup_prompt_evolution()
            actual = {
                prompt.id: prompt.to_dict()
                for prompt in runner.prompt_db.get_all_prompts()
            }
            if expected:
                assert actual == expected
            else:
                assert len(actual) == 1
                prompt = next(iter(actual.values()))
                assert prompt["generation"] == 0
                assert prompt["prompt_text"] == PROMPT_TEXT
                expected = actual
            assert runner.prompt_db.get_total_evolution_costs() == expected_cost
            assert runner.prompt_sampler_evo.sample().id in expected
            runner.prompt_db.close()
            runner.prompt_db = None

    asyncio.run(run())
