"""Regression tests for prompt-evolver defaults and async LLM kwargs."""

import asyncio
from dataclasses import dataclass

import pytest

from shinka.core.prompt_evolver import AsyncSystemPromptEvolver
from shinka.database.prompt_dbase import create_system_prompt


SENTINEL_KWARGS = {"model_name": "sentinel-model", "temperature": 0.123}


@dataclass
class _Response:
    content: str
    cost: float = 0.1


class _RecordingAsyncLLM:
    def __init__(self):
        self.kwargs_log = []

    async def query(self, msg, system_msg, llm_kwargs=None):
        self.kwargs_log.append(llm_kwargs)
        return _Response(
            "<PROMPT>You are a rigorous engineer who writes precise code.</PROMPT>"
        )


def _parent_prompt():
    return create_system_prompt(
        prompt_text="You are an expert programmer.",
        generation=0,
        patch_type="init",
    )


@pytest.mark.parametrize("method_name", ["_diff_mutate_async", "_full_rewrite_async"])
def test_async_mutation_forwards_configured_llm_kwargs(method_name):
    llm = _RecordingAsyncLLM()
    evolver = AsyncSystemPromptEvolver(llm, llm_kwargs=dict(SENTINEL_KWARGS))

    asyncio.run(getattr(evolver, method_name)(_parent_prompt(), top_programs=[]))

    assert llm.kwargs_log == [SENTINEL_KWARGS]


@pytest.mark.parametrize("method_name", ["_diff_mutate_async", "_full_rewrite_async"])
def test_async_mutation_forwards_empty_kwargs_as_mapping(method_name):
    llm = _RecordingAsyncLLM()
    evolver = AsyncSystemPromptEvolver(llm)

    asyncio.run(getattr(evolver, method_name)(_parent_prompt(), top_programs=[]))

    assert llm.kwargs_log == [{}]
