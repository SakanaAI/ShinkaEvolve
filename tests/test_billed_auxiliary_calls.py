"""Empty replies still contribute to novelty, prompt, and meta budgets."""

import asyncio
from types import SimpleNamespace

import pytest

from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.prompt_evolver import AsyncSystemPromptEvolver, SystemPromptEvolver
from shinka.core.summarizer import MetaSummarizer
from shinka.database import Program
from shinka.database.prompt_dbase import create_system_prompt
from shinka.llm.providers import QueryResult


@pytest.fixture
def billed_reply():
    return QueryResult(None, "msg", "sys", [], "gpt-5-mini", {}, 1000, 0, cost=3.3)


def clients(reply):
    async def query_async(**kwargs):
        return reply

    return (
        SimpleNamespace(query=lambda **kwargs: reply, get_kwargs=lambda: {}),
        SimpleNamespace(query=query_async),
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_novelty_reports_empty_reply_cost(mode, billed_reply):
    sync_client, async_client = clients(billed_reply)
    program = Program(id="parent", code="pass", generation=1)
    if mode == "sync":
        judge = NoveltyJudge(novelty_llm_client=sync_client)
        result = judge.check_llm_novelty("pass", program)
    else:
        judge = object.__new__(AsyncNoveltyJudge)
        judge.async_llm_client = async_client
        judge.sync_judge = SimpleNamespace(language="python")
        result = asyncio.run(judge._check_llm_novelty_async("pass", program))
    assert result[2] == pytest.approx(3.3)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("patch_type", ["diff", "full"])
def test_prompt_evolution_reports_empty_reply_cost(mode, patch_type, billed_reply):
    sync_client, async_client = clients(billed_reply)
    parent = create_system_prompt(prompt_text="task", generation=0, patch_type="init")
    evolver_type = SystemPromptEvolver if mode == "sync" else AsyncSystemPromptEvolver
    evolver = evolver_type(
        sync_client if mode == "sync" else async_client,
        patch_types=[patch_type],
        patch_type_probs=[1.0],
    )
    result = evolver.evolve(parent, next_generation=1)
    if mode == "async":
        result = asyncio.run(result)
    assert result[0] is None
    assert result[2] == pytest.approx(3.3)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_meta_individual_summary_reports_empty_reply_cost(mode, billed_reply):
    program = Program(
        id="parent",
        code="pass",
        generation=1,
        correct=True,
        metadata={"patch_name": "patch"},
    )
    sync_client = SimpleNamespace(
        batch_kwargs_query_indexed=lambda **kwargs: [(0, billed_reply)]
    )
    if mode == "sync":
        summarizer = MetaSummarizer(meta_llm_client=sync_client)
        result = summarizer._step1_individual_summaries([program])
    else:

        async def batch(**kwargs):
            return [(0, billed_reply)]

        summarizer = AsyncMetaSummarizer(
            MetaSummarizer(), SimpleNamespace(batch_kwargs_query_indexed=batch)
        )
        result = asyncio.run(summarizer._step1_individual_summaries_async([program]))
    assert result == (None, 3.3)
