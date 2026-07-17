"""Regression tests for async/sync drift bugs in shinka.core.

Each test pins behavior of a LIVE async path against its (correct) sync twin:

1. Async summarizer / novelty / prompt-evolver must forward ``llm_kwargs`` on
   their LLM query calls (sync twins pin it; async copies used to drop it,
   silently falling back to default-sampled model/temperature).
2. ``_run_fix_patch_async`` must not ``NameError`` when the FIRST LLM response
   is ``None`` (twin ``_run_patch_async`` guards ``patch_name`` with
   ``"patch_name" in locals()``).
3. ``SystemPromptEvolver``/``AsyncSystemPromptEvolver`` must construct with the
   documented ``patch_types=None`` default without raising.
4. The dead ``_single_novelty_check_async`` (called undefined helpers) must be
   gone from ``AsyncNoveltyJudge``.
"""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.summarizer import MetaSummarizer
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.prompt_evolver import (
    SystemPromptEvolver,
    AsyncSystemPromptEvolver,
)
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import Program
from shinka.database.prompt_dbase import create_system_prompt


SENTINEL_KWARGS = {"model_name": "sentinel-model", "temperature": 0.123}


@dataclass
class DummyResponse:
    content: str
    cost: float = 0.1


class RecordingAsyncLLM:
    """Async LLM stub that records the ``llm_kwargs`` forwarded to ``query``."""

    def __init__(self, content: str = "NOVEL - a sufficiently long response body."):
        self.content = content
        self.kwargs_log: list = []

    def get_kwargs(self, model_sample_probs=None):
        # New dict each call, mirroring the real client.
        return dict(SENTINEL_KWARGS)

    async def query(self, msg, system_msg, llm_kwargs=None, **extra):
        self.kwargs_log.append(llm_kwargs)
        return DummyResponse(content=self.content, cost=0.1)


# --------------------------------------------------------------------------- #
# Bug 1: async LLM calls must forward llm_kwargs (not silently drop them)
# --------------------------------------------------------------------------- #


def test_async_summarizer_step2_forwards_llm_kwargs():
    sync = MetaSummarizer(meta_llm_client=None, max_recommendations=3)
    fake = RecordingAsyncLLM(content="Global insight block")
    summarizer = AsyncMetaSummarizer(sync, async_llm_client=fake)

    result, cost = asyncio.run(
        summarizer._step2_global_insights_async("individual summaries", None)
    )

    assert result == "Global insight block"
    assert fake.kwargs_log == [SENTINEL_KWARGS]


def test_async_summarizer_step3_forwards_llm_kwargs():
    sync = MetaSummarizer(meta_llm_client=None, max_recommendations=3)
    fake = RecordingAsyncLLM(content="1. Recommendation A")
    summarizer = AsyncMetaSummarizer(sync, async_llm_client=fake)

    result, cost = asyncio.run(
        summarizer._step3_generate_recommendations_async("global insights", None)
    )

    assert result == "1. Recommendation A"
    assert fake.kwargs_log == [SENTINEL_KWARGS]


def test_async_novelty_forwards_llm_kwargs():
    sync_judge = NoveltyJudge(language="python")
    fake = RecordingAsyncLLM(content="NOVEL - meaningfully different implementation.")
    judge = AsyncNoveltyJudge(sync_judge, async_llm_client=fake)
    similar = Program(
        id="similar-1",
        code="def existing():\n    return 1\n",
        language="python",
        generation=1,
    )

    is_novel, explanation, cost = asyncio.run(
        judge._check_llm_novelty_async("def proposed():\n    return 2\n", similar)
    )

    assert is_novel is True
    assert fake.kwargs_log == [SENTINEL_KWARGS]


def test_async_prompt_evolver_diff_forwards_llm_kwargs():
    fake = RecordingAsyncLLM(
        content="<PROMPT>You are a rigorous engineer who writes precise code.</PROMPT>"
    )
    evolver = AsyncSystemPromptEvolver(fake, llm_kwargs=dict(SENTINEL_KWARGS))
    parent = create_system_prompt(
        prompt_text="You are an expert programmer.",
        generation=0,
        patch_type="init",
    )

    asyncio.run(evolver._diff_mutate_async(parent, top_programs=[]))

    assert fake.kwargs_log == [SENTINEL_KWARGS]


def test_async_prompt_evolver_full_forwards_llm_kwargs():
    fake = RecordingAsyncLLM(
        content="<PROMPT>You are a rigorous engineer who writes precise code.</PROMPT>"
    )
    evolver = AsyncSystemPromptEvolver(fake, llm_kwargs=dict(SENTINEL_KWARGS))
    parent = create_system_prompt(
        prompt_text="You are an expert programmer.",
        generation=0,
        patch_type="init",
    )

    asyncio.run(evolver._full_rewrite_async(parent, top_programs=[]))

    assert fake.kwargs_log == [SENTINEL_KWARGS]


# --------------------------------------------------------------------------- #
# Bug 2: _run_fix_patch_async must handle a first None response gracefully
# --------------------------------------------------------------------------- #


class FirstNoneAsyncLLM:
    """Async LLM stub whose first (and only) response is ``None``."""

    def __init__(self):
        self.query_calls = 0

    def get_kwargs(self, model_sample_probs=None):
        return {"model_name": "fix-model", "temperature": 0.5}

    async def query(self, **kwargs):
        self.query_calls += 1
        return None


def _build_fix_runner(llm, max_patch_attempts=1):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.prompt_sampler = SimpleNamespace(
        sample_fix=lambda **kwargs: ("fix system", "fix message", "full")
    )
    runner.verbose = False
    runner.evo_config = SimpleNamespace(max_patch_attempts=max_patch_attempts)
    runner.llm = llm
    runner.llm_selection = None

    async def _save_patch_attempt_async(**kwargs):
        return None

    runner._save_patch_attempt_async = _save_patch_attempt_async
    return runner


def test_run_fix_patch_async_handles_first_none_response():
    llm = FirstNoneAsyncLLM()
    runner = _build_fix_runner(llm, max_patch_attempts=1)

    result = asyncio.run(
        runner._run_fix_patch_async(
            incorrect_program=SimpleNamespace(id="ip-1", code="def broken():\n    x\n"),
            ancestor_inspirations=[],
            generation=4,
        )
    )

    patch_text, meta_patch_data, success = result

    assert llm.query_calls == 1
    assert patch_text is None
    assert success is False
    # Graceful terminal path (NOT the swallowed-NameError fallback, which would
    # set error_attempt to a NameError string and drop llm_kwargs/metadata).
    assert meta_patch_data["error_attempt"] == (
        "Max fix attempts reached without success."
    )
    assert meta_patch_data["patch_name"] is None
    assert meta_patch_data["patch_description"] is None
    assert meta_patch_data["llm_result"] is None
    # llm_kwargs were spread into the metadata (accumulated context preserved).
    assert meta_patch_data["model_name"] == "fix-model"


# --------------------------------------------------------------------------- #
# Bug 3: evolver constructors must accept the documented default
# --------------------------------------------------------------------------- #


def test_system_prompt_evolver_constructs_with_default_patch_types():
    evolver = SystemPromptEvolver(llm_client=object())
    assert evolver.patch_types == ["diff", "full"]
    assert pytest.approx(sum(evolver.patch_type_probs), abs=1e-6) == 1.0


def test_async_system_prompt_evolver_constructs_with_default_patch_types():
    evolver = AsyncSystemPromptEvolver(llm_client=object())
    assert evolver.patch_types == ["diff", "full"]
    assert pytest.approx(sum(evolver.patch_type_probs), abs=1e-6) == 1.0


def test_system_prompt_evolver_rejects_invalid_patch_types():
    with pytest.raises(ValueError):
        SystemPromptEvolver(llm_client=object(), patch_types=["bogus"])
    with pytest.raises(ValueError):
        AsyncSystemPromptEvolver(llm_client=object(), patch_types=["bogus"])


# --------------------------------------------------------------------------- #
# Bug 4: dead method calling undefined helpers must be removed
# --------------------------------------------------------------------------- #


def test_async_novelty_judge_dead_method_removed():
    assert "_single_novelty_check_async" not in vars(AsyncNoveltyJudge)
    assert not hasattr(AsyncNoveltyJudge, "_single_novelty_check_async")
    # The helpers the dead method referenced were never defined anywhere.
    assert not hasattr(AsyncNoveltyJudge, "_construct_novelty_prompt")
    assert not hasattr(AsyncNoveltyJudge, "_parse_novelty_response")
