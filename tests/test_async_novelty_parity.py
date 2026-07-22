"""Regression tests for async novelty-judge parity and dead-code removal."""

import asyncio
from dataclasses import dataclass

from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.novelty_judge import NoveltyJudge
from shinka.database import Program


SENTINEL_KWARGS = {"model_name": "sentinel-model", "temperature": 0.123}


@dataclass
class _Response:
    content: str
    cost: float = 0.1


class _RecordingAsyncLLM:
    def __init__(self):
        self.kwargs_log = []

    def get_kwargs(self):
        return dict(SENTINEL_KWARGS)

    async def query(self, msg, system_msg, llm_kwargs=None):
        self.kwargs_log.append(llm_kwargs)
        return _Response("NOVEL - meaningfully different implementation.")


def test_llm_novelty_forwards_sampled_llm_kwargs():
    llm = _RecordingAsyncLLM()
    judge = AsyncNoveltyJudge(NoveltyJudge(language="python"), async_llm_client=llm)
    similar = Program(
        id="similar-1",
        code="def existing():\n    return 1\n",
        language="python",
        generation=1,
    )

    is_novel, _explanation, _cost = asyncio.run(
        judge._check_llm_novelty_async("def proposed():\n    return 2\n", similar)
    )

    assert is_novel is True
    assert llm.kwargs_log == [SENTINEL_KWARGS]


def test_dead_single_novelty_check_is_removed():
    assert "_single_novelty_check_async" not in vars(AsyncNoveltyJudge)
    assert not hasattr(AsyncNoveltyJudge, "_construct_novelty_prompt")
    assert not hasattr(AsyncNoveltyJudge, "_parse_novelty_response")
