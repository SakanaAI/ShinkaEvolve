"""Regression tests for async meta-summarizer LLM argument parity."""

import asyncio
from dataclasses import dataclass

from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer


SENTINEL_KWARGS = {"model_name": "sentinel-model", "temperature": 0.123}


@dataclass
class _Response:
    content: str
    cost: float = 0.1


class _RecordingAsyncLLM:
    def __init__(self, content):
        self.content = content
        self.kwargs_log = []

    def get_kwargs(self):
        return dict(SENTINEL_KWARGS)

    async def query(self, msg, system_msg, llm_kwargs=None):
        self.kwargs_log.append(llm_kwargs)
        return _Response(self.content)


def _summarizer(content):
    sync = MetaSummarizer(meta_llm_client=None, max_recommendations=3)
    llm = _RecordingAsyncLLM(content)
    return AsyncMetaSummarizer(sync, async_llm_client=llm), llm


def test_step2_forwards_sampled_llm_kwargs():
    summarizer, llm = _summarizer("Global insight block")

    result, _cost = asyncio.run(
        summarizer._step2_global_insights_async("individual summaries")
    )

    assert result == "Global insight block"
    assert llm.kwargs_log == [SENTINEL_KWARGS]


def test_step3_forwards_sampled_llm_kwargs():
    summarizer, llm = _summarizer("1. Recommendation A")

    result, _cost = asyncio.run(
        summarizer._step3_generate_recommendations_async("global insights")
    )

    assert result == "1. Recommendation A"
    assert llm.kwargs_log == [SENTINEL_KWARGS]
