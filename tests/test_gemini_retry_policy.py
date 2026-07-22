"""Regression tests for Gemini sync/async defaults and retry policy."""

import asyncio
from types import SimpleNamespace

from google.genai import types

from shinka.llm.providers.gemini import (
    DEFAULT_THINKING_BUDGET,
    query_gemini,
    query_gemini_async,
)


def _part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def _response():
    content = SimpleNamespace(parts=[_part("hi")])
    candidate = SimpleNamespace(content=content, finish_reason=types.FinishReason.STOP)
    return SimpleNamespace(candidates=[candidate], text=None)


class _SyncClient:
    def __init__(self, response):
        self.models = SimpleNamespace(generate_content=lambda **kwargs: response)


class _AsyncClient:
    def __init__(self, response):
        async def generate(**kwargs):
            return response

        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate))


def test_default_thinking_budget_matches_across_sync_and_async(monkeypatch):
    from shinka.llm.providers import gemini as gemini_module

    budgets = []

    def record_budget(thinking_budget):
        budgets.append(thinking_budget)
        return None

    monkeypatch.setattr(gemini_module, "build_gemini_thinking_config", record_budget)
    response = _response()

    query_gemini(
        _SyncClient(response), "gemini-2.5-flash", "msg", "sys", [], None
    )
    asyncio.run(
        query_gemini_async(
            _AsyncClient(response), "gemini-2.5-flash", "msg", "sys", [], None
        )
    )

    assert budgets == [DEFAULT_THINKING_BUDGET, DEFAULT_THINKING_BUDGET]
    assert DEFAULT_THINKING_BUDGET == 1024
