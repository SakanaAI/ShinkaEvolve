"""Regression tests for synchronous client retry pacing."""

import shinka.llm.llm as llm_module
from shinka.llm.constants import MAX_RETRIES
from shinka.llm.llm import LLMClient


def test_sync_query_sleeps_between_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    def always_fails(**kwargs):
        raise RuntimeError("transient failure")

    monkeypatch.setattr(llm_module, "query", always_fails)
    client = LLMClient(model_names="test-model", verbose=False)

    result = client.query("msg", "sys", llm_kwargs={"model_name": "test-model"})

    assert result is None
    assert sleeps == [1] * (MAX_RETRIES - 1)
