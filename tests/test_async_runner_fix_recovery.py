"""Regression tests for async fix-patch terminal recovery."""

import asyncio
from types import SimpleNamespace

from shinka.core.async_runner import ShinkaEvolveRunner


class _FirstNoneAsyncLLM:
    def __init__(self):
        self.query_calls = 0

    def get_kwargs(self, model_sample_probs=None):
        return {"model_name": "fix-model", "temperature": 0.5}

    async def query(self, **kwargs):
        self.query_calls += 1
        return None


def _runner(llm):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.prompt_sampler = SimpleNamespace(
        sample_fix=lambda **kwargs: ("fix system", "fix message", "full")
    )
    runner.verbose = False
    runner.evo_config = SimpleNamespace(max_patch_attempts=1)
    runner.llm = llm
    runner.llm_selection = None

    async def save_patch_attempt(**kwargs):
        return None

    runner._save_patch_attempt_async = save_patch_attempt
    return runner


def test_first_empty_response_returns_complete_failure_metadata():
    llm = _FirstNoneAsyncLLM()

    patch_text, metadata, success = asyncio.run(
        _runner(llm)._run_fix_patch_async(
            incorrect_program=SimpleNamespace(
                id="ip-1", code="def broken():\n    missing_name\n"
            ),
            ancestor_inspirations=[],
            generation=4,
        )
    )

    assert llm.query_calls == 1
    assert patch_text is None
    assert success is False
    assert metadata["error_attempt"] == "Max fix attempts reached without success."
    assert metadata["patch_name"] is None
    assert metadata["patch_description"] is None
    assert metadata["llm_result"] is None
    assert metadata["model_name"] == "fix-model"
