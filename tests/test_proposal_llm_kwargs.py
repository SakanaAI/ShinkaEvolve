import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from shinka.core import async_runner
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import Program
from shinka.llm.providers.result import QueryResult

RESPONSE = "<NAME>tweak</NAME>\n```python\nx = 2\n```"
RECORDED_KWARGS = {"model_name": "model-a", "temperature": 0.0, "max_tokens": 4096}
RESAMPLED_KWARGS = {"model_name": "model-a", "temperature": 1.0, "max_tokens": 8192}


class _RecordingLLM:
    """Mimics AsyncLLMClient: a query without llm_kwargs samples its own."""

    def __init__(self) -> None:
        self.sent_kwargs: List[Dict[str, Any]] = []

    def get_kwargs(
        self, model_sample_probs: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        return dict(RECORDED_KWARGS)

    async def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        model_sample_probs: Optional[List[float]] = None,
        model_posterior: Optional[List[float]] = None,
    ) -> QueryResult:
        sent = dict(llm_kwargs if llm_kwargs is not None else RESAMPLED_KWARGS)
        self.sent_kwargs.append(sent)
        return QueryResult(
            content=RESPONSE,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[],
            model_name=sent["model_name"],
            kwargs=sent,
            input_tokens=1,
            output_tokens=1,
        )


async def _apply_no_changes(**_kwargs: Any) -> Tuple[Any, ...]:
    return "x = 1", 0, None, "No changes applied", None, None


def _build_runner(llm: _RecordingLLM) -> Any:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.llm = llm
    runner.llm_selection = None
    runner.verbose = False
    runner.results_dir = "results"
    runner.evo_config = SimpleNamespace(max_patch_attempts=1, language="python")
    runner.prompt_sampler = SimpleNamespace(
        task_sys_msg=None,
        sample=lambda **_kwargs: ("system", "patch request", "diff"),
        sample_fix=lambda **_kwargs: ("system", "fix request", "full"),
        initial_program_prompt=lambda: ("system", "initial request"),
    )
    runner._get_current_system_prompt = lambda: (None, None)

    async def _save_patch_attempt_async(**_kwargs: Any) -> None:
        return None

    runner._save_patch_attempt_async = _save_patch_attempt_async
    return runner


def _recorded(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: metadata[key] for key in RECORDED_KWARGS}


def test_patch_proposal_sends_the_kwargs_it_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _apply_no_changes)
    llm = _RecordingLLM()
    runner = _build_runner(llm)

    _, metadata, success = asyncio.run(
        runner._run_patch_async(
            parent_program=Program(id="parent", code="x = 1", generation=0),
            archive_programs=[],
            top_k_programs=[],
            generation=1,
        )
    )

    assert success is False
    assert llm.sent_kwargs == [_recorded(metadata)]


def test_fix_proposal_sends_the_kwargs_it_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _apply_no_changes)
    llm = _RecordingLLM()
    runner = _build_runner(llm)

    _, metadata, success = asyncio.run(
        runner._run_fix_patch_async(
            incorrect_program=Program(id="broken", code="x = (", generation=0),
            ancestor_inspirations=[],
            generation=1,
        )
    )

    assert success is False
    assert llm.sent_kwargs == [_recorded(metadata)]


def test_initial_program_sends_the_kwargs_it_records() -> None:
    llm = _RecordingLLM()
    runner = _build_runner(llm)
    stored_metadata: List[Dict[str, Any]] = []

    async def _setup_initial_program_with_metadata(
        code: str,
        patch_name: Optional[str],
        patch_description: Optional[str],
        total_costs: float,
        metadata: Dict[str, Any],
    ) -> None:
        stored_metadata.append(metadata)

    runner._setup_initial_program_with_metadata = _setup_initial_program_with_metadata

    asyncio.run(runner._generate_initial_program())

    assert llm.sent_kwargs == [_recorded(stored_metadata[0])]
