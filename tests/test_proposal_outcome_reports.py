"""Every proposal the runner submits to the bandit ends in one reported outcome."""

import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from shinka.core import async_runner
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import Program
from shinka.llm.prioritization import AsymmetricUCB, BanditBase
from shinka.llm.providers.errors import LLMEndpointError, RequestRejectedError
from shinka.llm.providers.result import QueryResult

MODEL = "model-a"
RESPONSE = "<NAME>tweak</NAME>\n```python\nx = 2\n```"
REPLY_COST = 0.03


class _RecordingBandit:
    """Stands in for the LLM-selection bandit and lists the calls it gets."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def update_submitted(self, arm: str) -> float:
        self.calls.append(("submitted", arm))
        return 1.0

    def update_cost(self, arm: str, cost: float) -> float:
        self.calls.append(("cost", arm, cost))
        return cost

    def update_delivered(self, arm: str) -> None:
        self.calls.append(("delivered", arm))

    def update_failed(self, arm: str, cost: float = 0.0) -> None:
        self.calls.append(("failed", arm, cost))


class _ScriptedLLM:
    """Answers every query with `reply()`: a QueryResult, None, or a raise."""

    def __init__(self, reply: Callable[[], Optional[QueryResult]]) -> None:
        self.reply = reply

    def get_kwargs(
        self, model_sample_probs: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        return {"model_name": MODEL, "temperature": 0.0}

    async def query(self, **_kwargs: Any) -> Optional[QueryResult]:
        return self.reply()


def _reply(content: Optional[str] = RESPONSE) -> QueryResult:
    return QueryResult(
        content=content,
        msg="request",
        system_msg="system",
        new_msg_history=[],
        model_name=MODEL,
        kwargs={},
        input_tokens=1,
        output_tokens=1,
        cost=REPLY_COST,
    )


def _raise(error: BaseException) -> Callable[[], Optional[QueryResult]]:
    def reply() -> Optional[QueryResult]:
        raise error

    return reply


async def _patch_applies(**_kwargs: Any) -> Tuple[Any, ...]:
    return "x = 2", 1, None, None, "patch text", None


async def _patch_does_not_apply(**_kwargs: Any) -> Tuple[Any, ...]:
    return "x = 1", 0, None, "No changes applied", None, None


def _build_runner(reply: Callable[[], Optional[QueryResult]], bandit: Any) -> Any:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.llm = _ScriptedLLM(reply)
    runner.llm_selection = bandit
    runner.verbose = False
    runner.total_api_cost = 0.0
    runner.results_dir = "results"
    runner.lang_ext = "py"
    runner.evo_config = SimpleNamespace(max_patch_attempts=2, language="python")
    runner.prompt_sampler = SimpleNamespace(
        task_sys_msg=None,
        sample=lambda **_kwargs: ("system", "patch request", "diff"),
        sample_fix=lambda **_kwargs: ("system", "fix request", "full"),
    )
    runner._get_current_system_prompt = lambda: (None, None)

    async def _save_patch_attempt_async(**_kwargs: Any) -> None:
        return None

    runner._save_patch_attempt_async = _save_patch_attempt_async
    return runner


def _run_patch(runner: Any) -> Any:
    return asyncio.run(
        runner._run_patch_async(
            parent_program=Program(id="parent", code="x = 1", generation=0),
            archive_programs=[],
            top_k_programs=[],
            generation=1,
        )
    )


def _run_fix(runner: Any) -> Any:
    return asyncio.run(
        runner._run_fix_patch_async(
            incorrect_program=Program(id="broken", code="x = (", generation=0),
            ancestor_inspirations=[],
            generation=1,
        )
    )


BOTH_PROPOSAL_KINDS = pytest.mark.parametrize(
    "run", [_run_patch, _run_fix], ids=["patch", "fix"]
)


@BOTH_PROPOSAL_KINDS
def test_delivered_patch_is_reported_with_its_cost(
    run: Callable[[Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = _RecordingBandit()

    _, _, success = run(_build_runner(_reply, bandit))

    assert success is True
    assert bandit.calls == [
        ("submitted", MODEL),
        ("cost", MODEL, REPLY_COST),
        ("delivered", MODEL),
    ]


@BOTH_PROPOSAL_KINDS
def test_proposal_without_a_usable_patch_is_reported_as_failed(
    run: Callable[[Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_does_not_apply)
    bandit = _RecordingBandit()

    _, _, success = run(_build_runner(_reply, bandit))

    assert success is False
    # Two billed attempts; the bill is not reported as a cost observation.
    assert bandit.calls == [
        ("submitted", MODEL),
        ("failed", MODEL, pytest.approx(2 * REPLY_COST)),
    ]


@BOTH_PROPOSAL_KINDS
def test_empty_replies_are_reported_as_failed(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()

    _, _, success = run(_build_runner(lambda: None, bandit))

    assert success is False
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, 0.0)]


@BOTH_PROPOSAL_KINDS
def test_endpoint_failure_is_reported_as_failed(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()
    error = LLMEndpointError(401, "invalid_api_key")
    error.cost = 0.01

    _, metadata, success = run(_build_runner(_raise(error), bandit))

    assert success is False
    assert metadata["endpoint_error"] == {"status_code": 401, "code": "invalid_api_key"}
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, 0.01)]


@pytest.mark.parametrize(
    "error",
    [
        RequestRejectedError("LLM request rejected with HTTP 400"),
        TypeError("create() got an unexpected keyword argument 'temperature'"),
    ],
    ids=["rejected-request", "client-error"],
)
@BOTH_PROPOSAL_KINDS
def test_failed_request_is_reported_as_failed(
    run: Callable[[Any], Any], error: Exception
) -> None:
    bandit = _RecordingBandit()

    _, _, success = run(_build_runner(_raise(error), bandit))

    assert success is False
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, 0.0)]


@BOTH_PROPOSAL_KINDS
def test_cancelled_proposal_reports_no_outcome(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()

    with pytest.raises(asyncio.CancelledError):
        run(_build_runner(_raise(asyncio.CancelledError()), bandit))

    assert bandit.calls == [("submitted", MODEL)]


@BOTH_PROPOSAL_KINDS
def test_exhausted_budget_reports_no_outcome(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()

    with pytest.raises(async_runner.APIBudgetExceeded):
        run(_build_runner(_raise(async_runner.APIBudgetExceeded()), bandit))

    assert bandit.calls == [("submitted", MODEL)]


@BOTH_PROPOSAL_KINDS
def test_failure_before_a_model_is_chosen_reports_no_outcome(
    run: Callable[[Any], Any],
) -> None:
    bandit = _RecordingBandit()
    runner = _build_runner(_reply, bandit)

    def no_prompt(**_kwargs: Any) -> Tuple[str, str, str]:
        raise RuntimeError("no parent to build a prompt from")

    runner.prompt_sampler.sample = no_prompt
    runner.prompt_sampler.sample_fix = no_prompt

    _, _, success = run(runner)

    assert success is False
    assert bandit.calls == []


@BOTH_PROPOSAL_KINDS
def test_runner_without_a_bandit_still_handles_failures(
    run: Callable[[Any], Any],
) -> None:
    error = LLMEndpointError(401, "invalid_api_key")

    _, metadata, success = run(_build_runner(_raise(error), None))

    assert success is False
    assert metadata["model_name"] == MODEL


@BOTH_PROPOSAL_KINDS
def test_endpoint_failures_turn_the_real_bandit_away_from_the_model(
    run: Callable[[Any], Any],
) -> None:
    bandit = AsymmetricUCB(arm_names=[MODEL, "model-b"], cost_aware_coef=0.5)
    runner = _build_runner(_raise(LLMEndpointError(402, "insufficient_quota")), bandit)
    bandit.update_submitted("model-b")
    bandit.update_cost("model-b", REPLY_COST)
    bandit.update_delivered("model-b")
    bandit.update("model-b", 1.0, baseline=1.0)

    for _ in range(3):
        run(runner)

    assert bandit.n_submitted.tolist() == [3.0, 1.0]
    assert bandit.reliability()[0] == pytest.approx(0.729 / 3.439)
    assert bandit.posterior()[0] < 0.1


@pytest.mark.parametrize("raising_step", ["_print_metadata_table", "to_dict"])
@BOTH_PROPOSAL_KINDS
def test_proposal_that_breaks_after_its_patch_applied_is_reported_once(
    run: Callable[[Any], Any], raising_step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = _RecordingBandit()
    runner = _build_runner(_reply, bandit)

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("cannot render")

    if raising_step == "to_dict":
        monkeypatch.setattr(QueryResult, "to_dict", broken)
    else:
        runner.verbose = True
        runner._print_metadata_table = broken

    _, _, success = run(runner)

    assert success is False
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, REPLY_COST)]


@BOTH_PROPOSAL_KINDS
def test_no_attempt_at_all_is_reported_once(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()
    runner = _build_runner(_reply, bandit)
    runner.evo_config.max_patch_attempts = 0

    _, _, success = run(runner)

    assert success is False
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, 0.0)]


class _OwnNamesBandit(BanditBase):
    """A bandit as users wrote them before outcome reports: it maps model
    names itself and never gave BanditBase any."""

    def __init__(self) -> None:
        super().__init__(n_arms=2)
        self.index = {MODEL: 0, "model-b": 1}
        self.submitted = [0, 0]
        self.costs = [0.0, 0.0]

    def update_submitted(self, arm: Any) -> float:
        self.submitted[self.index[arm]] += 1
        return float(self.submitted[self.index[arm]])

    def update_cost(self, arm: Any, cost: float) -> float:
        self.costs[self.index[arm]] += cost
        return cost

    def update(self, arm: Any, reward: Any, baseline: Any = None) -> None:
        return None

    def posterior(self, subset: Any = None, samples: Any = None, **_: Any) -> Any:
        return [0.5, 0.5]

    def decay(self, factor: float) -> None:
        return None

    def print_summary(self, console: Any = None) -> None:
        return None

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        return None


@BOTH_PROPOSAL_KINDS
def test_bandit_with_its_own_arm_names_keeps_working(
    run: Callable[[Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = _OwnNamesBandit()

    _, _, delivered = run(_build_runner(_reply, bandit))
    _, metadata, failed = run(
        _build_runner(_raise(LLMEndpointError(401, "invalid_api_key")), bandit)
    )

    assert delivered is True
    assert failed is False
    assert metadata["endpoint_error"] == {"status_code": 401, "code": "invalid_api_key"}
    assert bandit.submitted == [2, 0]
    assert bandit.costs == [REPLY_COST, 0.0]
