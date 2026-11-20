"""Every proposal the runner submits to the bandit ends in one reported outcome."""

import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from shinka.core import async_runner
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import Program
from shinka.llm.prioritization import AsymmetricUCB, BanditBase, ProposalOutcomes
from shinka.llm.providers.errors import LLMEndpointError, RequestRejectedError
from shinka.llm.providers.result import QueryResult

MODEL = "model-a"
OTHER = "model-b"
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


@BOTH_PROPOSAL_KINDS
def test_rejected_request_is_reported_as_failed(run: Callable[[Any], Any]) -> None:
    bandit = _RecordingBandit()
    error = RequestRejectedError("LLM request rejected with HTTP 400")

    _, _, success = run(_build_runner(_raise(error), bandit))

    assert success is False
    assert bandit.calls == [("submitted", MODEL), ("failed", MODEL, 0.0)]


@pytest.mark.parametrize(
    "error",
    [OSError(28, "No space left on device"), RuntimeError("bug in the runner")],
    ids=["disk-full", "runner-bug"],
)
@BOTH_PROPOSAL_KINDS
def test_runner_side_error_is_not_held_against_the_model(
    run: Callable[[Any], Any], error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model answered and its patch applied; saving the attempt broke."""
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = _RecordingBandit()
    runner = _build_runner(_reply, bandit)

    async def cannot_save(**_kwargs: Any) -> None:
        raise error

    runner._save_patch_attempt_async = cannot_save

    _, metadata, success = run(runner)

    assert success is False
    assert metadata["api_costs"] == REPLY_COST
    assert bandit.calls == [("submitted", MODEL)]


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
def test_proposal_that_breaks_after_its_patch_applied_reports_no_outcome(
    run: Callable[[Any], Any], raising_step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not delivered, and not the model's failure either."""
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
    assert bandit.calls == [("submitted", MODEL)]


@BOTH_PROPOSAL_KINDS
def test_no_attempt_at_all_reports_no_outcome(run: Callable[[Any], Any]) -> None:
    """With no attempt allowed the model is never asked."""
    bandit = _RecordingBandit()
    runner = _build_runner(_reply, bandit)
    runner.evo_config.max_patch_attempts = 0

    _, _, success = run(runner)

    assert success is False
    assert bandit.calls == [("submitted", MODEL)]


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


class _NoBaseInitBandit(_OwnNamesBandit):
    """Overrides everything it uses and never initialises BanditBase."""

    def __init__(self) -> None:
        self.index = {MODEL: 0, "model-b": 1}
        self.submitted = [0, 0]
        self.costs = [0.0, 0.0]


@pytest.mark.parametrize("bandit_class", [_OwnNamesBandit, _NoBaseInitBandit])
@BOTH_PROPOSAL_KINDS
def test_bandit_with_its_own_arm_names_keeps_working(
    run: Callable[[Any], Any], bandit_class: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = bandit_class()

    _, _, delivered = run(_build_runner(_reply, bandit))
    _, metadata, failed = run(
        _build_runner(_raise(LLMEndpointError(401, "invalid_api_key")), bandit)
    )

    assert delivered is True
    assert failed is False
    assert metadata["endpoint_error"] == {"status_code": 401, "code": "invalid_api_key"}
    assert bandit.submitted == [2, 0]
    assert bandit.costs == [REPLY_COST, 0.0]


class _SelectingBandit(_RecordingBandit):
    """Selects the first of the two models for every task."""

    def select_llm(self) -> Tuple[Any, Any]:
        return np.array([1.0, 0.0]), np.array([0.5, 0.5])


class _PoolLLM:
    """Two models; `replies[model]()` answers a query sent to that model."""

    model_names = [MODEL, OTHER]
    model_sample_probs = [0.5, 0.5]

    def __init__(self, replies: Dict[str, Callable[[], Optional[QueryResult]]]) -> None:
        self.replies = replies

    def get_kwargs(self, model_sample_probs: Any = None) -> Dict[str, Any]:
        model = self.model_names[int(np.argmax(model_sample_probs))]
        return {"model_name": model, "temperature": 0.0}

    async def query(self, **kwargs: Any) -> Optional[QueryResult]:
        return self.replies[kwargs["llm_kwargs"]["model_name"]]()


def _run_task(
    replies: Dict[str, Callable[[], Optional[QueryResult]]], bandit: Any
) -> None:
    """One proposal task as the runner runs it: one selection, then the
    resample and novelty loops around the real patch call."""
    runner = _build_runner(_reply, bandit)
    runner.llm = _PoolLLM(replies)
    runner.evo_config = SimpleNamespace(
        max_patch_attempts=1,
        max_patch_resamples=3,
        max_novelty_attempts=3,
        max_api_costs=None,
        language="python",
    )
    runner.db_config = SimpleNamespace(parent_selection_strategy="weighted")
    runner.novelty_judge = None
    runner.slot_available = asyncio.Event()
    parent = Program(id="parent", code="x = 1", generation=0)

    async def sample(**_kwargs: Any) -> Tuple[Any, ...]:
        return parent, [], [], False

    async def no_embedding(_exec_fname: str) -> Tuple[None, float]:
        return None, 0.0

    async def settled(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def not_submitted(**_kwargs: Any) -> None:
        raise RuntimeError("the test stops before evaluation")

    runner.async_db = SimpleNamespace(sample_with_fix_mode_async=sample)
    runner._get_code_embedding_async = no_embedding
    runner._record_terminal_failed_proposal = settled
    runner._record_evaluation_charge = settled
    runner._record_unattached_cost = settled
    runner._submit_evaluation_job_with_slot = not_submitted
    asyncio.run(
        runner._generate_evolved_proposal(
            1, "task", "unused", "unused", None, None, None, 0, None, 1
        )
    )


def _outcomes(bandit: _RecordingBandit) -> List[Tuple[Any, ...]]:
    return [call for call in bandit.calls if call[0] in ("delivered", "failed")]


def test_task_whose_patches_never_apply_reports_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three novelty attempts of three resamples call the same model nine times."""
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_does_not_apply)
    bandit = _SelectingBandit()

    _run_task({MODEL: _reply}, bandit)

    assert bandit.calls.count(("submitted", MODEL)) == 9
    assert _outcomes(bandit) == [("failed", MODEL, pytest.approx(9 * REPLY_COST))]


def test_task_that_delivers_after_failed_resamples_reports_no_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied = iter([_patch_does_not_apply, _patch_does_not_apply, _patch_applies])

    async def third_time_lucky(**kwargs: Any) -> Tuple[Any, ...]:
        return await next(applied)(**kwargs)

    monkeypatch.setattr(async_runner, "apply_patch_async", third_time_lucky)
    bandit = _SelectingBandit()

    _run_task({MODEL: _reply}, bandit)

    assert bandit.calls.count(("submitted", MODEL)) == 3
    assert _outcomes(bandit) == [("delivered", MODEL)]
    # The cost observation is the call that delivered, as before.
    assert ("cost", MODEL, REPLY_COST) in bandit.calls


def test_task_that_moves_to_another_model_reports_both_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_applies)
    bandit = _SelectingBandit()

    _run_task(
        {MODEL: _raise(LLMEndpointError(401, "invalid_api_key")), OTHER: _reply},
        bandit,
    )

    assert _outcomes(bandit) == [("failed", MODEL, 0.0), ("delivered", OTHER)]


def test_cancelled_task_reports_no_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_runner, "apply_patch_async", _patch_does_not_apply)
    replies = iter([_reply, _raise(asyncio.CancelledError())])
    bandit = _SelectingBandit()

    with pytest.raises(asyncio.CancelledError):
        _run_task({MODEL: lambda: next(replies)()}, bandit)

    assert bandit.calls.count(("submitted", MODEL)) == 2
    assert _outcomes(bandit) == []


def test_outcomes_of_one_task_reach_the_real_bandit_once() -> None:
    bandit = AsymmetricUCB(arm_names=[MODEL, OTHER], cost_aware_coef=0.5)
    outcomes = ProposalOutcomes(bandit)

    for _ in range(9):
        outcomes.failed(MODEL, cost=REPLY_COST)
    outcomes.failed(OTHER, cost=REPLY_COST)
    outcomes.delivered(OTHER)
    outcomes.report()
    outcomes.report()  # Nothing is left to report.

    assert bandit.recent_failed.tolist() == [1.0, 0.0]
    assert bandit.recent_delivered.tolist() == [0.0, 1.0]
    # The bills of a model that gave the task nothing are its cost so far.
    assert bandit.total_costs.tolist() == [pytest.approx(9 * REPLY_COST), 0.0]


def test_outcomes_without_a_bandit_are_dropped() -> None:
    outcomes = ProposalOutcomes(None)
    outcomes.failed(MODEL)
    outcomes.delivered(OTHER)

    outcomes.report()
