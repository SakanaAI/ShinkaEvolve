"""An unreachable LLM endpoint must stop the run, not spend its generations."""

import asyncio
import logging
from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.endpoint_failures import (
    ENDPOINT_FAILURES_BEFORE_STOP,
    endpoint_failures_before_stop,
    probabilities_without,
)
from shinka.database import DatabaseConfig, Program
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import LLMEndpointError, RequestRejectedError

BAD = "gpt-5-mini"
ALSO_BAD = "gpt-5-nano"
GOOD = "gpt-5.1"


class _Proposals:
    """A runner reduced to proposal generation, with a scripted provider."""

    def __init__(self, monkeypatch, models, dead_models, fix_mode=False, weights=None):
        self.requested_models: list[str] = []
        self.settled: list[dict] = []
        self.dead_models = set(dead_models)
        self.rejecting_models: set[str] = set()
        runner: Any = object.__new__(ShinkaEvolveRunner)
        runner.evo_config = EvolutionConfig(max_patch_attempts=1)
        runner.db_config = DatabaseConfig()
        runner.llm_selection = None
        runner.verbose = False
        runner.total_api_cost = 0.0
        runner.slot_available = asyncio.Event()
        parent = Program(id="parent", code="pass", generation=0, correct=not fix_mode)

        async def sample(**kwargs):
            return parent, [], [], fix_mode

        async def save_attempt(**kwargs):
            return None

        async def settle(**kwargs):
            self.settled.append(kwargs)

        runner.async_db = SimpleNamespace(sample_with_fix_mode_async=sample)
        runner.prompt_sampler = SimpleNamespace(
            task_sys_msg="task",
            sample=lambda **kwargs: ("sys", "patch", "diff"),
            sample_fix=lambda **kwargs: ("sys", "patch", "diff"),
        )
        runner._get_current_system_prompt = lambda: (None, None)
        runner._save_patch_attempt_async = save_attempt
        runner._record_terminal_failed_proposal = settle
        runner.llm = AsyncLLMClient(
            model_names=models, model_sample_probs=weights, verbose=False
        )
        monkeypatch.setattr(llm_module, "query_async", self._query)
        self.runner = runner

    async def _query(self, **kwargs):
        self.requested_models.append(kwargs["model_name"])
        if kwargs["model_name"] in self.dead_models:
            raise LLMEndpointError(401, "invalid_api_key")
        if kwargs["model_name"] in self.rejecting_models:
            raise RequestRejectedError("OpenAI rejected the request with HTTP 403")
        return None  # The endpoint answered, with nothing usable.

    def propose(self, generation: int) -> None:
        asyncio.run(
            self.runner._generate_evolved_proposal(
                generation, "task", "unused", "unused", None, None, None, 0, None, 1
            )
        )

    @property
    def settled_generations(self) -> list[int]:
        return [failure["generation"] for failure in self.settled]


@pytest.mark.parametrize("fix_mode", [False, True])
def test_single_model_pool_stops_on_the_first_endpoint_failure(monkeypatch, fix_mode):
    proposals = _Proposals(monkeypatch, [BAD], dead_models=[BAD], fix_mode=fix_mode)

    proposals.propose(generation=7)

    assert proposals.requested_models == [BAD]
    # The generation is left for a resumed run instead of being settled as failed.
    assert proposals.settled == []
    assert proposals.runner._endpoint_stop == {
        "generation": 7,
        "status_code": 401,
        "code": "invalid_api_key",
    }
    assert proposals.runner.slot_available.is_set()


def test_rejected_request_costs_its_generation_and_the_run_goes_on(monkeypatch):
    """A prompt the provider refuses (OpenRouter moderation: 403) is no dead endpoint."""
    proposals = _Proposals(monkeypatch, [BAD], dead_models=[])
    proposals.rejecting_models.add(BAD)

    proposals.propose(generation=7)

    assert proposals.runner._endpoint_stop is None
    assert proposals.settled_generations == [7]


def test_proposals_already_in_flight_are_not_settled_after_the_stop(monkeypatch):
    proposals = _Proposals(monkeypatch, [BAD], dead_models=[BAD])

    proposals.propose(generation=7)
    proposals.propose(generation=8)

    assert proposals.settled == []
    assert proposals.runner._endpoint_stop["generation"] == 7


@pytest.mark.parametrize("fix_mode", [False, True])
def test_mixed_pool_resamples_a_model_whose_endpoint_works(monkeypatch, fix_mode):
    proposals = _Proposals(
        monkeypatch,
        [BAD, GOOD],
        dead_models=[BAD],
        fix_mode=fix_mode,
        weights=[1.0, 0.0],
    )

    proposals.propose(generation=1)

    assert proposals.requested_models[0] == BAD
    assert set(proposals.requested_models[1:]) == {GOOD}
    assert proposals.runner._endpoint_stop is None
    # The working model gave no patch, so this is an ordinary failed proposal.
    assert proposals.settled_generations == [1]
    assert "endpoint_error" not in proposals.settled[0]["meta_patch_data"]


def test_switching_away_from_a_failed_endpoint_is_reported(monkeypatch, caplog):
    """Otherwise a dead model in a working pool leaves no trace in the run log."""
    proposals = _Proposals(
        monkeypatch, [BAD, GOOD], dead_models=[BAD], weights=[1.0, 0.0]
    )

    with caplog.at_level(logging.WARNING, logger="shinka.core.async_runner"):
        proposals.propose(generation=4)

    (warning,) = [
        record.getMessage() for record in caplog.records if BAD in record.getMessage()
    ]
    assert "HTTP 401" in warning
    assert "invalid_api_key" in warning
    assert "generation 4" in warning


def test_mixed_pool_stops_after_consecutive_proposals_without_an_endpoint(monkeypatch):
    proposals = _Proposals(monkeypatch, [BAD, ALSO_BAD], dead_models=[BAD, ALSO_BAD])
    last = ENDPOINT_FAILURES_BEFORE_STOP

    for generation in range(1, last + 1):
        assert proposals.runner._endpoint_stop is None
        proposals.propose(generation)

    # Each proposal asked every model once before giving up.
    assert sorted(proposals.requested_models[:2]) == sorted([BAD, ALSO_BAD])
    assert len(proposals.requested_models) == 2 * last
    assert proposals.settled_generations == list(range(1, last))
    assert proposals.runner._endpoint_stop["generation"] == last


def test_a_reply_between_endpoint_failures_restarts_the_count(monkeypatch):
    proposals = _Proposals(monkeypatch, [BAD, ALSO_BAD], dead_models=[BAD, ALSO_BAD])

    for generation in range(1, ENDPOINT_FAILURES_BEFORE_STOP):
        proposals.propose(generation)
    proposals.dead_models.clear()
    proposals.propose(generation=10)
    proposals.dead_models.update([BAD, ALSO_BAD])
    for generation in range(11, 10 + ENDPOINT_FAILURES_BEFORE_STOP):
        proposals.propose(generation)

    assert proposals.runner._endpoint_stop is None


def test_coordinator_ends_the_run_once_in_flight_work_has_drained():
    runner: Any = object.__new__(ShinkaEvolveRunner)
    runner._endpoint_stop = {"generation": 4, "status_code": 401, "code": "revoked"}
    started: list[int] = []

    async def start_proposals(count):
        started.append(count)

    async def run() -> None:
        runner.should_stop = asyncio.Event()
        runner.slot_available = asyncio.Event()
        runner.finalization_complete = asyncio.Event()
        runner.active_proposal_tasks = {}
        runner._start_proposals = start_proposals
        in_flight = [1, 0]
        runner._get_in_flight_work_count = lambda: in_flight.pop(0)
        runner.slot_available.set()
        await asyncio.wait_for(runner._proposal_coordinator_task(), timeout=5)

    asyncio.run(run())

    assert started == []
    assert runner.should_stop.is_set()
    assert runner.finalization_complete.is_set()


def test_run_stopped_by_an_endpoint_failure_is_reported_as_failed():
    runner: Any = ShinkaEvolveRunner.__new__(ShinkaEvolveRunner)
    runner.verbose = False
    evolved = Program(id="child", code="child", generation=1, correct=True)
    runner.db = SimpleNamespace(get_all_programs=lambda: [evolved])
    runner.evo_config = SimpleNamespace(num_generations=5, max_api_costs=None)
    runner.console = Console(file=StringIO())

    asyncio.run(runner._print_final_summary())
    assert runner.evolution_failed is False

    runner._endpoint_stop = {"generation": 2, "status_code": 403, "code": "blocked"}
    asyncio.run(runner._print_final_summary())
    assert runner.evolution_failed is True


def test_stop_threshold_depends_on_the_pool_size():
    assert endpoint_failures_before_stop(1) == 1
    assert endpoint_failures_before_stop(2) == ENDPOINT_FAILURES_BEFORE_STOP
    assert ENDPOINT_FAILURES_BEFORE_STOP > 1


def test_probabilities_without_keeps_the_remaining_weights_in_proportion():
    models = ["a", "b", "c"]

    assert probabilities_without(models, {"a"}, [0.5, 0.25, 0.25]) == [0.0, 0.5, 0.5]
    assert probabilities_without(models, {"b"}) == [0.5, 0.0, 0.5]
    # A one-hot choice of the failed model leaves no weight to keep.
    assert probabilities_without(models, {"a"}, [1.0, 0.0, 0.0]) == [0.0, 0.5, 0.5]
    assert probabilities_without(models, {"a", "b", "c"}) is None
