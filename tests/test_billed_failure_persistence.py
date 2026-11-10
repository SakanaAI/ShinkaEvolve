"""Failed billed work survives resume even when no program was produced."""

import asyncio
from types import SimpleNamespace

import pytest

from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.llm.providers import QueryResult


@pytest.fixture
def runner(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db = db
    runner.prompt_db = None
    runner.total_api_cost = 0.0
    runner.llm_selection = None
    runner.verbose = False
    runner.evo_config = EvolutionConfig(max_patch_attempts=2, max_api_costs=5)
    runner.prompt_sampler = SimpleNamespace(
        initial_program_prompt=lambda: ("sys", "msg")
    )

    async def record(**kwargs):
        db.record_attempt_event(**kwargs)

    async def save(**kwargs):
        pass

    runner.async_db = SimpleNamespace(record_attempt_event_async=record)
    runner._save_patch_attempt_async = save
    yield runner
    db.close()


@pytest.mark.parametrize("reason,expected_calls", [(None, 2), ("max_output_tokens", 1)])
def test_failed_initialization_charges_and_restores_cost(
    runner, reason, expected_calls
):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        return QueryResult(
            None,
            "msg",
            "sys",
            [],
            "gpt-5-mini",
            {},
            10,
            0,
            cost=3.3,
            incomplete_reason=reason,
        )

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    with pytest.raises(RuntimeError):
        asyncio.run(runner._generate_initial_program())
    assert len(calls) == expected_calls
    assert runner.total_api_cost == pytest.approx(3.3 * expected_calls)
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(
        3.3 * expected_calls
    )


def test_resume_includes_terminal_failure_costs_without_counting_attempt_details(
    runner,
):
    runner.db.add(
        Program(id="seed", code="pass", generation=0, metadata={"api_costs": 1.0})
    )
    runner.db.record_attempt_event(
        1,
        "proposal",
        "failed",
        {
            "node_kind": "failed_proposal",
            "api_costs": 2.0,
            "embed_cost": 0.1,
            "novelty_cost": 0.2,
        },
    )
    runner.db.record_attempt_event(1, "patch", "failed", {"api_costs": 100.0})
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)


def test_exhausted_budget_does_not_send_another_initial_request(runner):
    runner.total_api_cost = 6.6

    async def query(**kwargs):
        pytest.fail("budget exhausted before initialization")

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(runner._generate_initial_program())


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_initial_cost_survives_attempt_artifact_failure(runner, error_type):
    async def query(**kwargs):
        return QueryResult(None, "msg", "sys", [], "gpt-5-mini", {}, 10, 0, cost=3.3)

    async def save(**kwargs):
        raise error_type("artifact unavailable")

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    runner._save_patch_attempt_async = save
    with pytest.raises(error_type):
        asyncio.run(runner._generate_initial_program())
    assert runner.total_api_cost == pytest.approx(3.3)
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)


@pytest.mark.parametrize("setup_state", ["none", "persisted", "posted"])
def test_initial_setup_failure_charges_exactly_once(runner, setup_state):
    async def query(**kwargs):
        return QueryResult(
            "```python\npass\n```", "msg", "sys", [], "gpt-5-mini", {}, 10, 10, cost=3.3
        )

    async def setup(code, name, description, cost, metadata):
        if setup_state != "none":
            runner.db.add(
                Program(id="seed", code=code, generation=0, metadata=metadata)
            )
        if setup_state == "posted":
            runner.total_api_cost += cost
        raise OSError("setup failed")

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    runner._setup_initial_program_with_metadata = setup
    with pytest.raises(OSError):
        asyncio.run(runner._generate_initial_program())
    assert runner.total_api_cost == pytest.approx(3.3)
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)


def test_initialization_checks_known_cost_before_another_attempt(runner):
    runner.total_api_cost = 4.0
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        return QueryResult(None, "msg", "sys", [], "gpt-5-mini", {}, 10, 0, cost=3.3)

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    with pytest.raises(RuntimeError):
        asyncio.run(runner._generate_initial_program())
    assert len(calls) == 1
    assert runner.total_api_cost == pytest.approx(7.3)


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_proposal_cost_survives_failure_artifact_error(runner, error_type):
    runner._update_avg_proposal_cost = lambda cost: None
    runner._classify_failed_proposal = lambda **kwargs: "proposal"

    async def artifact(**kwargs):
        raise error_type("artifact unavailable")

    runner._write_failure_artifact_async = artifact
    with pytest.raises(error_type):
        asyncio.run(
            runner._record_terminal_failed_proposal(
                generation=1,
                exec_fname="unused",
                proposal_started_at=0,
                sampling_worker_id=None,
                active_proposals_at_start=1,
                parent_program=None,
                archive_programs=[],
                top_k_programs=[],
                code_diff=None,
                meta_patch_data={},
                code_embedding=None,
                embed_cost=0.0,
                novelty_cost=0.0,
                api_costs=3.3,
                failure_stage="proposal",
                failure_reason="max_output_tokens",
            )
        )
    assert runner.total_api_cost == pytest.approx(3.3)
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
