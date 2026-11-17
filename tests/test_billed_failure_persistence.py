"""Failed billed work survives resume even when no program was produced."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shinka.api_budget import APIBudgetExceeded, run_request
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.billing import BillingPersistenceError
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)
from shinka.llm.providers import QueryResult
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import EmptyResponseError
import shinka.llm.llm as llm_module


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


@pytest.mark.parametrize("commit_before_error", [False, True])
def test_transient_charge_write_retries_same_identity(runner, commit_before_error):
    async def run():
        runner.evo_config.max_api_costs = None
        runner.async_db = AsyncProgramDatabase(runner.db, max_workers=1)
        write_actual = runner.async_db.record_attempt_event_async
        provider = AsyncMock()
        identities = []

        async def write(**kwargs):
            identities.append(kwargs["details"]["api_charge_id"])
            if len(identities) == 1:
                if commit_before_error:
                    await write_actual(**kwargs)
                raise OSError("journal acknowledgement lost")

            scope, token = runner._begin_api_cost_scope()
            try:
                with pytest.raises(APIBudgetExceeded):
                    await run_request(provider, lambda _: 0.0, "gpt-5-mini")
            finally:
                await runner._finish_api_cost_scope(scope, token, "initialization", 0)
            await write_actual(**kwargs)

        runner.async_db.record_attempt_event_async = write
        try:
            await runner._record_unattached_cost(1.0, "initialization", 0)
            assert len(identities) == 2 and len(set(identities)) == 1
            provider.assert_not_awaited()
            assert not runner._paid_admission_closed()
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())


def test_unwritable_charge_is_retained_for_shutdown_and_closes_admission(runner):
    async def run():
        runner.should_stop = asyncio.Event()
        runner.finalization_complete = asyncio.Event()
        record = runner.async_db.record_attempt_event_async
        runner.async_db.record_attempt_event_async = AsyncMock(
            side_effect=OSError("disk unavailable")
        )
        with pytest.raises(BillingPersistenceError, match="generation 1"):
            await runner._record_evaluation_charge(1, "stable-id", 1.0)

        assert runner.should_stop.is_set()
        assert runner.finalization_complete.is_set()
        assert runner._paid_admission_closed()
        assert await runner._get_total_api_costs() == 0.0
        runner.async_db.record_attempt_event_async = record
        await runner._flush_unwritten_charges()
        await runner._flush_unwritten_charges()
        assert await runner._get_total_api_costs() == pytest.approx(1.0)

    asyncio.run(run())


def test_setup_restores_handoff_charge_before_seed_requests(runner, tmp_path):
    async def run():
        await runner._record_evaluation_charge(1, "already-billed", 1.0)
        runner.evo_config.max_api_costs = 1.0
        runner.results_dir = tmp_path
        runner.db_config = runner.db.config
        runner.console = None
        runner.max_db_workers = 1
        runner.enable_deadlock_debugging = False
        runner.wandb_logger = SimpleNamespace(enabled=False)
        del runner.prompt_sampler
        runner.llm = SimpleNamespace(
            query=AsyncMock(), get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
        )
        runner.db.close()
        try:
            with pytest.raises(APIBudgetExceeded):
                await runner._setup_async()
            assert runner.total_api_cost == pytest.approx(1.0)
            runner.llm.query.assert_not_awaited()
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())


def test_persisted_prompt_cost_read_errors_are_reported(runner):
    def unavailable():
        raise OSError("prompt database unavailable")

    runner.prompt_db = SimpleNamespace(get_total_evolution_costs=unavailable)
    with pytest.raises(OSError, match="prompt database unavailable"):
        asyncio.run(runner._get_total_api_costs())


def test_prompt_charges_are_restored_with_prompt_evolution_disabled(runner, tmp_path):
    db = SystemPromptDatabase(
        SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
    )
    try:
        db.add(create_system_prompt(
            prompt_text="prompt", generation=0, patch_type="init",
            metadata={"llm": {"cost": 1.25}},
        ))
    finally:
        db.close()
    assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(1.25)


def test_unreadable_meta_charge_closes_admission(runner, monkeypatch):
    async def write(program):
        raise OSError("write unavailable")

    def read(program_id):
        raise OSError("read unavailable")

    runner._persist_program_metadata_async = write
    monkeypatch.setattr(runner.db, "get", read)
    program = Program(id="seed", code="pass", generation=0)
    with pytest.raises(BillingPersistenceError, match="reconcile meta charge"):
        asyncio.run(runner._record_meta_cost(program, 1.0))
    assert runner._paid_admission_closed()


def test_exhausted_budget_does_not_send_another_initial_request(runner):
    runner.total_api_cost = 6.6

    async def query(**kwargs):
        pytest.fail("budget exhausted before initialization")

    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
    )
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(runner._generate_initial_program())


def test_initialization_can_use_an_unpriced_model_at_the_cap(runner):
    model = "local/test@http://127.0.0.1:1234/v1"
    runner.evo_config.llm_models = [model]
    runner.total_api_cost = 5.0
    query = AsyncMock(return_value=QueryResult(
        " ```python\npass\n```", "msg", "sys", [], model, {}, 1, 1
    ))
    runner.llm = SimpleNamespace(
        query=query, get_kwargs=lambda **kwargs: {"model_name": model}
    )
    runner._setup_initial_program_with_metadata = AsyncMock()
    asyncio.run(runner._generate_initial_program())
    query.assert_awaited_once()


def test_file_seed_can_start_with_no_budget_configuration(runner):
    del runner.evo_config
    runner._setup_initial_program_with_metadata = AsyncMock()
    asyncio.run(runner._setup_initial_program("pass"))
    runner._setup_initial_program_with_metadata.assert_awaited_once()


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


@pytest.mark.parametrize("boundary", ["insert", "postprocess"])
def test_cancelled_seed_persists_cumulative_embedding_charges(runner, tmp_path, boundary):
    async def run():
        committed = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def embed():
            cost = 0.4 if not requests else 0.6
            requests.append(cost)
            return [0.1], cost

        async def embedding(path):
            return await run_request(
                embed, lambda reply: reply[1], "text-embedding-3-small", "embedding"
            )

        async def insert(program, **kwargs):
            runner.db.add(program)
            if boundary == "insert":
                await stop_after_commit(program)

        async def stop_after_commit(program):
            committed.set()
            await release.wait()

        runner.results_dir = tmp_path
        runner.db_config = runner.db.config
        runner.lang_ext = "py"
        runner.max_proposal_jobs = runner.max_evaluation_jobs = runner.max_db_workers = 1
        runner.scheduler = SimpleNamespace(run=lambda *args: (None, 0.0))
        runner.meta_summarizer = None
        runner._record_progress = lambda: None
        runner._get_code_embedding_async = embedding
        runner.async_db.add_program_async = insert
        runner._persist_program_metadata_async = stop_after_commit
        task = asyncio.create_task(runner._setup_initial_program("pass\n"))
        await asyncio.wait_for(committed.wait(), 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        expected = sum(requests)
        assert expected > 0.0
        (seed,) = runner.db.get_programs_by_generation(0)
        assert seed.metadata["embed_cost"] == pytest.approx(expected)
        assert runner.total_api_cost == pytest.approx(expected)
        assert await runner._get_total_api_costs() == pytest.approx(expected)
        assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 0

    asyncio.run(run())


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


def test_initialization_retains_cost_carried_by_cancelled_query(runner, monkeypatch):
    async def run():
        billed = asyncio.Event()

        async def provider(**kwargs):
            billed.set()
            raise EmptyResponseError("no text", None, {"cost": 3.3})

        runner.llm = AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
        monkeypatch.setattr(llm_module, "query_async", provider)
        task = asyncio.create_task(runner._generate_initial_program())
        await asyncio.wait_for(billed.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert runner.total_api_cost == pytest.approx(3.3)
        assert await runner._get_total_api_costs() == pytest.approx(3.3)

    asyncio.run(run())


def test_initialization_cancellation_finishes_cost_reconciliation(runner):
    async def run():
        reading = asyncio.Event()
        release_read = asyncio.Event()
        get_cost = runner._get_total_api_costs

        async def query(**kwargs):
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
                incomplete_reason="max_output_tokens",
            )

        async def read_cost():
            reading.set()
            await release_read.wait()
            return await get_cost()

        runner.llm = SimpleNamespace(
            query=query, get_kwargs=lambda **kwargs: {"model_name": "gpt-5-mini"}
        )
        runner._get_total_api_costs = read_cost
        task = asyncio.create_task(runner._generate_initial_program())
        await asyncio.wait_for(reading.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_read.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runner.total_api_cost == pytest.approx(3.3)
        assert await get_cost() == pytest.approx(3.3)

    asyncio.run(run())
