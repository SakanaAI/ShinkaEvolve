"""Surplus-work cancellation retains known fees once in live and SQLite totals."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

import shinka.core.async_runner as runner_module
import shinka.embed.embedding as embedding_module
import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers import QueryResult
from shinka.llm.providers.errors import EmptyResponseError
from shinka.llm.providers.local_openai import query_local_openai_async
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.openai import query_openai_async
from shinka.pricing.catalog import get_catalog


@pytest.fixture
def runner(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db = db
    runner.db_config = db.config
    runner.prompt_db = None
    runner.evo_config = EvolutionConfig(max_patch_attempts=1, max_api_costs=5)
    runner.llm_selection = None
    runner.verbose = False
    runner.total_api_cost = 0.0
    runner.max_proposal_jobs = runner.max_evaluation_jobs = runner.max_db_workers = 1
    runner.failed_jobs_for_retry = {}
    runner.active_proposal_tasks = {}
    runner.running_jobs = []
    runner.slot_available = asyncio.Event()
    runner.llm = AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
    runner.prompt_sampler = SimpleNamespace(
        task_sys_msg="task",
        sample=lambda **kwargs: ("sys", "patch", "diff"),
        sample_fix=lambda **kwargs: ("sys", "patch", "diff"),
    )
    runner._get_current_system_prompt = lambda: (None, None)
    runner._update_avg_proposal_cost = lambda cost: None

    async def record(**kwargs):
        db.record_attempt_event(**kwargs)

    async def artifact(**kwargs):
        return "unused", {}

    runner.async_db = SimpleNamespace(record_attempt_event_async=record)
    runner._write_failure_artifact_async = artifact
    yield runner
    db.close()


def start_proposal(runner, fix_mode, generation=1):
    parent = Program(id="parent", code="pass", generation=0, correct=not fix_mode)

    async def sample(**kwargs):
        return parent, [], [], fix_mode

    runner.async_db.sample_with_fix_mode_async = sample
    task_id = f"task-{generation}"
    task = asyncio.create_task(
        runner._generate_evolved_proposal(
            generation, task_id, "unused", "unused", None, None, None, 0, None, 1
        )
    )
    runner.active_proposal_tasks[task_id] = task
    return task


def assert_recorded_once(runner, expected):
    assert runner.total_api_cost == pytest.approx(expected)
    assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1


def test_run_cancellation_closes_paid_admission_before_draining(runner, monkeypatch):
    async def run():
        producer_started = asyncio.Event()
        draining = asyncio.Event()
        release = asyncio.Event()
        runner.pricing_snapshot = get_catalog()
        runner.finalization_complete = asyncio.Event()
        runner.should_stop = asyncio.Event()
        runner.meta_summarizer = None
        runner._setup_async = AsyncMock()
        runner._verify_database_ready = AsyncMock()
        runner._cleanup_async = AsyncMock()

        async def produce():
            producer_started.set()
            await asyncio.Event().wait()

        async def drain():
            draining.set()
            await release.wait()

        runner._proposal_coordinator_task = produce
        runner._job_monitor_task = asyncio.Event().wait
        runner._cancel_completed_job_batches = drain
        provider = AsyncMock()
        monkeypatch.setattr(llm_module, "query_async", provider)

        task = asyncio.create_task(runner._run_async())
        await asyncio.wait_for(producer_started.wait(), 2)
        task.cancel()
        await asyncio.wait_for(draining.wait(), 2)
        try:
            assert runner.should_stop.is_set()
            assert await start_proposal(runner, False) is None
            provider.assert_not_awaited()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


@pytest.mark.parametrize("fix_mode", [False, True])
@pytest.mark.parametrize("client_retries", [False, True])
def test_proposal_stops_fresh_requests_at_cost_limit(
    runner, monkeypatch, fix_mode, client_retries
):
    async def run():
        runner.evo_config.max_api_costs = 1.0
        runner.evo_config.max_patch_attempts = 3
        runner.evo_config.max_patch_resamples = 3
        runner.evo_config.max_novelty_attempts = 2
        calls = 0

        async def provider(**kwargs):
            nonlocal calls
            calls += 1
            raise EmptyResponseError("no text", None, {"cost": 0.60})

        async def noop(*args, **kwargs):
            pass

        monkeypatch.setattr(llm_module, "MAX_RETRIES", 2 if client_retries else 1)
        monkeypatch.setattr(llm_module.asyncio, "sleep", noop)
        monkeypatch.setattr(llm_module, "query_async", provider)
        runner._save_patch_attempt_async = noop
        runner._write_failure_artifact_async = AsyncMock(return_value=("unused", {}))

        assert await start_proposal(runner, fix_mode) is None
        assert calls == 2
        assert_recorded_once(runner, 1.20)
        assert await runner._get_total_api_costs() == pytest.approx(1.20)
        assert runner._get_generations_without_program_due_to_proposal_failure() == []
        runner._write_failure_artifact_async.assert_not_awaited()

    asyncio.run(run())


def test_running_requests_drain_while_fresh_requests_see_pending_cost(
    runner, monkeypatch
):
    async def run():
        runner.evo_config.max_api_costs = 1.0
        runner.evo_config.max_patch_attempts = 2
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        waiting_to_record = asyncio.Event()
        finish_recording = asyncio.Event()
        calls = []

        async def provider(**kwargs):
            index = len(calls)
            calls.append(kwargs)
            started[index].set()
            await release[index].wait()
            raise EmptyResponseError(
                "no text", "max_output_tokens", {"cost": 1.0 if index == 0 else 0.60}
            )

        async def save(**kwargs):
            if kwargs["generation"] == 1:
                waiting_to_record.set()
                await finish_recording.wait()

        monkeypatch.setattr(llm_module, "query_async", provider)
        runner._save_patch_attempt_async = save
        first = start_proposal(runner, False)
        second = start_proposal(runner, False, generation=2)
        await asyncio.wait_for(started[1].wait(), 2)
        release[0].set()
        await asyncio.wait_for(waiting_to_record.wait(), 2)

        assert runner.total_api_cost == 0.0
        assert await start_proposal(runner, False, generation=3) is None
        assert len(calls) == 2
        assert runner.db.conn.execute(
            "SELECT COUNT(*) FROM attempt_log WHERE generation = 3"
        ).fetchone()[0] == 0
        release[1].set()
        finish_recording.set()
        await asyncio.gather(first, second)
        assert runner.total_api_cost == pytest.approx(1.60)
        assert await runner._get_total_api_costs() == pytest.approx(1.60)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("model", "query_provider"),
    [
        ("gpt-5-mini", query_openai_async),
        ("local/gpt-5-mini@http://127.0.0.1:1234/v1", query_local_openai_async),
    ],
)
def test_transport_retry_rechecks_cost_after_backoff(
    runner, monkeypatch, model, query_provider
):
    async def run():
        runner.evo_config.max_api_costs = 1.0
        runner.llm = AsyncLLMClient(model_names=model, verbose=False)
        waiting_to_retry = asyncio.Event()
        resume_retry = asyncio.Event()
        requests = []

        def reply(request):
            requests.append(request)
            return httpx.Response(500, json={"error": {"message": "retry"}})

        async def backoff_sleep(seconds):
            waiting_to_retry.set()
            await resume_retry.wait()

        async with openai.AsyncOpenAI(
            api_key="mock",
            max_retries=3,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
        ) as client:

            async def provider(**kwargs):
                resolved = resolve_model_backend(kwargs.pop("model_name"))
                return await query_provider(client, resolved.api_model_name, **kwargs)

            monkeypatch.setattr(llm_module.asyncio, "sleep", backoff_sleep)
            monkeypatch.setattr(llm_module, "query_async", provider)
            task = start_proposal(runner, False)
            await asyncio.wait_for(waiting_to_retry.wait(), 2)
            assert len(requests) == 1
            runner.total_api_cost = 1.0
            resume_retry.set()
            assert await task is None
            assert await start_proposal(runner, False, generation=2) is None
            assert len(requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["pending", "persisted", "submit_failed"])
def test_successful_proposal_can_submit_at_cap_without_paid_embedding(
    tmp_path, runner, monkeypatch, outcome
):
    async def run():
        runner.evo_config.max_api_costs = 1.0
        runner.results_dir = tmp_path
        runner.novelty_judge = None
        runner.submitted_jobs = {}
        candidate = tmp_path / "candidate.py"
        candidate.write_text("pass", encoding="utf-8")

        async def apply_patch(**kwargs):
            return "pass", 1, str(candidate), None, "patch", None

        async def provider(msg, system_msg, model_name, **kwargs):
            return QueryResult(
                "patch", msg, system_msg, [], model_name, {}, 1, 1, cost=1.0
            )

        async def noop(**kwargs):
            pass

        async def submit(**kwargs):
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            if outcome == "submit_failed":
                raise OSError("scheduler rejected job")
            return "job", 1, 0.0, 0.0, 1

        async def failure_artifact(**kwargs):
            # Resume can recover an artifact without a journal entry. The
            # already billed launch failure must be terminal before that point.
            row = runner.db.conn.execute("SELECT status FROM attempt_log").fetchone()
            assert row["status"] == "failed"
            assert await runner._get_total_api_costs() == pytest.approx(1.0)
            return "unused", {}

        def unexpected_embedding(**kwargs):
            pytest.fail("paid embedding admitted")

        transport = SimpleNamespace(
            with_options=unexpected_embedding,
            embeddings=SimpleNamespace(create=unexpected_embedding),
        )
        monkeypatch.setattr(
            embedding_module,
            "get_async_client_embed",
            lambda model: (transport, model),
        )
        runner.embedding_client = embedding_module.AsyncEmbeddingClient()
        monkeypatch.setattr(llm_module, "query_async", provider)
        monkeypatch.setattr(runner_module, "apply_patch_async", apply_patch)
        runner._submit_evaluation_job_with_slot = submit
        runner._save_patch_attempt_async = noop
        runner._write_failure_artifact_async = failure_artifact

        parent = Program(id="parent", code="pass", generation=0, correct=True)

        async def sample(**kwargs):
            return parent, [], [], False

        runner.async_db.sample_with_fix_mode_async = sample
        job = await runner._generate_evolved_proposal(
            1, "task", str(candidate), str(tmp_path), None, None, None, 0, None, 1
        )
        if outcome == "submit_failed":
            assert job is None
        else:
            assert job is not None
            assert job.meta_patch_data["api_costs"] == 1.0
            assert job.embed_cost == 0.0
        assert runner.total_api_cost == 1.0
        if outcome == "persisted":
            assert job is not None
            runner.db.add(
                Program(
                    id="candidate",
                    generation=1,
                    code="pass",
                    metadata={**job.meta_patch_data, "source_job_id": "job"},
                )
            )
        assert await runner._get_total_api_costs() == pytest.approx(1.0)
        assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[
            0
        ] == 1

    asyncio.run(run())


@pytest.mark.parametrize("fix_mode", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "artifact",
        "terminal_artifact",
        "retry_delay",
        "next_request",
        "prior_attempt",
        "prior_resample",
    ],
)
def test_surplus_cancellation_preserves_known_fees(
    runner, monkeypatch, fix_mode, boundary
):
    async def run():
        reached = asyncio.Event()
        finish_request = asyncio.Event()
        real_sleep = asyncio.sleep
        calls = []
        has_prior_cost = boundary in {"prior_attempt", "prior_resample"}
        if has_prior_cost:
            monkeypatch.setattr(llm_module, "MAX_RETRIES", 1)
        if boundary == "prior_attempt":
            runner.evo_config.max_patch_attempts = 2

        async def provider(**kwargs):
            calls.append(kwargs)
            if boundary == "next_request" and len(calls) == 2:
                reached.set()
                await finish_request.wait()
            first_retry = boundary in {"retry_delay", "next_request"}
            prior = has_prior_cost and len(calls) == 1
            finishing_request = boundary == "next_request" and len(calls) == 2
            if boundary == "retry_delay":
                reached.set()
            raise EmptyResponseError(
                "no text",
                None if first_retry or prior else "max_output_tokens",
                {"cost": 0.5 if prior or finishing_request else 3.55},
            )

        async def save(**kwargs):
            if (
                boundary != "terminal_artifact"
                and kwargs["response"].incomplete_reason == "max_output_tokens"
            ):
                reached.set()
                await asyncio.Event().wait()

        async def artifact(**kwargs):
            reached.set()
            await asyncio.Event().wait()

        async def no_sleep(seconds):
            pass

        runner._save_patch_attempt_async = save
        if boundary == "terminal_artifact":
            runner._write_failure_artifact_async = artifact
        monkeypatch.setattr(llm_module, "query_async", provider)
        if boundary == "next_request":
            monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)
        task = start_proposal(runner, fix_mode)
        await asyncio.wait_for(reached.wait(), 2)
        if boundary == "next_request":
            cancellation = asyncio.create_task(runner._cancel_surplus_inflight_work())
            # Wait for cancellation to reach the provider shield.
            await real_sleep(0)
            await real_sleep(0)
            assert not task.done()
            finish_request.set()
            await cancellation
        else:
            await runner._cancel_surplus_inflight_work()
        assert task.cancelled()
        expected = 4.05 if has_prior_cost or boundary == "next_request" else 3.55
        assert_recorded_once(runner, expected)
        assert await runner._get_total_api_costs() == pytest.approx(expected)
        assert runner.prompt_sampler.task_sys_msg == "task"
        assert len(calls) == (2 if has_prior_cost or boundary == "next_request" else 1)

    asyncio.run(run())


@pytest.mark.parametrize("commit_before_cancel", [False, True])
@pytest.mark.parametrize("boundary", ["cancelled_journal", "terminal_journal"])
def test_cancellation_during_journal_finishes_same_insert(
    runner, monkeypatch, commit_before_cancel, boundary
):
    async def run():
        artifact_started = asyncio.Event()
        journal_started = asyncio.Event()
        release_journal = asyncio.Event()

        async def provider(**kwargs):
            raise EmptyResponseError("no text", "max_output_tokens", {"cost": 3.55})

        async def save(**kwargs):
            if boundary == "cancelled_journal":
                artifact_started.set()
                await asyncio.Event().wait()

        async def record(**kwargs):
            if commit_before_cancel:
                runner.db.record_attempt_event(**kwargs)
            journal_started.set()
            await release_journal.wait()
            if not commit_before_cancel:
                runner.db.record_attempt_event(**kwargs)

        runner.async_db.record_attempt_event_async = record
        runner._save_patch_attempt_async = save
        monkeypatch.setattr(llm_module, "query_async", provider)
        task = start_proposal(runner, False)
        if boundary == "cancelled_journal":
            await asyncio.wait_for(artifact_started.wait(), 2)
            task.cancel()
        await asyncio.wait_for(journal_started.wait(), 2)
        task.cancel()
        # Let cancellation reach the journal await, then cancel again.
        await asyncio.sleep(0)
        task.cancel()
        release_journal.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert_recorded_once(runner, 3.55)
        assert await runner._get_total_api_costs() == pytest.approx(3.55)

    asyncio.run(run())
