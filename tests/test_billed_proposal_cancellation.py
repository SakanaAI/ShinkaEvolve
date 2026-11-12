"""Surplus-work cancellation retains known fees once in live and SQLite totals."""

import asyncio
from types import SimpleNamespace

import pytest

import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import EmptyResponseError


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


def start_proposal(runner, fix_mode):
    parent = Program(id="parent", code="pass", generation=0, correct=not fix_mode)

    async def sample(**kwargs):
        return parent, [], [], fix_mode

    runner.async_db.sample_with_fix_mode_async = sample
    task = asyncio.create_task(
        runner._generate_evolved_proposal(
            1, "task", "unused", "unused", None, None, None, 0, None, 1
        )
    )
    runner.active_proposal_tasks["task"] = task
    return task


def assert_recorded_once(runner, expected):
    assert runner.total_api_cost == pytest.approx(expected)
    assert runner.db.conn.execute("SELECT COUNT(*) FROM attempt_log").fetchone()[0] == 1


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
                await asyncio.Event().wait()
            first_retry = boundary in {"retry_delay", "next_request"}
            prior = has_prior_cost and len(calls) == 1
            if boundary == "retry_delay":
                reached.set()
            raise EmptyResponseError(
                "no text",
                None if first_retry or prior else "max_output_tokens",
                {"cost": 0.5 if prior else 3.55},
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
        await runner._cancel_surplus_inflight_work()
        assert task.cancelled()
        expected = 4.05 if has_prior_cost else 3.55
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
