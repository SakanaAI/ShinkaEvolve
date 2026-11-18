import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from threading import Event
from unittest.mock import AsyncMock

import pytest

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.llm import AsyncLLMClient


class RecordingExecutor(ThreadPoolExecutor):
    def __init__(self):
        super().__init__()
        self.submissions: list[Future] = []

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        self.submissions.append(future)
        return future


@pytest.mark.parametrize("old_finishes", ["before_retry", "during_retry", "after_retry"])
def test_cancelled_seed_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_finishes: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    query = AsyncMock(side_effect=AssertionError("seed run must stay offline"))
    monkeypatch.setattr(AsyncLLMClient, "query", query)
    results_dir = tmp_path / "run"
    public = results_dir / "gen_0" / "results"

    def make_runner():
        return ShinkaEvolveRunner(
            EvolutionConfig(
                num_generations=1,
                results_dir=str(results_dir),
                embedding_model=None,
                llm_models=["gpt-5-mini"],
                llm_dynamic_selection=None,
            ),
            LocalJobConfig(),
            DatabaseConfig(num_islands=1),
            init_program_str="pass\n",
            verbose=False,
        )

    async def run():
        loop = asyncio.get_running_loop()
        executor = RecordingExecutor()
        loop.set_default_executor(executor)
        previous, retry = make_runner(), make_runner()
        previous_started, retry_started = asyncio.Event(), asyncio.Event()
        release_previous, release_retry = Event(), Event()
        attempts: dict[int, Path] = {}

        def evaluator(started, release, score):
            def evaluate(_program, output):
                path = Path(output)
                attempts[score] = path
                loop.call_soon_threadsafe(started.set)
                assert release.wait(10), "test did not release evaluator"
                correct, metrics = {"correct": True}, {"combined_score": score}
                (path / "correct.json").write_text(json.dumps(correct))
                (path / "metrics.json").write_text(json.dumps(metrics))
                return {"correct": correct, "metrics": metrics}, 0.0

            return evaluate

        previous.scheduler.run = evaluator(previous_started, release_previous, 1)
        retry.scheduler.run = evaluator(retry_started, release_retry, 2)
        previous_run = asyncio.create_task(previous.run_async())
        retry_run = None
        try:
            await asyncio.wait_for(previous_started.wait(), 5)
            # Keep the worker's future so the test can await its actual completion.
            previous_worker = asyncio.wrap_future(executor.submissions[-1])
            previous_run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(previous_run, 5)

            async def finish_previous():
                release_previous.set()
                await asyncio.wait_for(previous_worker, 5)
                assert (attempts[1] / "metrics.json").is_file()

            if old_finishes == "before_retry":
                await finish_previous()
                assert not public.exists()

            retry_run = asyncio.create_task(retry.run_async())
            await asyncio.wait_for(retry_started.wait(), 5)
            if old_finishes == "during_retry":
                await finish_previous()

            release_retry.set()
            await asyncio.wait_for(retry_run, 10)
            if old_finishes == "after_retry":
                await finish_previous()

            with sqlite3.connect(results_dir / "programs.sqlite") as db:
                [(generation, correct, score, metadata)] = db.execute(
                    "SELECT generation, correct, combined_score, metadata FROM programs"
                ).fetchall()
                assert (generation, correct, score) == (0, 1, 2.0), metadata
                assert db.execute("SELECT COUNT(*) FROM attempt_log").fetchone() == (0,)
            assert retry.total_api_cost == 0.0
            assert json.loads((public / "correct.json").read_text()) == {"correct": True}
            assert json.loads((public / "metrics.json").read_text()) == {
                "combined_score": 2
            }
        finally:
            release_previous.set()
            release_retry.set()
            for task in (previous_run, retry_run):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    query.assert_not_awaited()
