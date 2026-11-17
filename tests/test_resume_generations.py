import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.config import EvolutionConfig
from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _resume_runner(tmp_path: Path, db_config: DatabaseConfig, budget: int):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.results_dir = str(tmp_path)
    runner.db_config = db_config
    runner.evo_config = EvolutionConfig(
        num_generations=budget, embedding_model=None, init_program_path=None
    )
    runner.console = None
    runner.max_db_workers = 1
    runner.enable_deadlock_debugging = False
    runner.wandb_logger = SimpleNamespace(enabled=False)
    runner.verbose = False
    runner.next_generation_to_submit = 1
    runner.completed_generations = 0
    runner.max_proposal_jobs = 4
    runner.assigned_generations = set()
    runner.active_proposal_tasks = {}
    runner.total_proposals_generated = 0
    runner.meta_summarizer = None
    runner.llm_selection = None
    runner.lang_ext = "py"
    runner._get_total_api_costs = AsyncMock(return_value=0.0)
    runner._load_bandit_state = lambda: None
    runner._log_wandb_population_progress = lambda: None
    runner._generate_initial_program = AsyncMock()
    return runner


@pytest.mark.parametrize(("score", "expected_baseline"), [(0.5, 0.5), (None, 0.0)])
def test_resume_seed_only_database_does_not_generate_another_seed(
    tmp_path, score, expected_baseline
):
    async def run():
        results_dir = tmp_path / "results#1"
        config = DatabaseConfig(db_path=str(results_dir / "programs.sqlite"))
        db = ProgramDatabase(config)
        db.add(
            Program(
                id="seed", code="pass", generation=0, correct=True, combined_score=score
            )
        )
        assert db.last_iteration == 0
        db.close()

        runner = _resume_runner(results_dir, config, budget=2)
        runner.llm_selection = SimpleNamespace(set_baseline_score=Mock())
        try:
            await runner._setup_async()
            await runner._verify_database_ready()

            runner._generate_initial_program.assert_not_awaited()
            runner.llm_selection.set_baseline_score.assert_called_once_with(
                expected_baseline
            )
            assert runner.completed_generations == 1
            assert runner._get_remaining_generation_slots() == 1
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


@pytest.mark.parametrize("existing_copy", [False, True])
def test_resume_finishes_seed_islands_before_sampling(
    tmp_path, monkeypatch, existing_copy
):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"), archive_size=2
        )
        db = ProgramDatabase(config)
        seed = Program(
            id="seed", code="pass", generation=0, correct=True, combined_score=1.0
        )
        db.add(seed, defer_maintenance=True)
        if existing_copy:
            db._update_archive(seed)
            db.add(
                Program(
                    id="later",
                    code="pass",
                    generation=1,
                    correct=True,
                    combined_score=0.5,
                )
            )
        copy_ids = (
            db.island_manager.copy_program_to_islands(seed) if existing_copy else []
        )
        db.close()

        for _ in range(2):
            runner = _resume_runner(tmp_path, config, budget=3)
            try:
                await runner._setup_async()
                sampled_islands = []
                for island in (0, 1):
                    monkeypatch.setattr(
                        "shinka.database.island_sampler.UniformIslandSampler.sample_island",
                        lambda sampler, initialized: sorted(initialized)[island],
                    )
                    parent, *_ = await runner.async_db.sample_with_fix_mode_async()
                    sampled_islands.append(parent.island_idx)
                assert sampled_islands == [0, 1]
                programs = runner.db.get_programs_by_generation(0)
                assert len(programs) == 2
                assert "_needs_island_copies" not in runner.db.get("seed").metadata
                recovered_ids = [p.id for p in programs if p.island_idx == 1]
                assert not copy_ids or copy_ids == recovered_ids
                copy_ids = recovered_ids
            finally:
                await runner.async_db.close_async()
                runner.db.close()

    asyncio.run(run())


def test_resume_preserves_configured_seed_across_interruption_after_archive(tmp_path):
    async def run():
        source = tmp_path / "gen_0" / "main.py"
        source.parent.mkdir()
        source.write_text("pass\n")
        config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"))
        interrupted = _resume_runner(tmp_path, config, budget=1)
        interrupted.evo_config.init_program_path = str(source)
        interrupted._setup_initial_program = AsyncMock(
            side_effect=RuntimeError("interrupted")
        )
        try:
            with pytest.raises(RuntimeError, match="interrupted"):
                await interrupted._setup_async()
        finally:
            await interrupted.async_db.close_async()
            interrupted.db.close()
        assert not source.exists()

        runner = _resume_runner(tmp_path, config, budget=1)
        runner.evo_config.init_program_path = str(source)
        runner._setup_initial_program = AsyncMock()
        try:
            await runner._setup_async()

            runner._setup_initial_program.assert_awaited_once_with("pass\n")
            runner._generate_initial_program.assert_not_awaited()
            (archived,) = tmp_path.rglob("main.py")
            assert archived.read_text() == "pass\n"
            assert archived != source
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


def test_resume_retries_gap_and_preserves_interrupted_artifacts(tmp_path):
    async def run():
        config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"))
        db = ProgramDatabase(config)
        db.add(Program(id="seed", code="pass", generation=0))
        db.add(Program(id="later", code="pass", generation=2, correct=False))
        db.close()

        interrupted = tmp_path / "gen_1"
        (interrupted / "results").mkdir(parents=True)
        (interrupted / "main.py").write_text("old candidate")
        (interrupted / "results" / "partial.txt").write_text("old evaluation")
        (interrupted / ".generation_lock").touch()

        runner = _resume_runner(tmp_path, config, budget=3)
        submitted = []

        async def persist_proposal(generation, task_id, exec_fname, results_dir, *_):
            submitted.append(generation)
            assert not list(Path(results_dir).iterdir())
            Path(exec_fname).write_text("new candidate")
            await runner.async_db.add_program_async(
                Program(id="recovered", code="pass", generation=generation)
            )

        runner._generate_evolved_proposal = persist_proposal
        try:
            await runner._setup_async()
            await runner._start_proposals(4)
            await asyncio.gather(*runner.active_proposal_tasks.values())

            assert submitted == [1]
            assert runner._get_remaining_generation_slots() == 0
            assert await runner._count_completed_generations_from_db() == 3
            assert (interrupted / "main.py").read_text() == "new candidate"
            (archived,) = tmp_path.rglob("partial.txt")
            assert archived.read_text() == "old evaluation"
            assert (archived.parent.parent / "main.py").read_text() == "old candidate"
            assert not (interrupted / ".generation_lock").exists()
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


def test_resume_counts_only_distinct_budgeted_originals_with_dynamic_islands(
    tmp_path, monkeypatch
):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"),
            num_islands=2,
            island_spawn_strategy="best",
        )
        db = ProgramDatabase(config)
        db.add(Program(id="seed", code="pass", generation=0, correct=True))
        db.add(
            Program(
                id="child",
                code="pass",
                generation=2,
                parent_id="seed",
                correct=True,
                combined_score=1.0,
                metadata={"similarity": float("nan")},
            )
        )
        assert db.island_manager.spawn_new_island()
        db.add(
            Program(id="duplicate", code="pass", generation=0), defer_maintenance=True
        )
        db.add(Program(id="outside-budget", code="pass", generation=9))
        # Model a retained island copy whose original is no longer available.
        db.conn.execute("DELETE FROM programs WHERE id = 'child'")
        db.conn.commit()
        (copy,) = db.get_programs_by_generation(2)
        assert copy.metadata["_spawned_island"] is True
        db.close()

        runner = _resume_runner(tmp_path, config, budget=4)
        try:
            await runner._setup_async()

            assert runner.completed_generations == 1
            assert await runner._get_missing_persisted_generations() == [1, 2, 3]
            assert runner._get_remaining_generation_slots() == 3

            query_rows = runner.async_db._query_generation_rows_async
            fail_next_read = False

            async def check_incremental_query(query, params=()):
                nonlocal fail_next_read
                if fail_next_read:
                    fail_next_read = False
                    raise sqlite3.OperationalError("transient read failure")
                plan = await query_rows("EXPLAIN QUERY PLAN " + query, params)
                assert any("rowid>?" in row[3] for row in plan), plan
                return await query_rows(query, params)

            monkeypatch.setattr(
                runner.async_db, "_query_generation_rows_async", check_incremental_query
            )
            with pytest.raises(sqlite3.IntegrityError):
                await runner.async_db.add_program_async(
                    Program(id="seed", code="pass", generation=1),
                    defer_maintenance=True,
                )
            assert await runner._count_completed_generations_from_db() == 1

            await runner.async_db.add_program_async(
                Program(id="failed-eval", code="pass", generation=3, correct=False),
                defer_maintenance=True,
            )
            fail_next_read = True
            assert await runner._count_completed_generations_from_db() == 2
            with monkeypatch.context() as patch:
                patch.setattr(
                    ProgramDatabase,
                    "run_post_add_maintenance",
                    Mock(side_effect=RuntimeError("maintenance interrupted")),
                )
                with pytest.raises(RuntimeError, match="maintenance interrupted"):
                    await runner.async_db.add_program_async(
                        Program(id="recovered", code="pass", generation=2)
                    )
            assert await runner._count_completed_generations_from_db() == 3
            assert await runner._count_completed_generations_from_db() == 3
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


@pytest.mark.parametrize("retry_code", ["x = 1\n", "x = 10\n"])
def test_best_export_uses_selected_copy_code_after_its_generation_is_retried(
    tmp_path, monkeypatch, retry_code
):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"),
            num_islands=1,
            island_spawn_strategy="best",
        )
        db = ProgramDatabase(config)
        db.add(
            Program(
                id="seed",
                code="x = 0\n",
                generation=0,
                correct=True,
                combined_score=0.0,
            )
        )
        db.add(
            Program(
                id="source",
                code="x = 10\n",
                generation=2,
                correct=True,
                combined_score=10.0,
            )
        )
        assert db.island_manager.spawn_new_island()
        db.conn.execute("DELETE FROM programs WHERE id = 'source'")
        db.conn.commit()
        (selected,) = db.get_programs_by_generation(2)
        db.close()

        best = tmp_path / "best"
        best.mkdir()
        (best / "main.py").write_text("old best")
        runner = _resume_runner(tmp_path, config, budget=3)
        runner.best_program_id = "seed"
        try:
            await runner._setup_async()
            retry_dir = tmp_path / "gen_2"
            retry_dir.mkdir()
            (retry_dir / "main.py").write_text(retry_code)
            (retry_dir / "retry-only.txt").touch()
            await runner.async_db.add_program_async(
                Program(
                    id="retry",
                    code=retry_code,
                    generation=2,
                    correct=True,
                    combined_score=1.0,
                )
            )
            write_text = Path.write_text

            def fail_replacement(path, data, **kwargs):
                if data == selected.code:
                    raise OSError("could not stage selected code")
                return write_text(path, data, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(Path, "write_text", fail_replacement)
                with pytest.raises(OSError, match="could not stage selected code"):
                    await runner._update_best_solution_async()
            assert runner.best_program_id == "seed"
            assert (best / "main.py").read_text() == "old best"

            await runner._update_best_solution_async()

            assert runner.best_program_id == selected.id
            assert (best / "main.py").read_text() == selected.code
            assert not (best / "retry-only.txt").exists()
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


def test_resume_does_not_reschedule_terminal_proposal_failures(tmp_path):
    async def run():
        config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"))
        db = ProgramDatabase(config)
        db.add(Program(id="seed", code="pass", generation=0))
        db.add(Program(id="persisted-failure", code="pass", generation=4))
        db.record_attempt_event(
            2,
            "proposal",
            "failed",
            {"node_kind": "failed_proposal", "max_similarity": float("nan")},
        )
        db.close()

        failure = tmp_path / "gen_2" / "failure.json"
        failure.parent.mkdir()
        failure.write_text("terminal artifact")
        runner = _resume_runner(tmp_path, config, budget=5)
        runner._generate_proposal_async = AsyncMock()
        try:
            await runner._setup_async()
            await runner._start_proposals(4)
            await asyncio.gather(*runner.active_proposal_tasks.values())

            assert [
                call.args[0] for call in runner._generate_proposal_async.await_args_list
            ] == [1, 3]
            assert runner.completed_generations == 2
            assert runner._get_remaining_generation_slots() == 0
            assert failure.read_text() == "terminal artifact"
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


def test_resume_journals_orphan_terminal_failure_once_before_budget_check(tmp_path):
    async def run():
        config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"))
        writer = _resume_runner(tmp_path, config, budget=4)
        failure_dir = tmp_path / "gen_1"
        failure_dir.mkdir()
        _, payload = await writer._write_failure_artifact_async(
            generation=1,
            exec_fname=str(failure_dir / "main.py"),
            parent_program=None,
            archive_programs=[],
            top_k_programs=[],
            meta_patch_data=None,
            embed_cost=0.2,
            novelty_cost=0.0,
            api_costs=0.8,
            failure_stage="proposal",
            failure_class="proposal_generation_failed",
            failure_reason="No applicable patch",
            code_embedding=None,
            proposal_started_at=0.0,
            sampling_worker_id=None,
            active_proposals_at_start=1,
        )
        invalid_dir = tmp_path / "gen_2"
        invalid_dir.mkdir()
        (invalid_dir / "failure.json").write_text(json.dumps(payload))
        durable_dir = tmp_path / "gen_3"
        durable_dir.mkdir()
        durable_failure = durable_dir / "failure.json"
        durable_failure.write_text(json.dumps({**payload, "generation": 3}))
        historical_failure = tmp_path / "gen_9" / "failure.json"
        historical_failure.parent.mkdir()
        historical_failure.write_text(json.dumps({**payload, "generation": 9}))
        db = ProgramDatabase(config)
        db.add(Program(id="stored", code="pass", generation=3), defer_maintenance=True)
        db.add(
            Program(
                id="copy", code="pass", generation=1, metadata={"_spawned_island": True}
            ),
            defer_maintenance=True,
        )
        db.close()

        for _ in range(2):
            runner = _resume_runner(tmp_path, config, budget=4)
            runner.evo_config.max_api_costs = 1.0
            runner.prompt_db = None
            del runner._get_total_api_costs
            del runner._generate_initial_program
            try:
                with pytest.raises(RuntimeError, match="API cost budget exhausted"):
                    await runner._setup_async()

                assert runner.total_api_cost == pytest.approx(2.0)
                event, historical = runner.db.conn.execute(
                    "SELECT generation, stage, details FROM attempt_log ORDER BY generation"
                ).fetchall()
                assert (event["generation"], event["stage"]) == (1, "proposal")
                assert historical["generation"] == 9
                details = json.loads(event["details"])
                assert details["recovered_failure_json_path"] == str(
                    failure_dir / "failure.json"
                )
                assert (failure_dir / "failure.json").is_file()
                assert durable_failure.is_file()
                assert not invalid_dir.exists()
            finally:
                await runner.async_db.close_async()
                runner.db.close()

    asyncio.run(run())
