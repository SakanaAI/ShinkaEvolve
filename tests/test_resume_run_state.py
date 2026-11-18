"""A resumed run picks up the state the interrupted session left behind."""

import asyncio
import logging
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.config import EvolutionConfig
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.llm import AsymmetricUCB


def _program(generation: int, score: float, **metadata) -> Program:
    return Program(
        id=f"g{generation}",
        code=f"value = {generation}\n",
        parent_id="g0" if generation else None,
        generation=generation,
        correct=True,
        combined_score=score,
        metadata={"patch_name": f"patch-{generation}", **metadata},
    )


def _write_run(db_path: Path, scores: list[float]) -> None:
    """A run interrupted after `len(scores) - 1` generations."""
    config = DatabaseConfig(db_path=str(db_path), num_islands=1)
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        for generation, score in enumerate(scores):
            db.add(_program(generation, score))


def _resume_runner(results_dir: Path, db_path: str | None) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.results_dir = str(results_dir)
    runner.db_config = DatabaseConfig(db_path=db_path, num_islands=1)
    runner.evo_config = EvolutionConfig(
        num_generations=10, embedding_model=None, init_program_path=None
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
    runner._log_wandb_population_progress = lambda: None
    runner._generate_initial_program = AsyncMock()
    return runner


def _setup(runner: ShinkaEvolveRunner, check=lambda: None) -> None:
    async def run():
        try:
            await runner._setup_async()
            check()
        finally:
            await runner.async_db.close_async()
            runner.db.close()

    asyncio.run(run())


# -- database file -------------------------------------------------------------


@pytest.mark.parametrize("configured", ["evolution_db.sqlite", None])
def test_resume_continues_in_a_legacy_database_file(tmp_path, configured):
    _write_run(tmp_path / "evolution_db.sqlite", [1.0, 2.0, 3.0])
    runner = _resume_runner(tmp_path, configured)

    def check():
        assert runner.db_config.db_path == str(tmp_path / "evolution_db.sqlite")
        assert runner.completed_generations == 3
        assert {p.id for p in runner.db.get_all_programs()} == {"g0", "g1", "g2"}
        runner._generate_initial_program.assert_not_awaited()

    _setup(runner, check)
    assert not (tmp_path / "programs.sqlite").exists()


def test_new_run_stores_programs_under_a_custom_db_path(tmp_path, caplog):
    runner = _resume_runner(tmp_path, "custom.sqlite")

    with caplog.at_level(logging.WARNING, logger="shinka.database.run_db_path"):
        _setup(runner)

    assert runner.db_config.db_path == str(tmp_path / "custom.sqlite")
    assert (tmp_path / "custom.sqlite").exists()
    assert not (tmp_path / "programs.sqlite").exists()
    assert caplog.records == []
    runner._generate_initial_program.assert_awaited_once()


def test_new_run_without_db_path_stores_programs_in_programs_sqlite(tmp_path):
    runner = _resume_runner(tmp_path, None)

    _setup(runner)

    assert runner.db_config.db_path == str(tmp_path / "programs.sqlite")
    assert [path.name for path in tmp_path.glob("*.sqlite")] == ["programs.sqlite"]


@pytest.mark.parametrize("configured", ["custom.sqlite", None, "renamed.sqlite"])
def test_resume_continues_in_a_custom_database_file(tmp_path, caplog, configured):
    """Also after db_path was dropped from the config or changed."""
    _write_run(tmp_path / "custom.sqlite", [1.0, 2.0, 3.0])
    runner = _resume_runner(tmp_path, configured)

    def check():
        assert runner.db_config.db_path == str(tmp_path / "custom.sqlite")
        assert runner.completed_generations == 3
        assert {p.id for p in runner.db.get_all_programs()} == {"g0", "g1", "g2"}
        runner._generate_initial_program.assert_not_awaited()

    with caplog.at_level(logging.WARNING, logger="shinka.database.run_db_path"):
        _setup(runner, check)

    assert [path.name for path in tmp_path.glob("*.sqlite")] == ["custom.sqlite"]
    warned = any("already holds the programs" in r.getMessage() for r in caplog.records)
    assert warned == (configured != "custom.sqlite")


def test_config_reused_for_a_second_run_keeps_the_custom_name(tmp_path):
    """Setup writes the resolved path back into the config object."""
    first = _resume_runner(tmp_path / "run1", "custom.sqlite")
    (tmp_path / "run1").mkdir()
    _setup(first)
    second = _resume_runner(tmp_path / "run2", None)
    (tmp_path / "run2").mkdir()
    second.db_config = first.db_config

    _setup(second)

    assert second.db_config.db_path == str(tmp_path / "run2" / "custom.sqlite")
    second._generate_initial_program.assert_awaited_once()


# -- bandit baseline -----------------------------------------------------------


def _bandit() -> AsymmetricUCB:
    return AsymmetricUCB(arm_names=["model-a", "model-b"], seed=0)


@pytest.mark.parametrize("num_children", [0, 2, 4])
def test_resume_without_saved_bandit_state_sets_baseline_from_seed(
    tmp_path, num_children
):
    """The bandit state is only saved every fifth completed generation."""
    _write_run(tmp_path / "programs.sqlite", [-5.0] + [-4.0] * num_children)
    runner = _resume_runner(tmp_path, None)
    runner.llm_selection = _bandit()

    _setup(runner)

    assert runner.llm_selection._baseline == -5.0
    # An improvement on a negative-score task is still a positive reward.
    reward, _ = runner.llm_selection.update("model-a", -4.0, baseline=-5.0)
    assert reward == 1.0


def test_resume_keeps_the_saved_bandit_state(tmp_path):
    _write_run(tmp_path / "programs.sqlite", [-5.0, -4.0, -3.0])
    saved = _bandit()
    saved.set_baseline_score(-5.0)
    saved.update("model-b", -3.0, baseline=-5.0)
    saved.save_state(tmp_path / "bandit_state.pkl")
    runner = _resume_runner(tmp_path, None)
    runner.llm_selection = _bandit()

    _setup(runner)

    assert runner.llm_selection._baseline == -5.0
    assert list(runner.llm_selection.n_completed) == list(saved.n_completed)


def test_resume_with_an_incorrect_seed_keeps_a_zero_baseline(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        seed = _program(0, -5.0)
        seed.correct = False
        db.add(seed)
        db.add(_program(1, -4.0))
    runner = _resume_runner(tmp_path, None)
    runner.llm_selection = _bandit()

    _setup(runner)

    assert runner.llm_selection._baseline == 0.0


# -- meta summarizer -----------------------------------------------------------


def _meta_summarizer() -> AsyncMetaSummarizer:
    return AsyncMetaSummarizer(MetaSummarizer(async_mode=True), None)


def test_resume_restores_the_meta_state(tmp_path):
    _write_run(tmp_path / "programs.sqlite", [1.0, 2.0, 3.0])
    interrupted = _meta_summarizer()
    interrupted.persist_state_in(str(tmp_path))
    sync = interrupted.sync_summarizer
    sync.meta_summary = "summaries"
    sync.meta_scratch_pad = "scratchpad"
    sync.meta_recommendations = "1. recommendation"
    sync.meta_recommendations_history = ["1. recommendation"]
    sync.total_programs_processed = 2
    interrupted.add_evaluated_program(_program(2, 3.0))

    runner = _resume_runner(tmp_path, None)
    runner.meta_summarizer = _meta_summarizer()
    _setup(runner)

    restored = runner.meta_summarizer
    assert restored.get_current() == ("1. recommendation", "summaries", "scratchpad")
    assert restored.total_programs_processed == 2
    assert [p.id for p in restored.evaluated_since_last_meta] == ["g2"]

    # The resumed session keeps the state file up to date as well.
    restored.add_evaluated_program(_program(3, 4.0))
    reloaded = _meta_summarizer()
    reloaded.persist_state_in(str(tmp_path))
    assert reloaded.restore_persisted_state()
    assert [p.id for p in reloaded.evaluated_since_last_meta] == ["g2", "g3"]


def test_new_run_does_not_load_meta_state(tmp_path):
    stale = _meta_summarizer()
    stale.persist_state_in(str(tmp_path))
    stale.sync_summarizer.meta_recommendations = "1. stale"
    stale.sync_summarizer.persist_state()

    runner = _resume_runner(tmp_path, None)
    runner.meta_summarizer = _meta_summarizer()
    _setup(runner)

    assert runner.meta_summarizer.get_current() == (None, None, None)
