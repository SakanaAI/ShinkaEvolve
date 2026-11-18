import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.config import EvolutionConfig
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.launch import LocalJobConfig


def _settled_runner(results_dir: Path, budget: int) -> ShinkaEvolveRunner:
    return ShinkaEvolveRunner(
        EvolutionConfig(
            results_dir=str(results_dir),
            num_generations=budget,
            embedding_model=None,
            llm_models=[],
            llm_dynamic_selection=None,
            init_program_path=None,
        ),
        LocalJobConfig(),
        DatabaseConfig(num_islands=1),
        verbose=False,
        banner_style="minimal",
    )


def test_settled_resume_rebuilds_best_after_interrupted_export(tmp_path):
    async def run():
        config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
        db = ProgramDatabase(config)
        db.add(Program(id="seed", code="x = 0\n", generation=0, correct=True))
        winner = Program(
            id="winner", code="x = 10\n", generation=1, correct=True, combined_score=10.0
        )
        db.add(winner)
        db.close()

        source = tmp_path / "gen_1"
        source.mkdir()
        (source / "main.py").write_text(winner.code)
        (source / "evidence.txt").write_text("winner evaluation")
        previous = tmp_path / ".best-interrupted" / "previous"
        previous.mkdir(parents=True)
        (previous / "main.py").write_text("x = 0\n")

        runner = _settled_runner(tmp_path, budget=2)
        runner.scheduler.run = AsyncMock()
        runner.scheduler.submit_async_nonblocking = AsyncMock()
        try:
            await asyncio.wait_for(runner.run_async(), timeout=10)

            assert runner.completed_generations == 2
            assert runner.total_proposals_generated == 0
            runner.scheduler.run.assert_not_called()
            runner.scheduler.submit_async_nonblocking.assert_not_called()
            assert (tmp_path / "best" / "main.py").read_text() == winner.code
            assert (tmp_path / "best" / "evidence.txt").read_text() == "winner evaluation"
            assert (previous / "main.py").read_text() == "x = 0\n"
        finally:
            runner.db.close()

    asyncio.run(run())
