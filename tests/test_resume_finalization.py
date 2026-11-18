import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"), num_islands=1
        )
        db = ProgramDatabase(config)
        db.add(Program(id="seed", code="x = 0\n", generation=0, correct=True))
        winner = Program(
            id="winner",
            code="x = 10\n",
            generation=1,
            correct=True,
            combined_score=10.0,
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
            assert (
                tmp_path / "best" / "evidence.txt"
            ).read_text() == "winner evaluation"
            assert (previous / "main.py").read_text() == "x = 0\n"
        finally:
            runner.db.close()

    asyncio.run(run())


def test_settled_resume_replays_deferred_children_and_leaves_pruned_rows_alone(
    tmp_path,
):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"), num_islands=1, archive_size=1
        )
        db = ProgramDatabase(config)
        for generation in range(4):
            db.add(
                Program(
                    id=f"program-{generation}",
                    code=f"x = {generation}\n",
                    generation=generation,
                    correct=True,
                    combined_score=float(generation),
                    metadata={
                        "safe_processing": True,
                        "source_generation": generation,
                        "source_job_id": f"job-{generation}",
                    },
                ),
                defer_maintenance=generation == 3,
            )
        legacy_child = db.get("program-3")
        legacy_child.metadata["pipeline_accounted_seconds"] = 0.1
        db.conn.execute(
            "UPDATE programs SET metadata = ? WHERE id = ?",
            (json.dumps(legacy_child.metadata), legacy_child.id),
        )
        very_old_child = db.get("program-1")
        inline_metadata = {
            key: value
            for key, value in very_old_child.metadata.items()
            if not key.startswith("postprocess_db_maintenance_")
        }
        db.conn.execute(
            "UPDATE programs SET metadata = ? WHERE id = ?",
            (json.dumps(inline_metadata), very_old_child.id),
        )
        db.conn.commit()
        db.add(
            Program(
                id="failed-proposal",
                code="pass",
                generation=3,
                metadata={
                    "safe_processing": False,
                    "source_job_id": "failed:proposal:3",
                },
            ),
            defer_maintenance=True,
        )
        pruned_metadata = db.get("program-2").metadata
        db.close()

        for _ in range(2):
            runner = _settled_runner(tmp_path, budget=4)
            runner.db_config.archive_size = 1
            try:
                await asyncio.wait_for(runner.run_async(), timeout=10)

                db = runner.db
                assert runner.completed_generations == 4
                assert runner.total_proposals_generated == 0
                assert db.best_program_id == "program-3"
                (archived,) = db.conn.execute(
                    "SELECT program_id FROM archive"
                ).fetchall()
                assert archived[0] == "program-3"
                assert db.get("program-3").metadata[
                    "postprocess_db_maintenance_applied"
                ]
                assert (
                    "postprocess_db_maintenance_applied"
                    not in db.get("failed-proposal").metadata
                )
                assert db.get("program-1").metadata == inline_metadata
                assert db.get("program-2").metadata == pruned_metadata
            finally:
                runner.db.close()

    asyncio.run(run())


@pytest.mark.parametrize(("threshold", "island_count"), [(0, 4), (1, 3)])
def test_interrupted_child_maintenance_replays_migration_and_preserves_spawn(
    tmp_path, monkeypatch, threshold, island_count
):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=2,
        enable_dynamic_islands=True,
        stagnation_threshold=threshold,
        migration_interval=1,
        migration_rate=0.5,
        island_elitism=False,
    )
    db = ProgramDatabase(config)
    db.add(
        Program(id="seed", generation=0, code="pass", correct=True, combined_score=10)
    )
    child = Program(
        id="child",
        generation=1,
        code="pass",
        correct=True,
        combined_score=5,
        parent_id="seed",
        metadata={"postprocess_db_maintenance_applied": False},
    )
    db.add(child, defer_maintenance=True)

    scheduled_operations = db.check_scheduled_operations

    def interrupt_after_migration():
        scheduled_operations()
        assert len(db.get("child").migration_history) == 1
        raise RuntimeError("interrupted before recording completed maintenance")

    with monkeypatch.context() as patch:
        patch.setattr(db, "check_scheduled_operations", interrupt_after_migration)
        with pytest.raises(RuntimeError, match="interrupted before recording"):
            db.run_post_add_maintenance(child)

    assert len(db.island_manager.get_island_populations()) == island_count
    assert db.best_score_generation == 1
    assert db.get("child").migration_history == []
    assert db.get("child").metadata["postprocess_db_maintenance_applied"] is False

    for _ in range(2):
        db.run_post_add_maintenance(child)
        assert len(db.island_manager.get_island_populations()) == island_count
        assert len(db.get("child").migration_history) == 1
        assert db.get("child").metadata["postprocess_db_maintenance_applied"] is True
        assert db.get("seed").children_count == 1
    db.close()


def test_replay_preserves_later_archive_decisions_after_partial_maintenance(
    tmp_path, monkeypatch
):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"), num_islands=1, archive_size=1
    )
    db = ProgramDatabase(config)
    db.add(
        Program(id="seed", generation=0, code="pass", correct=True, combined_score=0)
    )
    child = Program(
        id="child",
        generation=1,
        code="pass",
        correct=True,
        combined_score=1,
        metadata={"postprocess_db_maintenance_applied": False},
    )
    db.add(child, defer_maintenance=True)

    def interrupt_before_receipt():
        raise RuntimeError("interrupted after archive decision")

    with monkeypatch.context() as patch:
        patch.setattr(db, "check_scheduled_operations", interrupt_before_receipt)
        with pytest.raises(RuntimeError, match="interrupted after archive decision"):
            db.run_post_add_maintenance(child)

    db.add(
        Program(id="newer", generation=2, code="pass", correct=True, combined_score=2)
    )
    db.close()

    # The already-applied archive decision must survive pruning and a new size limit.
    config.archive_size = 2
    db = ProgramDatabase(config)
    db.recover_pending_child_maintenance()
    assert [row[0] for row in db.conn.execute("SELECT program_id FROM archive")] == [
        "newer"
    ]
    assert db.get("child").metadata["postprocess_db_maintenance_applied"] is True
    db.close()
