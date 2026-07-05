"""Test dynamic island spawning on stagnation."""

import tempfile
from pathlib import Path
from shinka.database import DatabaseConfig, ProgramDatabase, Program


def test_stagnation_detection():
    """Test that stagnation is correctly detected."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_stagnation.db"

        config = DatabaseConfig(
            db_path=str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=5,  # Very short threshold for testing
        )

        db = ProgramDatabase(config=config, embedding_model="", read_only=False)

        # Add initial program (generation 0)
        initial_program = Program(
            id="initial_prog",
            code="def initial(): return 0",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
        )
        db.add(initial_program)

        # At generation 0, should not be stagnant
        assert not db.is_stagnant(0), "Should not be stagnant at generation 0"
        assert not db.is_stagnant(4), (
            "Should not be stagnant at generation 4 (under threshold)"
        )

        # At generation 5+, should be stagnant (5 gens without improvement)
        assert db.is_stagnant(5), "Should be stagnant at generation 5 (met threshold)"
        assert db.is_stagnant(10), (
            "Should be stagnant at generation 10 (past threshold)"
        )

        db.close()
        print("✓ Stagnation detection test passed!")


def test_dynamic_island_spawning():
    """Test that new islands are spawned when stagnation is detected."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_spawn.db"

        config = DatabaseConfig(
            db_path=str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=3,  # Very short threshold for testing
        )

        db = ProgramDatabase(config=config, embedding_model="", read_only=False)

        # Add initial program (generation 0)
        initial_program = Program(
            id="initial_prog",
            code="def initial(): return 0",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
        )
        db.add(initial_program)

        # Get initial island count
        initial_islands = db.island_manager.get_island_populations()
        print(f"Initial islands: {initial_islands}")

        # Add programs without improvement until stagnation
        for gen in range(1, 5):
            program = Program(
                id=f"prog_gen_{gen}",
                code=f"def test(): return {gen}",
                correct=True,
                combined_score=0.5,  # Lower than initial, no improvement
                generation=gen,
                island_idx=0,
            )
            db.add(program)

        # Check that a new island was spawned
        final_islands = db.island_manager.get_island_populations()
        print(f"Final islands: {final_islands}")

        # Should have more islands now (at least one spawned)
        assert len(final_islands) > len(initial_islands), (
            f"Expected new island to be spawned. Initial: {initial_islands}, Final: {final_islands}"
        )

        # The new island should have the initial program
        spawned_island_idx = max(final_islands.keys())
        assert spawned_island_idx >= config.num_islands, (
            f"Spawned island index {spawned_island_idx} should be >= configured num_islands {config.num_islands}"
        )

        db.close()
        print("✓ Dynamic island spawning test passed!")


def test_no_spawning_when_disabled():
    """Test that no islands are spawned when feature is disabled."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_disabled.db"

        config = DatabaseConfig(
            db_path=str(db_path),
            num_islands=2,
            enable_dynamic_islands=False,  # Disabled
            stagnation_threshold=3,
        )

        db = ProgramDatabase(config=config, embedding_model="", read_only=False)

        # Add initial program
        initial_program = Program(
            id="initial_prog",
            code="def initial(): return 0",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
        )
        db.add(initial_program)

        initial_islands = db.island_manager.get_island_populations()

        # Add programs without improvement
        for gen in range(1, 10):
            program = Program(
                id=f"prog_gen_{gen}",
                code=f"def test(): return {gen}",
                correct=True,
                combined_score=0.5,
                generation=gen,
                island_idx=0,
            )
            db.add(program)

        final_islands = db.island_manager.get_island_populations()

        # Should NOT have spawned new islands
        assert len(final_islands) == len(initial_islands), (
            f"No new islands should be spawned when disabled. Initial: {initial_islands}, Final: {final_islands}"
        )

        db.close()
        print("✓ No-spawning-when-disabled test passed!")


def test_stagnation_reset_on_improvement():
    """Test that stagnation counter resets when best score improves."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reset.db"

        config = DatabaseConfig(
            db_path=str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=5,
        )

        db = ProgramDatabase(config=config, embedding_model="", read_only=False)

        # Add initial program
        initial_program = Program(
            id="initial_prog",
            code="def initial(): return 0",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
        )
        db.add(initial_program)

        # Add programs without improvement up to threshold - 1
        for gen in range(1, 4):
            program = Program(
                id=f"prog_gen_{gen}",
                code=f"def test(): return {gen}",
                correct=True,
                combined_score=0.5,  # No improvement
                generation=gen,
                island_idx=0,
            )
            db.add(program)

        # Now add a better program - should reset stagnation
        better_program = Program(
            id="better_prog",
            code="def better(): return 100",
            correct=True,
            combined_score=2.0,  # Improvement!
            generation=4,
            island_idx=0,
        )
        db.add(better_program)

        # Check that best_score_generation was updated
        assert db.best_score_generation == 4, (
            f"best_score_generation should be 4, got {db.best_score_generation}"
        )
        assert db.best_score_ever == 2.0, (
            f"best_score_ever should be 2.0, got {db.best_score_ever}"
        )

        # Should not be stagnant at generation 8 (only 4 gens since improvement)
        assert not db.is_stagnant(8), (
            "Should not be stagnant 4 generations after improvement"
        )

        # Should be stagnant at generation 9 (5 gens since improvement)
        assert db.is_stagnant(9), "Should be stagnant 5 generations after improvement"

        db.close()
        print("✓ Stagnation reset on improvement test passed!")


def test_spawn_strategies():
    """Test different island spawn strategies."""

    strategies = ["initial", "best", "archive_random"]

    for strategy in strategies:
        print(f"\n=== Testing spawn strategy: {strategy} ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / f"test_spawn_{strategy}.db"

            config = DatabaseConfig(
                db_path=str(db_path),
                num_islands=2,
                enable_dynamic_islands=True,
                stagnation_threshold=3,
                island_spawn_strategy=strategy,
            )

            db = ProgramDatabase(config=config, embedding_model="", read_only=False)

            # Add initial program (generation 0)
            initial_program = Program(
                id="initial_prog",
                code="def initial(): return 0",
                correct=True,
                combined_score=1.0,
                generation=0,
                island_idx=0,
            )
            db.add(initial_program)

            # Add a better program (will be the "best")
            best_program = Program(
                id="best_prog",
                code="def best(): return 100",
                correct=True,
                combined_score=5.0,  # Higher score
                generation=1,
                island_idx=0,
            )
            db.add(best_program)

            # Add programs without improvement until stagnation triggers
            for gen in range(2, 6):
                program = Program(
                    id=f"prog_gen_{gen}",
                    code=f"def test(): return {gen}",
                    correct=True,
                    combined_score=0.5,  # Lower than best, no improvement
                    generation=gen,
                    island_idx=0,
                )
                db.add(program)

            # Check that a new island was spawned
            final_islands = db.island_manager.get_island_populations()
            print(f"Final islands: {final_islands}")

            # Should have spawned at least one new island
            assert len(final_islands) > 2, (
                f"Expected new island for strategy '{strategy}'. Islands: {final_islands}"
            )

            # Check the spawned program's metadata
            spawned_island_idx = max(final_islands.keys())
            db.cursor.execute(
                "SELECT metadata FROM programs WHERE island_idx = ?",
                (spawned_island_idx,),
            )
            row = db.cursor.fetchone()
            if row:
                import json

                metadata = json.loads(row["metadata"] or "{}")
                assert metadata.get("_spawn_strategy") == strategy, (
                    f"Expected spawn strategy '{strategy}' in metadata"
                )
                print(
                    f"Spawned program metadata: _spawn_strategy={metadata.get('_spawn_strategy')}"
                )

            db.close()

        print(f"✓ Strategy '{strategy}' test passed!")


if __name__ == "__main__":
    test_stagnation_detection()
    test_dynamic_island_spawning()
    test_no_spawning_when_disabled()
    test_stagnation_reset_on_improvement()
    test_spawn_strategies()
    print("\n✓ All dynamic island tests passed!")
