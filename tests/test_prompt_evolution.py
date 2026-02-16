"""
Tests for the meta-prompt evolution system.

Tests cover:
1. SystemPromptDatabase CRUD operations
2. SystemPrompt dataclass
3. Prompt fitness tracking and selection
4. Integration with Program tracking
"""

import pytest
import tempfile
from pathlib import Path

from shinka.database.prompt_dbase import (
    SystemPromptDatabase,
    SystemPromptConfig,
    SystemPrompt,
    create_system_prompt,
)
from shinka.database import Program
from shinka.core.prompt_evolver import SystemPromptSampler


class TestSystemPromptDataclass:
    """Test SystemPrompt dataclass functionality."""

    def test_create_system_prompt(self):
        """Test creating a new SystemPrompt."""
        prompt = create_system_prompt(
            prompt_text="You are an expert programmer.",
            generation=0,
            patch_type="init",
        )

        assert prompt.id is not None
        assert len(prompt.id) == 36  # UUID length
        assert prompt.prompt_text == "You are an expert programmer."
        assert prompt.generation == 0
        assert prompt.patch_type == "init"
        assert prompt.fitness == 0.0
        assert prompt.program_count == 0

    def test_prompt_to_dict(self):
        """Test serialization to dictionary."""
        prompt = create_system_prompt(
            prompt_text="Test prompt",
            generation=1,
            patch_type="diff",
            metadata={"test_key": "test_value"},
        )

        data = prompt.to_dict()
        assert data["prompt_text"] == "Test prompt"
        assert data["generation"] == 1
        assert data["patch_type"] == "diff"
        assert data["metadata"]["test_key"] == "test_value"

    def test_prompt_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-id-123",
            "prompt_text": "Test prompt",
            "generation": 2,
            "patch_type": "full",
            "fitness": 0.5,
            "program_count": 10,
            "metadata": {"key": "value"},
        }

        prompt = SystemPrompt.from_dict(data)
        assert prompt.id == "test-id-123"
        assert prompt.prompt_text == "Test prompt"
        assert prompt.generation == 2
        assert prompt.fitness == 0.5
        assert prompt.program_count == 10

    def test_update_fitness(self):
        """Test fitness update calculation using percentiles."""
        prompt = create_system_prompt(
            prompt_text="Test",
            generation=0,
            patch_type="init",
        )

        # Initial state
        assert prompt.fitness == 0.0
        assert prompt.program_count == 0

        # First percentile update (e.g., program beats 50% of others)
        prompt.update_fitness(percentile=0.5, program_score=0.75)
        assert prompt.program_count == 1
        assert prompt.correct_program_count == 1
        assert prompt.fitness == 0.5
        assert prompt.total_percentile == 0.5
        assert prompt.program_scores == [0.75]

        # Second percentile update (beats 80% of others)
        prompt.update_fitness(percentile=0.8, program_score=0.95)
        assert prompt.program_count == 2
        assert prompt.correct_program_count == 2
        assert prompt.fitness == 0.65  # (0.5 + 0.8) / 2
        assert prompt.total_percentile == 1.3
        assert prompt.program_scores == [0.75, 0.95]

        # Third percentile update (beats 30% of others)
        prompt.update_fitness(percentile=0.3, program_score=0.4)
        assert prompt.program_count == 3
        assert prompt.correct_program_count == 3
        assert abs(prompt.fitness - 0.5333) < 0.01  # (0.5 + 0.8 + 0.3) / 3
        assert prompt.program_scores == [0.75, 0.95, 0.4]

    def test_update_fitness_correct_only(self):
        """Test that only correct programs contribute to fitness."""
        prompt = create_system_prompt(
            prompt_text="Test",
            generation=0,
            patch_type="init",
        )

        # Add a correct program with 80% percentile
        prompt.update_fitness(percentile=0.8, correct=True, program_score=0.9)
        assert prompt.program_count == 1
        assert prompt.correct_program_count == 1
        assert prompt.fitness == 0.8
        assert prompt.program_scores == [0.9]

        # Add an incorrect program - should not affect fitness or scores
        prompt.update_fitness(percentile=0.0, correct=False, program_score=0.1)
        assert prompt.program_count == 2
        assert prompt.correct_program_count == 1  # Still 1
        assert prompt.fitness == 0.8  # Unchanged
        assert prompt.program_scores == [0.9]  # No new score added


class TestSystemPromptDatabase:
    """Test SystemPromptDatabase operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_prompts.db"
            config = SystemPromptConfig(
                db_path=str(db_path),
                archive_size=5,
            )
            db = SystemPromptDatabase(config)
            yield db
            db.close()

    @pytest.fixture
    def memory_db(self):
        """Create an in-memory database for testing."""
        config = SystemPromptConfig(
            db_path=None,  # In-memory
            archive_size=5,
        )
        db = SystemPromptDatabase(config)
        yield db
        db.close()

    def test_add_and_get_prompt(self, memory_db):
        """Test adding and retrieving a prompt."""
        prompt = create_system_prompt(
            prompt_text="Test prompt",
            generation=0,
            patch_type="init",
        )

        memory_db.add(prompt)
        retrieved = memory_db.get(prompt.id)

        assert retrieved is not None
        assert retrieved.id == prompt.id
        assert retrieved.prompt_text == "Test prompt"

    def test_get_nonexistent_prompt(self, memory_db):
        """Test getting a prompt that doesn't exist."""
        result = memory_db.get("nonexistent-id")
        assert result is None

    def test_get_all_prompts(self, memory_db):
        """Test getting all prompts."""
        prompts = []
        for i in range(3):
            prompt = create_system_prompt(
                prompt_text=f"Test prompt {i}",
                generation=i,
                patch_type="init",
            )
            memory_db.add(prompt)
            prompts.append(prompt)

        all_prompts = memory_db.get_all_prompts()
        assert len(all_prompts) == 3

    def test_archive_management(self, memory_db):
        """Test archive size management."""
        # Add prompts up to archive size
        for i in range(5):
            prompt = create_system_prompt(
                prompt_text=f"Prompt {i}",
                generation=i,
                patch_type="init",
            )
            memory_db.add(prompt)

        archive = memory_db.get_archive()
        assert len(archive) == 5

        # Add one more - should replace worst
        prompt = create_system_prompt(
            prompt_text="New prompt",
            generation=5,
            patch_type="init",
        )
        # Give it a higher fitness
        prompt.fitness = 10.0
        prompt.program_count = 5
        memory_db.add(prompt)

        archive = memory_db.get_archive()
        assert len(archive) <= 5

    def test_update_fitness(self, memory_db):
        """Test updating prompt fitness with percentiles."""
        prompt = create_system_prompt(
            prompt_text="Test",
            generation=0,
            patch_type="init",
        )
        memory_db.add(prompt)

        # Update fitness using percentiles with program scores
        memory_db.update_fitness(
            prompt.id, percentile=0.5, program_id="prog1", program_score=0.7
        )
        memory_db.update_fitness(
            prompt.id, percentile=0.9, program_id="prog2", program_score=0.95
        )

        updated = memory_db.get(prompt.id)
        assert updated.program_count == 2
        assert updated.correct_program_count == 2
        assert updated.fitness == 0.7  # (0.5 + 0.9) / 2
        assert len(updated.program_ids) == 2
        assert updated.program_scores == [0.7, 0.95]

    def test_sample_prompt(self, memory_db):
        """Test sampling prompts from archive."""
        # Add prompts with different fitness values (using percentiles)
        for i in range(3):
            prompt = create_system_prompt(
                prompt_text=f"Prompt {i}",
                generation=i,
                patch_type="init",
            )
            memory_db.add(prompt)
            # Update fitness with increasing percentiles and scores
            percentile = (i + 1) * 0.25  # 0.25, 0.5, 0.75
            score = (i + 1) * 0.3  # 0.3, 0.6, 0.9
            memory_db.update_fitness(
                prompt.id, percentile=percentile, program_score=score
            )
            memory_db.update_fitness(
                prompt.id, percentile=percentile, program_score=score
            )
            memory_db.update_fitness(
                prompt.id, percentile=percentile, program_score=score
            )

        # Sample multiple times
        samples = [memory_db.sample() for _ in range(10)]
        assert all(s is not None for s in samples)

    def test_get_best_prompt(self, memory_db):
        """Test getting the best prompt."""
        # Add prompts with different fitness
        prompt1 = create_system_prompt("Prompt 1", generation=0, patch_type="init")
        prompt2 = create_system_prompt("Prompt 2", generation=1, patch_type="init")
        prompt3 = create_system_prompt("Prompt 3", generation=2, patch_type="init")

        memory_db.add(prompt1)
        memory_db.add(prompt2)
        memory_db.add(prompt3)

        # Update fitness with different percentiles and scores
        for _ in range(3):
            memory_db.update_fitness(prompt1.id, percentile=0.3, program_score=0.4)
            memory_db.update_fitness(prompt2.id, percentile=0.9, program_score=0.95)
            memory_db.update_fitness(prompt3.id, percentile=0.5, program_score=0.6)

        best = memory_db.get_best_prompt()
        assert best is not None
        assert best.id == prompt2.id

    def test_prompt_lineage(self, memory_db):
        """Test getting prompt lineage."""
        # Create chain of prompts
        prompt0 = create_system_prompt("Prompt 0", generation=0, patch_type="init")
        memory_db.add(prompt0)

        prompt1 = create_system_prompt(
            "Prompt 1", parent_id=prompt0.id, generation=1, patch_type="diff"
        )
        memory_db.add(prompt1)

        prompt2 = create_system_prompt(
            "Prompt 2", parent_id=prompt1.id, generation=2, patch_type="full"
        )
        memory_db.add(prompt2)

        # Get lineage
        lineage = memory_db.get_lineage(prompt2.id)
        assert len(lineage) == 2
        assert lineage[0].id == prompt0.id
        assert lineage[1].id == prompt1.id

    def test_persistence(self, temp_db):
        """Test database persistence."""
        prompt = create_system_prompt(
            prompt_text="Persistent prompt",
            generation=0,
            patch_type="init",
        )
        temp_db.add(prompt)
        temp_db.save()

        # Get the db_path before closing
        db_path = temp_db.config.db_path

        temp_db.close()

        # Reopen and verify
        config = SystemPromptConfig(db_path=db_path, archive_size=5)
        db2 = SystemPromptDatabase(config)

        retrieved = db2.get(prompt.id)
        assert retrieved is not None
        assert retrieved.prompt_text == "Persistent prompt"
        db2.close()


class TestSystemPromptSampler:
    """Test SystemPromptSampler functionality."""

    @pytest.fixture
    def sampler_with_prompts(self):
        """Create a sampler with some prompts."""
        config = SystemPromptConfig(db_path=None, archive_size=10)
        db = SystemPromptDatabase(config)

        # Add prompts with varying fitness (using percentiles)
        for i in range(5):
            prompt = create_system_prompt(
                prompt_text=f"Prompt {i}",
                generation=i,
                patch_type="init",
            )
            db.add(prompt)
            # Give each prompt different percentile-based fitness with scores
            percentile = (i + 1) * 0.15  # 0.15, 0.30, 0.45, 0.60, 0.75
            score = (i + 1) * 0.18  # Corresponding scores
            for _ in range(3):
                db.update_fitness(prompt.id, percentile=percentile, program_score=score)

        sampler = SystemPromptSampler(prompt_db=db, exploration_constant=1.0)
        yield sampler, db
        db.close()

    def test_sample(self, sampler_with_prompts):
        """Test sampling prompts."""
        sampler, db = sampler_with_prompts

        # Sample should return a prompt
        prompt = sampler.sample()
        assert prompt is not None
        assert isinstance(prompt, SystemPrompt)

    def test_get_best_prompt(self, sampler_with_prompts):
        """Test getting best prompt through sampler."""
        sampler, db = sampler_with_prompts

        best = sampler.get_best_prompt()
        assert best is not None

    def test_get_archive(self, sampler_with_prompts):
        """Test getting archive through sampler."""
        sampler, db = sampler_with_prompts

        archive = sampler.get_archive()
        assert len(archive) == 5

    def test_ucb_exploration_effect(self):
        """Test that UCB exploration constant affects selection."""
        config = SystemPromptConfig(db_path=None, archive_size=10)
        db = SystemPromptDatabase(config)

        # Add prompts with very different percentile-based fitness
        low_fit = create_system_prompt("Low", generation=0, patch_type="init")
        high_fit = create_system_prompt("High", generation=1, patch_type="init")

        db.add(low_fit)
        db.add(high_fit)

        # Give very different percentile-based fitness with scores
        for _ in range(5):
            db.update_fitness(low_fit.id, percentile=0.2, program_score=0.3)
            db.update_fitness(high_fit.id, percentile=0.9, program_score=0.95)

        # Low exploration = more greedy (exploit high fitness)
        low_explore_sampler = SystemPromptSampler(
            db, exploration_constant=0.1, epsilon=0.0
        )
        high_explore_sampler = SystemPromptSampler(
            db, exploration_constant=10.0, epsilon=0.0
        )

        # Count how often high-fitness is selected
        # With low exploration, UCB should favor exploitation (high fitness prompt)
        low_explore_high_count = sum(
            1 for _ in range(100) if low_explore_sampler.sample().id == high_fit.id
        )

        # With high exploration, UCB should explore more evenly
        # (both prompts have same number of samples, so exploration term is equal,
        # but high exploration makes fitness difference matter less relatively)
        high_explore_high_count = sum(
            1 for _ in range(100) if high_explore_sampler.sample().id == high_fit.id
        )

        # Low exploration should select high fitness more often
        # (exploitation dominates over exploration)
        assert low_explore_high_count >= high_explore_high_count

        db.close()


class TestPromptEvolutionPrompts:
    """Test the prompt evolution prompt templates."""

    def test_diff_prompt_construction(self):
        """Test constructing diff evolution prompts."""
        from shinka.prompts.prompts_prompt_evo import (
            construct_diff_evolution_prompt,
        )

        prompt = create_system_prompt(
            prompt_text="Original prompt",
            generation=0,
            patch_type="init",
        )
        prompt.fitness = 0.5
        prompt.program_count = 10

        # New signature takes top_programs list instead of recent_improvements
        sys_msg, user_msg = construct_diff_evolution_prompt(prompt, top_programs=[])

        assert "expert prompt engineer" in sys_msg.lower()
        assert "Original prompt" in user_msg
        assert (
            "targeted" in sys_msg.lower()
        )  # Diff should mention targeted modifications

    def test_full_prompt_construction(self):
        """Test constructing full rewrite prompts."""
        from shinka.prompts.prompts_prompt_evo import (
            construct_full_evolution_prompt,
        )

        prompt = create_system_prompt(
            prompt_text="Original prompt",
            generation=0,
            patch_type="init",
        )

        # New signature takes top_programs list instead of best_program_info
        sys_msg, user_msg = construct_full_evolution_prompt(prompt, top_programs=[])

        assert "rewrite" in sys_msg.lower()
        assert "Original prompt" in user_msg

    def test_inspiration_prompt_formatting(self):
        """Test formatting inspiration prompts for prompt evolution context."""
        from shinka.prompts.prompts_prompt_evo import format_inspiration_prompts

        inspirations = [
            create_system_prompt(
                prompt_text=f"Inspiration {i}",
                generation=i,
                patch_type="diff",
            )
            for i in range(3)
        ]
        for i, insp in enumerate(inspirations):
            insp.fitness = 0.3 * (i + 1)
            insp.program_count = 5

        formatted = format_inspiration_prompts(inspirations, max_inspirations=2)

        assert "Inspiration 1" in formatted
        assert "Inspiration 2" in formatted
        assert "Inspiration 3" not in formatted
        assert "Performance: fitness=" in formatted


class TestEvolutionConfigPromptSettings:
    """Test EvolutionConfig prompt evolution settings."""

    def test_default_prompt_evolution_config(self):
        """Test default prompt evolution settings."""
        from shinka.core.runner import EvolutionConfig

        config = EvolutionConfig()

        assert config.evolve_prompts is False
        assert config.prompt_patch_types == ["diff", "full"]
        assert config.prompt_patch_type_probs == [0.7, 0.3]
        assert config.prompt_evolution_interval is None
        assert config.prompt_archive_size == 10

    def test_custom_prompt_evolution_config(self):
        """Test custom prompt evolution settings."""
        from shinka.core.runner import EvolutionConfig

        config = EvolutionConfig(
            evolve_prompts=True,
            prompt_patch_types=["diff", "full", "cross"],
            prompt_patch_type_probs=[0.5, 0.3, 0.2],
            prompt_evolution_interval=20,
            prompt_archive_size=15,
        )

        assert config.evolve_prompts is True
        assert "cross" in config.prompt_patch_types
        assert config.prompt_evolution_interval == 20
        assert config.prompt_archive_size == 15


class TestProgramSystemPromptId:
    """Test Program dataclass with system_prompt_id."""

    def test_program_with_prompt_id(self):
        """Test creating a program with system_prompt_id."""
        program = Program(
            id="test-program-id",
            code="print('hello')",
            generation=1,
            system_prompt_id="test-prompt-id",
        )

        assert program.system_prompt_id == "test-prompt-id"

    def test_program_without_prompt_id(self):
        """Test creating a program without system_prompt_id."""
        program = Program(
            id="test-program-id",
            code="print('hello')",
            generation=1,
        )

        assert program.system_prompt_id is None

    def test_program_to_dict_with_prompt_id(self):
        """Test program serialization with system_prompt_id."""
        program = Program(
            id="test-id",
            code="code",
            generation=1,
            system_prompt_id="prompt-id",
        )

        data = program.to_dict()
        assert data["system_prompt_id"] == "prompt-id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
