"""Island sampling strategies for parent selection."""

import logging
import random
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

logger = logging.getLogger(__name__)


class IslandSampler(ABC):
    """Abstract base class for island sampling strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config

    @abstractmethod
    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample an island index from the list of initialized islands.

        Args:
            initialized_islands: List of island indices that have correct programs

        Returns:
            Selected island index
        """
        pass

    def _get_island_program_counts(self, island_indices: List[int]) -> dict[int, int]:
        """Get the number of programs in each island.

        Args:
            island_indices: List of island indices to query

        Returns:
            Dictionary mapping island_idx to program count
        """
        if not island_indices:
            return {}

        placeholders = ",".join("?" * len(island_indices))
        query = f"""
            SELECT island_idx, COUNT(*) as count
            FROM programs
            WHERE island_idx IN ({placeholders}) AND correct = 1
            GROUP BY island_idx
        """
        self.cursor.execute(query, island_indices)

        counts = {island_idx: 0 for island_idx in island_indices}
        for row in self.cursor.fetchall():
            counts[row["island_idx"]] = row["count"]

        return counts

    def _get_island_best_fitness(self, island_indices: List[int]) -> dict[int, float]:
        """Get the best fitness (combined_score) from each island.

        Args:
            island_indices: List of island indices to query

        Returns:
            Dictionary mapping island_idx to best combined_score
        """
        if not island_indices:
            return {}

        placeholders = ",".join("?" * len(island_indices))
        query = f"""
            SELECT island_idx, MAX(combined_score) as best_fitness
            FROM programs
            WHERE island_idx IN ({placeholders}) AND correct = 1
            GROUP BY island_idx
        """
        self.cursor.execute(query, island_indices)

        fitness = {}
        for row in self.cursor.fetchall():
            fitness[row["island_idx"]] = row["best_fitness"]

        return fitness


class UniformIslandSampler(IslandSampler):
    """Uniformly sample from initialized islands (default behavior)."""

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Uniformly sample an island."""
        return random.choice(initialized_islands)


class EqualIslandSampler(IslandSampler):
    """Sample the island with the fewest programs.

    If multiple islands have the same minimum count, sample uniformly among them.
    """

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample island with fewest programs."""
        counts = self._get_island_program_counts(initialized_islands)

        min_count = min(counts.values())
        islands_with_min = [
            island_idx for island_idx, count in counts.items() if count == min_count
        ]

        sampled = random.choice(islands_with_min)
        logger.debug(
            f"EqualIslandSampler: Island counts = {counts}, "
            f"min_count = {min_count}, sampled = {sampled}"
        )
        return sampled


class ProportionalIslandSampler(IslandSampler):
    """Sample islands proportional to their best fitness using Boltzmann distribution.

    Uses a medium temperature for the Boltzmann distribution.
    """

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        temperature: float = 1.0,
    ):
        super().__init__(cursor, conn, config)
        self.temperature = temperature

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample island proportional to best fitness."""
        fitness_dict = self._get_island_best_fitness(initialized_islands)

        # Extract fitness values in the same order as initialized_islands.
        # dtype=float coerces missing/NULL scores (None) to NaN, handled below.
        fitness_values = np.array(
            [fitness_dict.get(island_idx, 0.0) for island_idx in initialized_islands],
            dtype=np.float64,
        )

        n_islands = len(initialized_islands)
        # Replace non-finite fitness (NaN/inf from NULL or overflow) with the
        # smallest finite fitness so it neither dominates nor breaks the softmax.
        finite_mask = np.isfinite(fitness_values)
        if not finite_mask.any():
            # No usable fitness information: fall back to uniform sampling.
            probabilities = np.ones(n_islands) / n_islands
        else:
            fitness_values = np.where(
                finite_mask, fitness_values, fitness_values[finite_mask].min()
            )
            # Numerically stable Boltzmann softmax: subtract the max before
            # exponentiating so large-magnitude scores cannot overflow to inf
            # (inf / inf -> NaN would make np.random.choice raise). Subtracting
            # a constant is invariant, so the distribution is unchanged.
            logits = fitness_values / self.temperature
            exp_values = np.exp(logits - logits.max())
            total = exp_values.sum()
            if not np.isfinite(total) or total <= 0.0:
                probabilities = np.ones(n_islands) / n_islands
            else:
                probabilities = exp_values / total

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx]

        logger.debug(
            f"ProportionalIslandSampler: fitness = {fitness_dict}, "
            f"probabilities = {probabilities}, sampled = {sampled_island}"
        )
        return sampled_island


class WeightedIslandSampler(IslandSampler):
    """Sample islands considering both program count and fitness.

    More programs -> lower probability
    Higher fitness -> higher probability
    """

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        fitness_weight: float = 1.0,
        count_weight: float = 1.0,
    ):
        super().__init__(cursor, conn, config)
        self.fitness_weight = fitness_weight
        self.count_weight = count_weight

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample island using weighted combination of fitness and inverse count."""
        counts = self._get_island_program_counts(initialized_islands)
        fitness_dict = self._get_island_best_fitness(initialized_islands)

        n_islands = len(initialized_islands)
        # dtype=float coerces missing/NULL scores (None) to NaN, handled below.
        fitness_values = np.array(
            [fitness_dict.get(island_idx, 0.0) for island_idx in initialized_islands],
            dtype=np.float64,
        )
        count_values = np.array(
            [counts.get(island_idx, 0) for island_idx in initialized_islands],
            dtype=np.float64,
        )

        # Replace non-finite fitness with the smallest finite fitness (or 0.0).
        finite_mask = np.isfinite(fitness_values)
        if finite_mask.any():
            fitness_values = np.where(
                finite_mask, fitness_values, fitness_values[finite_mask].min()
            )
        else:
            fitness_values = np.zeros_like(fitness_values)

        # Shift fitness so weights stay non-negative and monotonic (higher
        # fitness -> higher weight) even when fitness is negative. For the
        # normal all-positive case we leave fitness untouched to preserve the
        # original weighting; otherwise we subtract the minimum and add a small
        # epsilon so the lowest-fitness island keeps a nonzero (but smallest)
        # probability instead of being dropped or producing negative weights.
        min_fitness = fitness_values.min()
        if min_fitness <= 0.0:
            fitness_values = fitness_values - min_fitness + 1e-9

        # Guard against zero/negative counts before exponentiating.
        safe_counts = np.maximum(count_values, 1.0)

        # Weight = fitness^fitness_weight / count^count_weight
        # More fitness -> higher weight, more programs -> lower weight
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weights = np.power(fitness_values, self.fitness_weight) / np.power(
                safe_counts, self.count_weight
            )

        # Guard non-finite/negative weights that would break np.random.choice.
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.clip(weights, 0.0, None)

        # Normalize to probabilities
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            # Fallback to uniform if all weights are zero/invalid
            probabilities = np.ones(n_islands) / n_islands
        else:
            probabilities = weights / total

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx]

        logger.debug(
            f"WeightedIslandSampler: counts = {counts}, fitness = {fitness_dict}, "
            f"weights = {weights}, probabilities = {probabilities}, "
            f"sampled = {sampled_island}"
        )
        return sampled_island


def create_island_sampler(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    config: Any,
    strategy: str = "uniform",
) -> IslandSampler:
    """Factory function to create island samplers.

    Args:
        cursor: Database cursor
        conn: Database connection
        config: Database configuration
        strategy: Sampling strategy name

    Returns:
        IslandSampler instance

    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == "uniform":
        return UniformIslandSampler(cursor, conn, config)
    elif strategy == "equal":
        return EqualIslandSampler(cursor, conn, config)
    elif strategy == "proportional":
        return ProportionalIslandSampler(cursor, conn, config, temperature=1.0)
    elif strategy == "weighted":
        return WeightedIslandSampler(
            cursor, conn, config, fitness_weight=1.0, count_weight=1.0
        )
    else:
        raise ValueError(
            f"Unknown island sampling strategy: {strategy}. "
            f"Valid options: 'uniform', 'equal', 'proportional', 'weighted'"
        )
