"""Island sampling strategies for parent selection."""

import logging
import math
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

    def _best_fitness_values(self, island_indices: List[int]) -> np.ndarray:
        """Best fitness per island, in the order given, as finite floats.

        An island whose best program has no usable score counts like the worst
        scored island, or as 0.0 when no island has a usable score.
        """
        fitness = self._get_island_best_fitness(island_indices)
        scores = [fitness.get(island_idx) for island_idx in island_indices]
        usable = [
            isinstance(score, (int, float)) and math.isfinite(score)
            for score in scores
        ]
        worst = min((s for s, ok in zip(scores, usable) if ok), default=0.0)
        return np.array(
            [score if ok else worst for score, ok in zip(scores, usable)], dtype=float
        )


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

    def island_probabilities(self, initialized_islands: List[int]) -> np.ndarray:
        """Boltzmann probabilities over the islands, in the order given."""
        fitness_values = self._best_fitness_values(initialized_islands)

        # Boltzmann distribution: exp(fitness / temperature). It only depends
        # on score differences, so subtract the maximum first: every exponent
        # is then <= 0 and large scores cannot overflow.
        scaled = fitness_values / self.temperature
        exp_values = np.exp(scaled - np.max(scaled))
        return exp_values / np.sum(exp_values)

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample island proportional to best fitness."""
        probabilities = self.island_probabilities(initialized_islands)

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx]

        logger.debug(
            f"ProportionalIslandSampler: probabilities = {probabilities}, "
            f"sampled = {sampled_island}"
        )
        return sampled_island


class WeightedIslandSampler(IslandSampler):
    """Sample islands considering both program count and fitness.

    More programs -> lower probability
    Higher fitness -> higher probability

    The fitness term is the island's best score while every island's best
    score is non-negative. A power of a negative score is not a usable weight,
    so once any best score is negative the scores are shifted to
    ``score - worst + (best - worst)``: the best island then weighs twice the
    worst one, and the order of the islands is kept.
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

    def island_probabilities(self, initialized_islands: List[int]) -> np.ndarray:
        """Sampling probabilities over the islands, in the order given."""
        counts = self._get_island_program_counts(initialized_islands)
        count_values = np.array(
            [max(counts.get(island_idx, 1), 1) for island_idx in initialized_islands],
            dtype=float,
        )
        fitness_values = self._best_fitness_values(initialized_islands)
        worst = np.min(fitness_values)
        if worst < 0:
            span = np.max(fitness_values) - worst
            fitness_values = fitness_values - worst + (span if span > 0 else 1.0)

        # Weight = fitness^fitness_weight / count^count_weight
        # More fitness -> higher weight, more programs -> lower weight
        weights = fitness_values**self.fitness_weight / count_values**self.count_weight

        # Normalize to probabilities
        total = np.sum(weights)
        if total <= 0 or not np.isfinite(total):
            # Fallback to uniform if the weights carry no preference
            return np.ones(len(weights)) / len(weights)
        return weights / total

    def sample_island(self, initialized_islands: List[int]) -> int:
        """Sample island using weighted combination of fitness and inverse count."""
        probabilities = self.island_probabilities(initialized_islands)

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx]

        logger.debug(
            f"WeightedIslandSampler: probabilities = {probabilities}, "
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
