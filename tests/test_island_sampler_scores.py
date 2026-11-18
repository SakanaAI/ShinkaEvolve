"""Island samplers across score ranges: negative, mixed-sign and large."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from typing import Optional

import numpy as np
import pytest

from shinka.database.island_sampler import IslandSampler, create_island_sampler


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE programs "
            "(island_idx INTEGER, combined_score REAL, correct BOOLEAN)"
        )
        yield connection


def _sampler(
    conn: sqlite3.Connection,
    strategy: str,
    best_scores: list[Optional[float]],
    counts: Optional[list[int]] = None,
) -> IslandSampler:
    """One island per score, holding `counts[i]` correct programs."""
    counts = counts or [1] * len(best_scores)
    rows = []
    for island_idx, (score, count) in enumerate(zip(best_scores, counts)):
        rows.append((island_idx, score))
        if score is not None:
            rows.extend((island_idx, score - 1.0) for _ in range(count - 1))
    conn.executemany("INSERT INTO programs VALUES (?, ?, 1)", rows)
    return create_island_sampler(conn.cursor(), conn, None, strategy)


def _probabilities(sampler: IslandSampler, num_islands: int) -> np.ndarray:
    probabilities = sampler.island_probabilities(list(range(num_islands)))  # type: ignore[attr-defined]
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert probabilities.sum() == pytest.approx(1.0)
    return probabilities


@pytest.mark.parametrize("strategy", ["proportional", "weighted"])
@pytest.mark.parametrize(
    "best_scores",
    [
        pytest.param([-1.0, 2.0], id="mixed-sign"),
        pytest.param([-5.0, -1.0], id="all-negative"),
        pytest.param([899.0, 900.0], id="large"),
        pytest.param([-1e6, 1e6], id="huge-range"),
        pytest.param([0.5, 2.0], id="positive"),
    ],
)
def test_better_island_is_preferred_on_any_score_range(conn, strategy, best_scores):
    sampler = _sampler(conn, strategy, best_scores)

    worse, better = _probabilities(sampler, 2)

    assert better > worse
    assert {sampler.sample_island([0, 1]) for _ in range(50)} <= {0, 1}


def test_proportional_keeps_boltzmann_probabilities(conn):
    sampler = _sampler(conn, "proportional", [1.0, 2.0, 4.0])

    expected = np.exp([1.0, 2.0, 4.0]) / np.exp([1.0, 2.0, 4.0]).sum()

    assert _probabilities(sampler, 3) == pytest.approx(expected)


def test_proportional_depends_only_on_score_differences(conn):
    small = _sampler(conn, "proportional", [1.0, 2.0])
    small_probabilities = _probabilities(small, 2)
    conn.execute("DELETE FROM programs")
    large = _sampler(conn, "proportional", [901.0, 902.0])

    assert _probabilities(large, 2) == pytest.approx(small_probabilities)


def test_weighted_keeps_fitness_over_count_for_non_negative_scores(conn):
    sampler = _sampler(conn, "weighted", [1.0, 2.0, 3.0], counts=[1, 2, 3])

    # fitness / count is 1 for every island.
    assert _probabilities(sampler, 3) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_weighted_never_drops_the_worst_negative_island(conn):
    sampler = _sampler(conn, "weighted", [-9.0, -3.0, -1.0])

    probabilities = _probabilities(sampler, 3)

    assert list(probabilities) == sorted(probabilities)
    assert probabilities[0] > 0
    assert probabilities[2] == pytest.approx(2 * probabilities[0])


def test_weighted_penalises_crowded_islands_with_negative_scores(conn):
    sampler = _sampler(conn, "weighted", [-2.0, -2.0], counts=[1, 4])

    sparse, crowded = _probabilities(sampler, 2)

    assert sparse == pytest.approx(4 * crowded)


@pytest.mark.parametrize("strategy", ["proportional", "weighted"])
@pytest.mark.parametrize(
    "best_scores",
    [
        pytest.param([0.0, 0.0], id="all-zero"),
        pytest.param([-3.0, -3.0], id="equal-negative"),
        pytest.param([None, None], id="unscored"),
    ],
)
def test_islands_without_a_preference_are_sampled_uniformly(
    conn, strategy, best_scores
):
    sampler = _sampler(conn, strategy, best_scores)

    assert _probabilities(sampler, 2) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("strategy", ["proportional", "weighted"])
@pytest.mark.parametrize("scored", [[-3.0, -1.0], [2.0, 5.0]])
def test_unscored_island_is_sampled_like_the_worst_scored_one(conn, strategy, scored):
    """A missing score is not a score of 0.0, whatever the scale of the others."""
    sampler = _sampler(conn, strategy, [None, *scored])

    unscored, worst, best = _probabilities(sampler, 3)

    assert unscored == pytest.approx(worst)
    assert best > worst
