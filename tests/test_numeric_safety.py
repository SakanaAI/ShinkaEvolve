"""Regression tests for numeric-safety correctness fixes.

Covers three verified bugs:

* ``get_best_program`` treated an exactly-0.0 combined score as ``-inf`` via the
  ``score or -inf`` falsy-zero antipattern, so a genuinely-0.0 program sorted as
  the worst and a negative-scoring peer was returned as "best" (Q13 / M3).
* The proportional / weighted island samplers crashed or inverted their
  preference on large-magnitude and negative fitness because of an unstable
  softmax and unguarded power weights (M4).
* ``_posterior_batch`` clobbered its virtual-pull loop counter ``k`` with the
  cost-aware blend coefficient, exiting the epsilon-greedy rollout after a
  single iteration instead of running ``samples`` times (M2).
"""

import tempfile
from pathlib import Path

import numpy as np

from shinka.database import DatabaseConfig, ProgramDatabase, Program
from shinka.database.island_sampler import (
    ProportionalIslandSampler,
    WeightedIslandSampler,
)
from shinka.llm import AsymmetricUCB


# --------------------------------------------------------------------------- #
# Q13 / M3: get_best_program must not treat an exactly-0.0 score as -inf.
# --------------------------------------------------------------------------- #
def test_get_best_program_prefers_zero_score_over_negatives():
    """A correct program scoring exactly 0.0 must beat negative-scoring peers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "best.db"
        config = DatabaseConfig(db_path=str(db_path), num_islands=2)
        db = ProgramDatabase(config=config, embedding_model="", read_only=False)

        scores = {"zero": 0.0, "neg_one": -1.0, "neg_two": -2.0}
        for name, score in scores.items():
            db.add(
                Program(
                    id=name,
                    code=f"def f(): return {score}",
                    correct=True,
                    combined_score=score,
                    generation=0,
                    island_idx=0,
                )
            )

        # Force the fallback sort path by clearing the tracked best id so the
        # buggy ``score or -inf`` sort key is actually exercised.
        db.best_program_id = None
        best = db.get_best_program()

        assert best is not None
        assert best.id == "zero", (
            f"Expected 0.0-scoring program to win, got {best.id} "
            f"(score={best.combined_score})"
        )
        assert best.combined_score == 0.0
        db.close()


# --------------------------------------------------------------------------- #
# M4: island samplers must be numerically safe and keep higher fitness ->
# higher probability.
# --------------------------------------------------------------------------- #
def _make_proportional(fitness_by_island):
    sampler = ProportionalIslandSampler(
        cursor=None, conn=None, config=None, temperature=1.0
    )
    # Bypass the DB helper with controlled fitness values.
    sampler._get_island_best_fitness = lambda islands: dict(fitness_by_island)
    return sampler


def _make_weighted(fitness_by_island, count_by_island):
    sampler = WeightedIslandSampler(
        cursor=None,
        conn=None,
        config=None,
        fitness_weight=1.0,
        count_weight=1.0,
    )
    sampler._get_island_best_fitness = lambda islands: dict(fitness_by_island)
    sampler._get_island_program_counts = lambda islands: dict(count_by_island)
    return sampler


def test_proportional_sampler_large_magnitude_fitness_no_overflow():
    """Large fitness must not overflow to inf/nan; higher fitness preferred."""
    np.random.seed(0)
    islands = [0, 1]
    # exp(800) overflows float64 -> old code produced inf/inf = nan and raised.
    sampler = _make_proportional({0: 800.0, 1: 700.0})

    counts = {0: 0, 1: 0}
    for _ in range(200):
        sampled = sampler.sample_island(islands)  # must not raise
        counts[sampled] += 1

    assert counts[0] > counts[1], f"Higher-fitness island not preferred: {counts}"


def test_weighted_sampler_all_negative_fitness_not_inverted():
    """All-negative fitness must not raise nor invert the preference."""
    np.random.seed(0)
    islands = [0, 1]
    # Old code: weights [-1, -10] -> normalized to [0.09, 0.91] => the WORSE
    # island (-10) is favored. The fix must favor the higher (-1) island.
    sampler = _make_weighted({0: -1.0, 1: -10.0}, {0: 1, 1: 1})

    counts = {0: 0, 1: 0}
    for _ in range(200):
        sampled = sampler.sample_island(islands)  # must not raise
        counts[sampled] += 1

    assert counts[0] > counts[1], f"Preference inverted for negatives: {counts}"


def test_weighted_sampler_zero_count_and_mixed_sign_no_crash():
    """Zero counts (div-by-zero) and mixed-sign fitness must stay safe."""
    np.random.seed(0)
    islands = [0, 1, 2]
    # island 1 has a negative fitness AND zero count (0**weight -> div by zero).
    sampler = _make_weighted(
        {0: 5.0, 1: -3.0, 2: 0.0}, {0: 2, 1: 0, 2: 1}
    )

    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(300):
        sampled = sampler.sample_island(islands)  # must not raise
        counts[sampled] += 1

    # Highest-fitness island is sampled most; the min-fitness island is starved.
    assert counts[0] == max(counts.values()), f"Highest fitness not top: {counts}"
    assert counts[1] == 0, f"Lowest-fitness island should be starved: {counts}"


def test_proportional_sampler_all_equal_fitness_is_uniform():
    """Equal fitness should give a well-formed (roughly uniform) distribution."""
    np.random.seed(0)
    islands = [0, 1, 2]
    sampler = _make_proportional({0: 3.0, 1: 3.0, 2: 3.0})

    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(300):
        counts[sampler.sample_island(islands)] += 1

    assert all(c > 0 for c in counts.values()), f"Not all islands sampled: {counts}"


# --------------------------------------------------------------------------- #
# M2: _posterior_batch must run the full epsilon-greedy rollout, not exit after
# one iteration because the cost coefficient clobbered the loop counter.
# --------------------------------------------------------------------------- #
class _CountingRng:
    """Wrap a numpy Generator and count how many times ``choice`` is called."""

    def __init__(self, rng):
        self._rng = rng
        self.choice_calls = 0

    def choice(self, *args, **kwargs):
        self.choice_calls += 1
        return self._rng.choice(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_posterior_batch_runs_full_rollout_with_cost_aware():
    """The virtual-pull loop must iterate ``samples`` times, not exit early."""
    bandit = AsymmetricUCB(
        arm_names=["a", "b", "c"],
        cost_aware_coef=0.5,
        auto_decay=None,
        exponential_base=None,
        seed=0,
    )

    # Make every arm "seen" (n > 0) so there are no unseen pre-allocations, and
    # provide distinct costs so the cost-aware blend branch (which held the bug)
    # actually executes.
    for arm, reward, cost in [("a", 1.0, 1.0), ("b", 0.5, 2.0), ("c", 0.2, 3.0)]:
        bandit.update(arm, reward=reward, baseline=0.0)
        bandit.update_cost(arm, cost=cost)

    counting = _CountingRng(bandit.rng)
    bandit.rng = counting

    idx = bandit._resolve_subset(None)
    samples = 5
    probs = bandit._posterior_batch(idx, samples)

    # One rng.choice call per rollout iteration; with the bug it was exactly 1.
    assert counting.choice_calls == samples, (
        f"Rollout exited early: {counting.choice_calls} of {samples} iterations"
    )
    assert np.isclose(probs.sum(), 1.0)


if __name__ == "__main__":
    test_get_best_program_prefers_zero_score_over_negatives()
    test_proportional_sampler_large_magnitude_fitness_no_overflow()
    test_weighted_sampler_all_negative_fitness_not_inverted()
    test_weighted_sampler_zero_count_and_mixed_sign_no_crash()
    test_proportional_sampler_all_equal_fitness_is_uniform()
    test_posterior_batch_runs_full_rollout_with_cost_aware()
    print("✅ All numeric-safety regression tests passed!")
