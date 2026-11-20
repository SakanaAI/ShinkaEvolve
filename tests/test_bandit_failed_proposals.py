"""A model whose proposals fail loses selection weight, and gets it back."""

import logging
from io import StringIO

import numpy as np
import pytest
from rich.console import Console

from shinka.llm.prioritization import AsymmetricUCB, FixedSampler, ThompsonSampler

ARMS = ["ok", "dead"]
OK, DEAD = 0, 1
BASELINE = 1.0
PATCH_COST = 0.02

BANDITS = {
    "ucb-runner-default": lambda seed: AsymmetricUCB(
        arm_names=ARMS, seed=seed, cost_aware_coef=0.5
    ),
    "ucb-reward-only": lambda seed: AsymmetricUCB(arm_names=ARMS, seed=seed),
    "ucb-cost-only": lambda seed: AsymmetricUCB(
        arm_names=ARMS, seed=seed, cost_aware_coef=1.0
    ),
    "thompson": lambda seed: ThompsonSampler(arm_names=ARMS, seed=seed),
    "thompson-greedy": lambda seed: ThompsonSampler(
        arm_names=ARMS, seed=seed, epsilon=0.0
    ),
    "fixed": lambda seed: FixedSampler(arm_names=ARMS, seed=seed),
}
ALL_BANDITS = pytest.mark.parametrize("make_bandit", BANDITS.values(), ids=BANDITS)
# What a failed proposal was billed: nothing, as for an endpoint failure or a
# rejected request, or as much as a proposal that delivers.
FAILED_COSTS = pytest.mark.parametrize("failed_cost", [0.0, PATCH_COST])


def _propose(bandit, world, fails, failed_cost=0.0):
    """One proposal, reported to the bandit the way async_runner reports it."""
    one_hot, _ = bandit.select_llm()
    arm = int(np.argmax(one_hot))
    bandit.update_submitted(ARMS[arm])
    if arm == DEAD and fails:
        bandit.update_failed(ARMS[arm], cost=failed_cost)
        return arm
    bandit.update_cost(ARMS[arm], PATCH_COST * world.uniform(0.5, 1.5))
    bandit.update_delivered(ARMS[arm])
    roll = world.random()
    if roll < 0.3:
        reward = BASELINE + world.uniform(0.05, 0.5)
    elif roll < 0.5:
        reward = None  # evaluated and incorrect
    else:
        reward = BASELINE - world.uniform(0.0, 0.3)
    bandit.update(ARMS[arm], reward, baseline=BASELINE)
    return arm


def _draws(make_bandit, seed, proposals, fails, failed_cost=0.0):
    bandit = make_bandit(seed)
    bandit.set_baseline_score(BASELINE)
    world = np.random.default_rng(1000 + seed)
    return np.array(
        [_propose(bandit, world, fails(step), failed_cost) for step in range(proposals)]
    )


@ALL_BANDITS
@FAILED_COSTS
def test_model_that_always_fails_gets_few_of_fifty_proposals(make_bandit, failed_cost):
    dead_submissions = [
        _draws(make_bandit, seed, 50, lambda step: True, failed_cost).sum()
        for seed in range(20)
    ]

    # Without the outcome reports the same pool sends 19 to 49 of 50 there.
    assert np.mean(dead_submissions) < 11
    assert max(dead_submissions) < 17


@ALL_BANDITS
def test_model_that_works_again_wins_its_share_back(make_bandit):
    runs = np.array(
        [_draws(make_bandit, seed, 150, lambda step: step < 30) for seed in range(20)]
    )

    while_failing = runs[:, :30].mean()
    long_after = runs[:, 100:].mean()

    assert while_failing < 0.3
    assert long_after > 1.4 * while_failing


@pytest.mark.parametrize("make_bandit", [BANDITS["thompson"], BANDITS["fixed"]])
def test_recovered_model_is_drawn_like_the_other_again(make_bandit):
    runs = np.array(
        [_draws(make_bandit, seed, 150, lambda step: step < 30) for seed in range(20)]
    )

    assert 0.35 < runs[:, 100:].mean() < 0.65


@ALL_BANDITS
def test_reliability_is_the_share_of_recent_proposals_that_delivered(make_bandit):
    bandit = make_bandit(0)
    assert bandit.reliability().tolist() == [1.0, 1.0]

    for step in range(80):
        if step % 4 == 0:
            bandit.update_failed("dead")
        else:
            bandit.update_delivered("dead")

    ok, dead = bandit.reliability()
    assert ok == 1.0
    assert dead == pytest.approx(0.75, abs=0.08)


@ALL_BANDITS
def test_dead_model_loses_its_reliability_but_is_still_tried(make_bandit):
    bandit = make_bandit(0)
    bandit.set_baseline_score(BASELINE)
    for _ in range(10):
        bandit.update_submitted("ok")
        bandit.update_cost("ok", PATCH_COST)
        bandit.update_delivered("ok")
        bandit.update("ok", BASELINE + 0.2, baseline=BASELINE)
    seen = []
    for _ in range(200):
        bandit.update_submitted("dead")
        bandit.update_failed("dead")
        seen.append(bandit.reliability()[DEAD])

    # The delivery assumed for a new arm fades: 0.9 / 1.9, 0.81 / 2.71, ...
    assert seen[0] == pytest.approx(0.9 / 1.9)
    assert seen[1] == pytest.approx(0.81 / 2.71)
    assert all(later < earlier for earlier, later in zip(seen, seen[1:]))
    assert seen[-1] < 1e-9
    # The probe share: 1 / (1 + failure_memory) of an even share, renormalised.
    probe = (1 / 22) / (1 + 1 / 22)
    for _ in range(20):
        assert bandit.posterior()[DEAD] == pytest.approx(probe)


@ALL_BANDITS
def test_reliability_recovers_with_delivered_proposals(make_bandit):
    bandit = make_bandit(0)
    for _ in range(100):
        bandit.update_failed("dead")

    recovery = []
    for _ in range(30):
        bandit.update_delivered("dead")
        recovery.append(bandit.reliability()[DEAD])

    assert 0.05 < recovery[0] < 0.15
    assert recovery[9] > 0.6
    assert recovery[-1] > 0.95


def test_fixed_sampler_weights_its_probabilities_by_reliability():
    bandit = FixedSampler(arm_names=["a", "b", "c"], prior_probs=[0.5, 0.3, 0.2])
    np.testing.assert_array_equal(bandit.posterior(), [0.5, 0.3, 0.2])

    bandit.update_failed("a")
    bandit.update_failed("a")
    reliability = bandit.reliability()

    assert reliability[0] == pytest.approx(0.81 / 2.71)
    expected = np.array([0.5 * reliability[0], 0.3, 0.2])
    np.testing.assert_allclose(bandit.posterior(), expected / expected.sum())
    restricted = bandit.posterior(subset=["a", "b"])
    np.testing.assert_allclose(
        restricted, [expected[0] / expected[:2].sum(), 0.3 / expected[:2].sum(), 0.0]
    )


def _paid_and_failing(cost_aware_coef, failed_cost):
    """`ok` has delivered billed, non-improving patches; `dead` only failed."""
    bandit = AsymmetricUCB(arm_names=ARMS, seed=0, cost_aware_coef=cost_aware_coef)
    bandit.set_baseline_score(BASELINE)
    for _ in range(5):
        bandit.update_submitted("ok")
        bandit.update_cost("ok", PATCH_COST)
        bandit.update_delivered("ok")
        bandit.update("ok", BASELINE, baseline=BASELINE)
        bandit.update_submitted("dead")
        bandit.update_failed("dead", cost=failed_cost)
    return bandit


@pytest.mark.parametrize("cost_aware_coef", [0.0, 0.25, 0.5, 1.0])
@FAILED_COSTS
def test_failing_for_free_does_not_make_a_model_attractive(
    cost_aware_coef, failed_cost
):
    bandit = _paid_and_failing(cost_aware_coef, failed_cost)

    ok, dead = bandit.posterior()

    # `dead` is not the arm that takes 1 - epsilon, and its share of epsilon
    # is cut by its reliability.
    assert ok > 0.9
    assert dead < 0.1


def test_arm_without_a_cost_report_is_not_treated_as_free():
    """Its proposal may be in flight, or may have failed before any bill."""
    bandit = AsymmetricUCB(arm_names=["billed", "unknown"], cost_aware_coef=1.0)
    for _ in range(3):
        bandit.update_submitted("billed")
        bandit.update_cost("billed", PATCH_COST)
        bandit.update_submitted("unknown")

    billed, unknown = bandit.posterior()

    assert unknown <= billed


def test_bill_of_a_failed_proposal_is_not_a_cost_observation():
    """An unbilled failure must not lower the arm's cost per patch."""
    steady = AsymmetricUCB(arm_names=["a", "b"], cost_aware_coef=1.0)
    flaky = AsymmetricUCB(arm_names=["a", "b"], cost_aware_coef=1.0)
    for bandit in (steady, flaky):
        for arm, cost in (("a", 0.02), ("b", 0.04), ("a", 0.02), ("b", 0.04)):
            bandit.update_submitted(arm)
            bandit.update_cost(arm, cost)
            bandit.update_delivered(arm)
    for _ in range(4):
        flaky.update_submitted("b")
        flaky.update_failed("b")

    np.testing.assert_array_equal(flaky.n_costs, steady.n_costs)
    np.testing.assert_array_equal(flaky.total_costs, steady.total_costs)

    flaky.update_failed("b", cost=0.5)

    assert flaky.total_costs[1] == pytest.approx(steady.total_costs[1] + 0.5)
    np.testing.assert_array_equal(flaky.n_costs, steady.n_costs)
    idx = np.arange(2)
    assert (
        flaky._normalized_cost_ratio(idx, 1.0)[1]
        < steady._normalized_cost_ratio(idx, 1.0)[1]
    )


def test_thompson_does_not_keep_drawing_a_model_that_never_completes():
    bandit = ThompsonSampler(arm_names=ARMS, seed=0)
    bandit.set_baseline_score(BASELINE)
    bandit.update_submitted("ok")
    bandit.update_delivered("ok")
    bandit.update("ok", BASELINE + 0.2, baseline=BASELINE)
    bandit.update_submitted("dead")

    # No outcome yet: still the arm to try first.
    assert bandit.posterior()[DEAD] == 1.0

    bandit.update_failed("dead")

    assert all(bandit.posterior()[DEAD] < 1.0 for _ in range(50))


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_failing_never_raises_a_negative_score(cls):
    """Linear mode without clamping has negative mean rewards."""
    bandit = cls(
        arm_names=ARMS,
        seed=0,
        shift_by_baseline=False,
        shift_by_parent=False,
        asymmetric_scaling=False,
    )
    for _ in range(3):
        bandit.update_failed("dead")
    scores = np.array([-2.0, -2.0])

    ok, dead = bandit._discount_by_reliability(scores, np.arange(2))

    assert ok == -2.0
    assert dead < ok


@ALL_BANDITS
def test_no_failure_leaves_the_selection_untouched(make_bandit):
    reported, silent = make_bandit(3), make_bandit(3)
    for bandit in (reported, silent):
        bandit.set_baseline_score(BASELINE)
    for step in range(30):
        arm = ARMS[step % 2]
        for bandit in (reported, silent):
            bandit.update_submitted(arm)
            bandit.update_cost(arm, PATCH_COST * (1 + step % 3))
            bandit.update(arm, BASELINE + 0.1 * (step % 4), baseline=BASELINE)
        reported.update_delivered(arm)

        np.testing.assert_array_equal(reported.posterior(), silent.posterior())


@ALL_BANDITS
def test_failure_memory_zero_switches_the_weighting_off(make_bandit):
    template = make_bandit(0)
    bandit = type(template)(arm_names=ARMS, seed=0, failure_memory=0)
    untouched = type(template)(arm_names=ARMS, seed=0, failure_memory=0)
    for _ in range(5):
        bandit.update_failed("dead")
        bandit.update_delivered("ok")

    assert bandit.reliability().tolist() == [1.0, 1.0]
    np.testing.assert_array_equal(bandit.posterior(), untouched.posterior())


@pytest.mark.parametrize("failure_memory", [-1.0, 0.5, float("nan")])
@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler, FixedSampler])
def test_failure_memory_must_be_zero_or_at_least_one(cls, failure_memory):
    with pytest.raises(ValueError, match="failure_memory"):
        cls(arm_names=ARMS, failure_memory=failure_memory)


@ALL_BANDITS
def test_outcome_report_for_an_unknown_arm_is_ignored_with_one_warning(
    make_bandit, caplog
):
    """The runner reports from its failure handlers, which must not raise."""
    bandit = make_bandit(0)

    with caplog.at_level(logging.WARNING, logger="shinka.llm.prioritization"):
        bandit.update_failed("not-in-the-pool", cost=0.5)
        bandit.update_delivered("not-in-the-pool")
        bandit.update_failed(7)

    assert caplog.text.count("cannot track the outcomes") == 1
    assert bandit.reliability().tolist() == [1.0, 1.0]
    assert bandit.get_state().get("total_costs", np.zeros(2)).tolist() == [0.0, 0.0]


@ALL_BANDITS
def test_outcome_counts_survive_save_and_load(make_bandit, tmp_path):
    bandit = make_bandit(0)
    for _ in range(4):
        bandit.update_failed("dead", cost=0.01)
    bandit.update_delivered("dead")
    bandit.update_delivered("ok")
    path = tmp_path / "bandit_state.pkl"

    bandit.save_state(path)
    restored = make_bandit(1)
    restored.load_state(path)

    np.testing.assert_array_equal(restored.recent_failed, bandit.recent_failed)
    np.testing.assert_array_equal(restored.recent_delivered, bandit.recent_delivered)
    np.testing.assert_array_equal(restored.reliability(), bandit.reliability())
    assert restored.reliability()[DEAD] < 0.5


@ALL_BANDITS
def test_outcome_counts_follow_the_model_when_the_pool_changes(make_bandit, tmp_path):
    bandit = make_bandit(0)
    for _ in range(4):
        bandit.update_failed("dead")
    path = tmp_path / "bandit_state.pkl"
    bandit.save_state(path)

    reordered = type(bandit)(arm_names=["new", "dead", "ok"], seed=0)
    reordered.load_state(path)

    assert reordered.reliability()[0] == 1.0
    assert reordered.reliability()[1] == bandit.reliability()[DEAD]
    assert reordered.reliability()[2] == 1.0


@ALL_BANDITS
def test_state_saved_without_outcome_counts_loads_as_fully_reliable(make_bandit):
    saved = make_bandit(0)
    saved.update_submitted("ok")
    saved.update("ok", BASELINE + 0.1, baseline=BASELINE)
    state = saved.get_state()
    del state["recent_delivered"], state["recent_failed"]

    restored = make_bandit(1)
    restored.update_failed("dead")
    restored.set_state(state)

    assert restored.reliability().tolist() == [1.0, 1.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
@ALL_BANDITS
def test_invalid_saved_outcome_counts_are_reset(make_bandit, bad, caplog):
    state = make_bandit(0).get_state()
    state["recent_failed"] = [0.0, bad]
    restored = make_bandit(1)

    with caplog.at_level(logging.WARNING, logger="shinka.llm.prioritization"):
        restored.set_state(state)

    assert "invalid proposal outcome counts" in caplog.text
    assert restored.reliability().tolist() == [1.0, 1.0]
    one_hot, probabilities = restored.select_llm()
    assert one_hot.sum() == 1.0
    assert np.isclose(probabilities.sum(), 1.0)


@ALL_BANDITS
def test_summary_names_the_models_whose_proposals_failed(make_bandit):
    bandit = make_bandit(0)

    def summary() -> str:
        buffer = StringIO()
        bandit.print_summary(console=Console(file=buffer, width=200))
        return buffer.getvalue()

    assert "returned a patch" not in summary()

    bandit.update_submitted("dead")
    bandit.update_failed("dead")

    assert "share of recent proposals that returned a patch" in summary()
    assert "dead 0.47" in summary()


@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_greedy_thompson_still_finds_out_that_a_model_works_again(epsilon):
    """Without the probe share a failed arm that stops winning is never drawn."""
    arms = ["best", "second", "third"]
    quality = {"best": 1.0, "second": 0.3, "third": 0.2}
    shares = []
    for seed in range(8):
        bandit = ThompsonSampler(arm_names=arms, seed=seed, epsilon=epsilon)
        bandit.set_baseline_score(0.0)
        world = np.random.default_rng(seed)
        picks = []
        for step in range(900):
            one_hot, _ = bandit.select_llm()
            arm = arms[int(np.argmax(one_hot))]
            picks.append(arm)
            bandit.update_submitted(arm)
            if arm == "best" and 150 <= step < 200:
                bandit.update_failed(arm)
                continue
            bandit.update_delivered(arm)
            bandit.update(arm, quality[arm] + world.normal(0, 0.05), baseline=0.0)
        shares.append(np.mean([arm == "best" for arm in picks[600:]]))

    assert min(shares) > 0.5


def test_model_that_was_cheap_before_it_died_does_not_keep_the_top_score():
    """Its mean cost stays low; only its reliability says that it is gone."""
    bandit = AsymmetricUCB(arm_names=["cheap", "dear"], seed=0, cost_aware_coef=1.0)
    bandit.set_baseline_score(BASELINE)
    for _ in range(20):
        for arm, cost in (("cheap", 0.001), ("dear", 0.02)):
            bandit.update_submitted(arm)
            bandit.update_cost(arm, cost)
            bandit.update_delivered(arm)
            bandit.update(arm, BASELINE, baseline=BASELINE)
    assert bandit.posterior()[0] == pytest.approx(0.8)

    shares = []
    for _ in range(60):
        bandit.update_submitted("cheap")
        bandit.update_failed("cheap")
        shares.append(bandit.posterior()[0])

    # One failure after twenty deliveries costs it little. Staying down
    # shrinks its share with every failure, down to the probe share.
    assert 0.7 < shares[0] < 0.8
    assert all(later <= earlier for earlier, later in zip(shares, shares[1:]))
    assert shares[20] < 0.3
    assert all(share < 0.05 for share in shares[40:])


def test_bills_of_a_model_that_never_delivered_count_against_it():
    def cost_ratios(failed_bill):
        bandit = AsymmetricUCB(arm_names=["billed", "failing"], cost_aware_coef=1.0)
        for _ in range(3):
            bandit.update_submitted("billed")
            bandit.update_cost("billed", PATCH_COST)
            bandit.update_delivered("billed")
            bandit.update_submitted("failing")
            bandit.update_failed("failing", cost=failed_bill)
        return bandit._normalized_cost_ratio(np.arange(2), 1.0)

    unbilled = cost_ratios(0.0)
    billed = cost_ratios(PATCH_COST)

    # No bill: the reference cost. Three bills, no patch: three times that.
    assert unbilled[1] == pytest.approx(unbilled[0])
    assert billed[1] == pytest.approx(billed[0] / 3)


def test_batch_posterior_gives_a_failing_model_few_of_the_samples():
    bandit = _paid_and_failing(cost_aware_coef=0.0, failed_cost=0.0)

    allocation = np.mean([bandit.posterior(samples=20) for _ in range(50)], axis=0)

    assert allocation.sum() == pytest.approx(1.0)
    assert allocation[DEAD] < 0.15


def test_failure_memory_zero_ignores_saved_outcome_counts():
    saved = AsymmetricUCB(arm_names=ARMS, seed=0)
    for _ in range(20):
        saved.update_failed("dead")
    restored = AsymmetricUCB(arm_names=ARMS, seed=0, failure_memory=0)

    restored.set_state(saved.get_state())

    assert restored.reliability().tolist() == [1.0, 1.0]
    assert restored.recent_failed.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("saved_failed", [40.0, 1e300, 1.7e308])
@ALL_BANDITS
def test_saved_counts_beyond_the_memory_are_scaled_down(make_bandit, saved_failed):
    """Saved with a longer memory, or edited: the arm must still recover."""
    state = make_bandit(0).get_state()
    state["recent_delivered"] = [0.0, saved_failed / 4]
    state["recent_failed"] = [0.0, saved_failed]
    restored = make_bandit(1)

    restored.set_state(state)

    assert restored.recent_failed[DEAD] == pytest.approx(8.0)
    assert restored.recent_delivered[DEAD] == pytest.approx(2.0)
    assert restored.reliability()[DEAD] == pytest.approx(0.2)
    for _ in range(30):
        restored.update_delivered("dead")
    assert restored.reliability()[DEAD] > 0.95


# Posteriors of commit 1bf7113 after the 3rd, 8th and 12th of twelve
# delivered proposals: a pool in which nothing fails selects as before.
POSTERIORS_BEFORE_OUTCOME_REPORTS = {
    "ucb-runner-default": [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
    "ucb-reward-only": [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
    "thompson": [[0.05, 0.9, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
    "fixed": [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
}


@pytest.mark.parametrize("kind", POSTERIORS_BEFORE_OUTCOME_REPORTS)
def test_pool_without_failures_selects_as_before(kind):
    arms = ["a", "b", "c"]
    bandit = {
        "ucb-runner-default": lambda: AsymmetricUCB(
            arm_names=arms, seed=7, cost_aware_coef=0.5
        ),
        "ucb-reward-only": lambda: AsymmetricUCB(arm_names=arms, seed=7),
        "thompson": lambda: ThompsonSampler(arm_names=arms, seed=7),
        "fixed": lambda: FixedSampler(
            arm_names=arms, seed=7, prior_probs=np.array([0.5, 0.3, 0.2])
        ),
    }[kind]()
    bandit.set_baseline_score(1.0)
    posteriors = []
    for step in range(12):
        arm = arms[step % 3]
        bandit.update_submitted(arm)
        bandit.update_cost(arm, 0.01 * (1 + step % 4))
        bandit.update_delivered(arm)
        bandit.update(arm, 1.0 + 0.1 * ((step * 5) % 7) - 0.2, baseline=1.0)
        if step in (2, 7, 11):
            posteriors.append(bandit.posterior().tolist())

    assert posteriors == POSTERIORS_BEFORE_OUTCOME_REPORTS[kind]
