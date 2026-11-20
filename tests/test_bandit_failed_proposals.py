"""A model whose proposals fail loses selection weight, and gets it back."""

import numpy as np
import pytest

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
    # All that is left is the probe share: 1 / (1 + failure_memory) of an
    # even share.
    for _ in range(20):
        ok, dead = bandit.posterior()
        assert dead == pytest.approx(1 / 22)
        assert ok == pytest.approx(21 / 22)


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
    reliability = bandit.reliability()[0]

    assert reliability == pytest.approx(0.81 / 2.71)
    # `a` keeps the reliable part of its half; `b` and `c` share the rest 3:2.
    lost = 0.5 * (1 - reliability)
    weighted = np.array([0.5 * reliability, 0.3 + 0.6 * lost, 0.2 + 0.4 * lost])
    probe = (1 / 33) * (1 - reliability)
    expected = weighted * (1 - probe) + [probe, 0.0, 0.0]
    np.testing.assert_allclose(bandit.posterior(), expected)
    assert bandit.posterior().sum() == pytest.approx(1.0)

    # Restricted to two arms, `b` takes everything `a` loses.
    probe = (1 / 22) * (1 - reliability)
    kept = 0.625 * reliability
    np.testing.assert_allclose(
        bandit.posterior(subset=["a", "b"]),
        [kept * (1 - probe) + probe, (1 - kept) * (1 - probe), 0.0],
    )


@pytest.mark.parametrize("prior", [[0.99, 0.01], [0.5, 0.49, 0.01], [0.9, 0.05, 0.05]])
def test_failing_never_raises_the_share_a_fixed_sampler_gives_a_model(prior):
    arms = [f"arm-{i}" for i in range(len(prior))]
    bandit = FixedSampler(arm_names=arms, prior_probs=prior)
    shares = []
    for _ in range(80):
        bandit.update_failed(arms[-1])
        shares.append(bandit.posterior()[-1])

    assert max(shares) <= prior[-1]
    # What is left is the probe share, or all of a share that is smaller.
    assert shares[-1] == pytest.approx(min(prior[-1], 1 / (11 * len(prior))), abs=1e-4)
    # The arms that still work keep their proportions.
    working = bandit.posterior()[:-1]
    np.testing.assert_allclose(
        working / working.sum(), np.array(prior[:-1]) / sum(prior[:-1])
    )


def test_fixed_sampler_never_draws_a_model_it_gives_no_probability():
    bandit = FixedSampler(arm_names=["a", "b", "off"], prior_probs=[0.5, 0.5, 0.0])
    for _ in range(30):
        bandit.update_failed("a")
        bandit.update_failed("off")

    assert bandit.posterior()[2] == 0.0
    assert bandit.posterior(subset=["a", "off"])[2] == 0.0


@pytest.mark.parametrize(
    "make_bandit",
    [
        lambda: AsymmetricUCB(
            arm_names=ARMS, seed=0, epsilon=0.0, exploration_coef=0.0
        ),
        lambda: AsymmetricUCB(arm_names=ARMS, seed=0, epsilon=0.0, cost_aware_coef=1.0),
        lambda: ThompsonSampler(arm_names=ARMS, seed=0, epsilon=0.0),
    ],
    ids=["ucb-reward-only", "ucb-cost-only", "thompson"],
)
def test_greedy_bandit_lets_go_of_its_best_model_when_it_dies(make_bandit):
    """With epsilon 0 the best arm holds all of the probability."""
    bandit = make_bandit()
    bandit.set_baseline_score(BASELINE)
    for _ in range(10):
        for arm, reward, cost in (
            ("dead", BASELINE + 1.0, 0.0002),
            ("ok", BASELINE, 0.02),
        ):
            bandit.update_submitted(arm)
            bandit.update_cost(arm, cost)
            bandit.update_delivered(arm)
            bandit.update(arm, reward, baseline=BASELINE)
    shares = []
    for _ in range(40):
        bandit.update_submitted("dead")
        bandit.update_failed("dead")
        shares.append(np.mean([bandit.posterior()[DEAD] for _ in range(20)]))

    assert shares[9] < 0.45
    assert shares[19] < 0.2
    assert shares[-1] < 0.06


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

    # At best `dead` ties with `ok`, which has earned no reward either. Of
    # that even share it keeps the reliable part, 0.13, and the probe share.
    assert ok > 0.89
    assert dead < 0.11


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


@pytest.mark.parametrize("cost_aware_coef", [0.0, 0.5])
def test_one_failure_does_not_take_the_lead_from_the_best_model(cost_aware_coef):
    """A failure costs a model that share of its probability, not its rank."""
    arms = ["best", "second", "third"]
    bandit = AsymmetricUCB(arm_names=arms, seed=0, cost_aware_coef=cost_aware_coef)
    bandit.set_baseline_score(BASELINE)
    for _ in range(20):
        for arm, gain in (("best", 0.25), ("second", 0.2), ("third", 0.2)):
            bandit.update_submitted(arm)
            bandit.update_cost(arm, PATCH_COST)
            bandit.update_delivered(arm)
            bandit.update(arm, BASELINE + gain, baseline=BASELINE)
    assert bandit.posterior()[0] == pytest.approx(0.8)

    bandit.update_submitted("best")
    bandit.update_failed("best")

    reliability = bandit.reliability()[0]
    assert reliability == pytest.approx(0.89, abs=0.01)
    # Still the best arm: it keeps 1 - epsilon, less the unreliable part.
    assert bandit.posterior()[0] == pytest.approx(0.8 * reliability, abs=0.01)


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
    assert shares[20] < 0.15
    assert all(share < 0.06 for share in shares[40:])


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
