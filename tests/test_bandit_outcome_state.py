"""Outcome counts of the selection bandits: settings, state files, summary."""

import logging
from io import StringIO

import numpy as np
import pytest
from rich.console import Console

from shinka.llm.prioritization import AsymmetricUCB, FixedSampler, ThompsonSampler

ARMS = ["ok", "dead"]
DEAD = 1
BASELINE = 1.0

BANDITS = {
    "ucb-runner-default": lambda seed: AsymmetricUCB(
        arm_names=ARMS, seed=seed, cost_aware_coef=0.5
    ),
    "thompson": lambda seed: ThompsonSampler(arm_names=ARMS, seed=seed),
    # The fixed sampler weighs by reliability only when asked to.
    "fixed": lambda seed: FixedSampler(arm_names=ARMS, seed=seed, failure_memory=10),
}
ALL_BANDITS = pytest.mark.parametrize("make_bandit", BANDITS.values(), ids=BANDITS)


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

    # A loaded state keeps the deliveries and half of the failures.
    np.testing.assert_array_equal(restored.recent_failed, bandit.recent_failed / 2)
    np.testing.assert_array_equal(restored.recent_delivered, bandit.recent_delivered)
    assert restored.reliability()[0] == 1.0
    assert bandit.reliability()[DEAD] < restored.reliability()[DEAD] < 0.6


@ALL_BANDITS
def test_model_fixed_between_sessions_is_not_held_to_all_its_failures(
    make_bandit, tmp_path
):
    """The run stopped on a dead model; it is resumed once the model works."""
    stopped = make_bandit(0)
    for _ in range(40):
        stopped.update_delivered("ok")
        stopped.update_failed("dead")
    path = tmp_path / "bandit_state.pkl"
    stopped.save_state(path)
    resumed = make_bandit(1)
    resumed.load_state(path)

    def deliveries_until_reliable(bandit) -> int:
        for delivered in range(1, 30):
            bandit.update_delivered("dead")
            if bandit.reliability()[DEAD] >= 0.5:
                return delivered
        return 30

    assert stopped.reliability()[DEAD] < 0.01
    assert resumed.reliability()[DEAD] == pytest.approx(0.09, abs=0.005)
    assert deliveries_until_reliable(resumed) == 4
    assert deliveries_until_reliable(stopped) == 7


@ALL_BANDITS
def test_restoring_a_state_in_memory_forgives_nothing(make_bandit):
    bandit = make_bandit(0)
    for _ in range(4):
        bandit.update_failed("dead")
    restored = make_bandit(1)

    restored.set_state(bandit.get_state())

    np.testing.assert_array_equal(restored.recent_failed, bandit.recent_failed)
    np.testing.assert_array_equal(restored.reliability(), bandit.reliability())


@ALL_BANDITS
def test_outcome_counts_follow_the_model_when_the_pool_changes(make_bandit, tmp_path):
    bandit = make_bandit(0)
    for _ in range(4):
        bandit.update_failed("dead")
    path = tmp_path / "bandit_state.pkl"
    bandit.save_state(path)

    reordered = type(bandit)(
        arm_names=["new", "dead", "ok"], seed=0, failure_memory=10
    )
    reordered.load_state(path)

    assert reordered.recent_failed.tolist() == [0.0, bandit.recent_failed[DEAD] / 2, 0.0]
    assert reordered.reliability()[0] == 1.0
    assert reordered.reliability()[1] < 0.5
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


@pytest.mark.parametrize(
    "bad", [["a", "b"], 5.0, None, [0.0, 10**400]], ids=["names", "scalar", "null", "huge"]
)
@ALL_BANDITS
def test_unreadable_saved_outcome_counts_are_reset_not_raised(make_bandit, bad, caplog):
    """Raising here would leave a bandit that restored everything else, with
    the log saying its state could not be loaded."""
    saved = make_bandit(0)
    saved.set_baseline_score(BASELINE)
    saved.update_submitted("ok")
    saved.update("ok", BASELINE + 0.1, baseline=BASELINE)
    saved.update_failed("dead")
    state = saved.get_state()
    state["recent_failed"] = bad
    restored = make_bandit(1)

    with caplog.at_level(logging.WARNING, logger="shinka.llm.prioritization"):
        restored.set_state(state)

    assert "invalid proposal outcome counts" in caplog.text
    assert restored.reliability().tolist() == [1.0, 1.0]
    assert restored.get_state()["baseline"] == BASELINE


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

    # A reliability that would print as 1.00 is not worth a caption; it gets
    # there long before it is exactly 1.0 again.
    for _ in range(60):
        bandit.update_delivered("dead")
    assert 0.995 < bandit.reliability()[DEAD] < 1.0
    assert "returned a patch" not in summary()


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


@pytest.mark.parametrize("failure_memory", [3.0, 7.0, 10.0, 13.0])
@ALL_BANDITS
def test_save_and_load_leave_the_counts_bit_for_bit(
    make_bandit, failure_memory, tmp_path
):
    """Rounding puts a full memory a few ulp above failure_memory."""
    template = make_bandit(0)
    bandit = type(template)(arm_names=ARMS, seed=0, failure_memory=failure_memory)
    world = np.random.default_rng(int(failure_memory))
    path = tmp_path / "bandit_state.pkl"
    for _ in range(300):
        if world.random() < 0.4:
            bandit.update_failed("dead")
        else:
            bandit.update_delivered("dead")
    bandit.save_state(path)

    restored = type(template)(arm_names=ARMS, seed=1, failure_memory=failure_memory)
    restored.load_state(path)

    # Halving is exact, so the saved counts came back bit for bit.
    np.testing.assert_array_equal(restored.recent_failed * 2, bandit.recent_failed)
    np.testing.assert_array_equal(restored.recent_delivered, bandit.recent_delivered)
