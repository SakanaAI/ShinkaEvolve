"""Loading a linear-mode state saved by the exponential/asymmetric mix.

Before exponential scaling was tied to ``asymmetric_scaling``, a bandit with
``asymmetric_scaling=False`` saved ``s = -inf`` for every arm. Async resume
loads ``bandit_state.pkl`` automatically, so such a state must be detected and
reset rather than interpreted as a linear sum that can never recover.
"""

import logging

import numpy as np
import pytest

from shinka.llm.prioritization import AsymmetricUCB, ThompsonSampler

ARMS = ["a", "b", "c"]
REWARDS = [("a", 1.0), ("b", 5.0), ("c", 0.2), ("a", 1.2), ("b", 4.8), ("c", 0.3)]
LOGGER = "shinka.llm.prioritization"


def _linear(cls):
    return cls(
        arm_names=ARMS,
        seed=0,
        auto_decay=None,
        shift_by_baseline=False,
        shift_by_parent=False,
        asymmetric_scaling=False,
    )


def _state_from_broken_run(cls):
    state = _linear(cls).get_state()
    n = len(ARMS)
    state["s"] = np.full(n, -np.inf)
    state["divs"] = np.full(n, 2.0)
    state["n_submitted"] = np.full(n, 2.0)
    state["n_completed"] = np.full(n, 2.0)
    state["obs_max"] = -np.inf
    state["obs_min"] = -np.inf
    return state


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_broken_linear_state_is_reset_with_a_warning(cls, caplog):
    bandit = _linear(cls)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        bandit.set_state(_state_from_broken_run(cls))

    assert "asymmetric_scaling=False" in caplog.text
    assert np.all(bandit.s == 0.0)
    assert np.all(bandit.divs == 0.0)
    assert np.all(bandit.n_completed == 0.0)

    # Learning resumes from the prior.
    for arm, reward in REWARDS:
        bandit.update(arm, reward=reward)
    assert np.all(np.isfinite(bandit.s))
    if cls is AsymmetricUCB:
        post = bandit.posterior()
        assert post[1] > post[0]
        assert post[1] > post[2]
    else:
        assert bandit.alpha[1] > bandit.alpha[2]


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_valid_linear_state_loads_unchanged(cls, caplog):
    source = _linear(cls)
    for arm, reward in REWARDS[:2]:
        source.update(arm, reward=reward)

    target = _linear(cls)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        target.set_state(source.get_state())

    assert caplog.text == ""
    np.testing.assert_allclose(target.s, source.s)
    np.testing.assert_allclose(target.divs, source.divs)


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_unsampled_arms_in_log_space_state_are_not_reset(cls, caplog):
    # Default mode is exponential + asymmetric, where -inf marks an unsampled arm.
    source = cls(arm_names=ARMS, seed=0, auto_decay=None)
    source.update("a", reward=1.0, baseline=0.0)
    assert np.isneginf(source.s[1:]).all()

    target = cls(arm_names=ARMS, seed=0, auto_decay=None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        target.set_state(source.get_state())

    assert caplog.text == ""
    np.testing.assert_allclose(target.s, source.s)
