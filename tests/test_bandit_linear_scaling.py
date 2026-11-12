"""Exponential scaling is only defined together with asymmetric clamping.

With ``asymmetric_scaling=False`` the accumulator ``s`` is updated linearly, so
it has to be initialised and read linearly too, whatever ``exponential_base``
is (it defaults to 1.0). Previously ``s`` started at ``-inf`` in that mode and
never left it, which collapsed model selection to uniform.
"""

import numpy as np
import pytest

from shinka.llm.prioritization import AsymmetricUCB, ThompsonSampler

REWARDS = [(0, 1.0), (1, 5.0), (2, 0.2), (0, 1.2), (1, 4.8), (2, 0.3)]


def _make(cls, **kwargs):
    return cls(
        n_arms=3,
        seed=0,
        auto_decay=None,
        shift_by_baseline=False,
        shift_by_parent=False,
        asymmetric_scaling=False,
        **kwargs,
    )


def _feed(bandit):
    for arm, reward in REWARDS:
        bandit.update(arm, reward=reward)
    return bandit


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_linear_mode_is_independent_of_exponential_base(cls):
    with_default = _feed(_make(cls))  # exponential_base left at its default
    explicit_none = _feed(_make(cls, exponential_base=None))

    assert not with_default.use_exponential_scaling
    assert np.all(np.isfinite(with_default.s))
    np.testing.assert_allclose(with_default.s, explicit_none.s)
    np.testing.assert_allclose(with_default._mean(), explicit_none._mean())
    np.testing.assert_allclose(with_default.posterior(), explicit_none.posterior())


def test_ucb_linear_mode_prefers_the_better_arm():
    bandit = _feed(_make(AsymmetricUCB))
    post = bandit.posterior()
    assert post[1] > post[0]
    assert post[1] > post[2]


def test_thompson_linear_mode_credits_the_better_arm():
    bandit = _feed(_make(ThompsonSampler))
    assert bandit.alpha[1] > bandit.alpha[2]


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_linear_mode_survives_decay(cls):
    bandit = _feed(_make(cls))
    bandit.decay(0.95)

    assert np.all(np.isfinite(bandit.s))
    assert np.isfinite(bandit._obs_min)
    assert np.isfinite(bandit._obs_max)
    assert bandit._have_obs_range()
