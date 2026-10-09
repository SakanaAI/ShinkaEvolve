"""decay() must be silent for arms that have never been sampled.

Unsampled arms sit at ``s = -inf`` in log-space. ``np.where`` evaluates both
branches, so the unused branch computes ``-inf + inf`` for them; the value is
discarded, and the guard keeps NumPy from reporting it.
"""

import warnings

import numpy as np
import pytest

from shinka.llm.prioritization import AsymmetricUCB, ThompsonSampler


@pytest.mark.parametrize("cls", [AsymmetricUCB, ThompsonSampler])
def test_decay_is_silent_with_unsampled_arms(cls):
    bandit = cls(n_arms=3, seed=0, auto_decay=None)
    bandit.update(0, reward=1.0, baseline=0.0)
    assert np.isneginf(bandit.s[1:]).all()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        bandit.decay(0.95)

    # The sampled arm shrinks toward the prior; unsampled arms stay unsampled.
    assert np.isfinite(bandit.s[0])
    assert np.isneginf(bandit.s[1:]).all()
