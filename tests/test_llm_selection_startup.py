"""The runner builds every model-selection bandit from the default config."""

import pytest

from shinka.core import EvolutionConfig
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.llm.prioritization import AsymmetricUCB, FixedSampler, ThompsonSampler

MODELS = ["model-a", "model-b"]


def _selection(name, **config):
    evo_config = EvolutionConfig(
        llm_models=MODELS, llm_dynamic_selection=name, **config
    )
    return ShinkaEvolveRunner._llm_selection_from(evo_config)


@pytest.mark.parametrize(
    "name, bandit_class",
    [
        ("ucb", AsymmetricUCB),
        ("ucb1", AsymmetricUCB),
        ("UCB", AsymmetricUCB),
        ("fixed", FixedSampler),
        ("thompson", ThompsonSampler),
        ("Thompson", ThompsonSampler),
    ],
)
def test_every_selection_starts_with_the_default_kwargs(name, bandit_class):
    """The default kwargs set UCB's cost term, which only UCB has."""
    bandit = _selection(name)

    assert type(bandit) is bandit_class
    assert bandit.n_arms == 2
    one_hot, probabilities = bandit.select_llm()
    assert one_hot.sum() == 1.0
    assert probabilities.sum() == pytest.approx(1.0)


def test_default_ucb_keeps_its_cost_term():
    assert _selection("ucb").cost_aware_coefficient == 0.5


def test_thompson_takes_its_own_kwargs_and_drops_the_cost_settings():
    bandit = _selection(
        "thompson",
        llm_dynamic_selection_kwargs={
            "cost_aware_coef": 0.5,
            "cost_exploration_coef": 0.1,
            "cost_power": 1.0,
            "cost_ref_percentile": 50.0,
            "epsilon": 0.3,
            "failure_memory": 0,
        },
    )

    assert bandit.epsilon == 0.3
    assert bandit._failure_memory == 0.0


def test_thompson_still_refuses_a_kwarg_it_does_not_know():
    with pytest.raises(TypeError, match="exploration_coef"):
        _selection("thompson", llm_dynamic_selection_kwargs={"exploration_coef": 1.0})


def test_fixed_selection_weighs_by_reliability_only_when_asked():
    assert _selection("fixed")._failure_memory == 0.0
    weighted = _selection("fixed", llm_dynamic_selection_kwargs={"failure_memory": 10})
    assert weighted._failure_memory == 10.0


def test_no_selection_and_a_ready_made_bandit_pass_through():
    ready_made = FixedSampler(arm_names=MODELS)

    assert _selection(None) is None
    assert _selection(ready_made) is ready_made


def test_unknown_selection_is_refused():
    with pytest.raises(ValueError, match="Invalid llm_dynamic_selection"):
        _selection("greedy")
