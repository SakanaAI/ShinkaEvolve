"""
Test script to verify bandit state persistence works correctly.
"""

from io import StringIO
import numpy as np
from pathlib import Path
import pickle
import tempfile
from rich.console import Console
from shinka.llm import AsymmetricUCB, ThompsonSampler, FixedSampler


def _exercise_sampler_after_resize(bandit):
    posterior = bandit.posterior()
    one_hot, selected_posterior = bandit.select_llm()

    assert posterior.shape == (bandit.n_arms,)
    assert one_hot.shape == (bandit.n_arms,)
    assert selected_posterior.shape == (bandit.n_arms,)
    assert np.isclose(posterior.sum(), 1.0)


def _save_state_without_arm_names(bandit, path: Path) -> dict:
    state = bandit.get_state()
    state.pop("arm_names", None)
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return state


def test_asymmetric_ucb_persistence():
    """Test AsymmetricUCB save/load."""
    print("Testing AsymmetricUCB persistence...")

    # Create a bandit with some arm names
    arm_names = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]
    bandit = AsymmetricUCB(
        arm_names=arm_names,
        exploration_coef=2.0,
        epsilon=0.1,
        auto_decay=0.95,
    )

    # Simulate some updates
    bandit.set_baseline_score(0.5)
    bandit.update_submitted("gpt-4o")
    bandit.update("gpt-4o", reward=0.8, baseline=0.5)
    bandit.update_submitted("gpt-4o-mini")
    bandit.update("gpt-4o-mini", reward=0.6, baseline=0.5)
    bandit.update_submitted("claude-3-5-sonnet")
    bandit.update("claude-3-5-sonnet", reward=0.9, baseline=0.5)

    print("Original bandit state:")
    bandit.print_summary()

    # Save state
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        bandit.save_state(save_path)
        print(f"\nSaved to {save_path}")

        # Create a new bandit and load state
        bandit2 = AsymmetricUCB(
            arm_names=arm_names,
            exploration_coef=2.0,
            epsilon=0.1,
            auto_decay=0.95,
        )
        bandit2.load_state(save_path)

        print("\nLoaded bandit state:")
        bandit2.print_summary()

        # Verify states match
        assert np.allclose(bandit.n_submitted, bandit2.n_submitted), (
            "n_submitted mismatch!"
        )
        assert np.allclose(bandit.n_completed, bandit2.n_completed), (
            "n_completed mismatch!"
        )
        assert np.allclose(bandit.s, bandit2.s), "s mismatch!"
        assert np.allclose(bandit.divs, bandit2.divs), "divs mismatch!"
        assert bandit._baseline == bandit2._baseline, "baseline mismatch!"
        assert bandit._obs_max == bandit2._obs_max, "obs_max mismatch!"
        assert bandit._obs_min == bandit2._obs_min, "obs_min mismatch!"

        print("✅ AsymmetricUCB persistence test passed!")


def test_thompson_sampler_persistence():
    """Test ThompsonSampler save/load."""
    print("\n" + "=" * 80)
    print("Testing ThompsonSampler persistence...")

    arm_names = ["model-a", "model-b", "model-c"]
    bandit = ThompsonSampler(
        arm_names=arm_names,
        epsilon=0.1,
        prior_alpha=1.0,
        prior_beta=1.0,
        auto_decay=0.95,
    )

    # Simulate some updates
    bandit.set_baseline_score(0.3)
    bandit.update_submitted("model-a")
    bandit.update("model-a", reward=0.7, baseline=0.3)
    bandit.update_submitted("model-b")
    bandit.update("model-b", reward=0.5, baseline=0.3)

    print("Original bandit state:")
    bandit.print_summary()

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "thompson_state.pkl"
        bandit.save_state(save_path)

        bandit2 = ThompsonSampler(
            arm_names=arm_names,
            epsilon=0.1,
            prior_alpha=1.0,
            prior_beta=1.0,
            auto_decay=0.95,
        )
        bandit2.load_state(save_path)

        print("\nLoaded bandit state:")
        bandit2.print_summary()

        # Verify states match
        assert np.allclose(bandit.alpha, bandit2.alpha), "alpha mismatch!"
        assert np.allclose(bandit.beta, bandit2.beta), "beta mismatch!"
        assert bandit._baseline == bandit2._baseline, "baseline mismatch!"

        print("✅ ThompsonSampler persistence test passed!")


def test_fixed_sampler_persistence():
    """Test FixedSampler save/load."""
    print("\n" + "=" * 80)
    print("Testing FixedSampler persistence...")

    arm_names = ["model-x", "model-y"]
    prior_probs = np.array([0.7, 0.3])
    bandit = FixedSampler(
        arm_names=arm_names,
        prior_probs=prior_probs,
    )

    bandit.set_baseline_score(0.5)

    print("Original bandit state:")
    bandit.print_summary()

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "fixed_state.pkl"
        bandit.save_state(save_path)

        bandit2 = FixedSampler(
            arm_names=arm_names,
            prior_probs=prior_probs,
        )
        bandit2.load_state(save_path)

        print("\nLoaded bandit state:")
        bandit2.print_summary()

        # Verify states match
        assert np.allclose(bandit.p, bandit2.p), "probabilities mismatch!"
        assert bandit._baseline == bandit2._baseline, "baseline mismatch!"

        print("✅ FixedSampler persistence test passed!")


def test_bandit_persistence_restores_next_random_selection(tmp_path):
    source = FixedSampler(
        arm_names=["a", "b", "c"],
        prior_probs=np.array([0.2, 0.3, 0.5]),
        seed=123,
    )
    source.select_llm()
    state_path = tmp_path / "bandit_state.pkl"
    source.save_state(state_path)
    expected_one_hot, expected_posterior = source.select_llm()

    restored = FixedSampler(
        arm_names=["a", "b", "c"],
        prior_probs=np.array([0.2, 0.3, 0.5]),
        seed=999,
    )
    restored.load_state(state_path)
    actual_one_hot, actual_posterior = restored.select_llm()

    assert np.array_equal(actual_one_hot, expected_one_hot)
    assert np.array_equal(actual_posterior, expected_posterior)


def test_asymmetric_ucb_loads_state_into_more_arms():
    source = AsymmetricUCB(arm_names=["a", "b", "c"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("b")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = AsymmetricUCB(arm_names=["a", "b", "c", "d", "e"], seed=1)
        target.load_state(save_path)

    assert target.n_submitted.shape == (5,)
    assert np.allclose(target.n_submitted[:3], source.n_submitted)
    assert np.allclose(target.n_submitted[3:], 0.0)
    assert np.all(np.isneginf(target.s[3:]))
    _exercise_sampler_after_resize(target)


def test_asymmetric_ucb_loads_state_into_fewer_arms():
    source = AsymmetricUCB(arm_names=["a", "b", "c", "d", "e"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("d")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = AsymmetricUCB(arm_names=["a", "b", "c"], seed=1)
        target.load_state(save_path)

    assert target.n_submitted.shape == (3,)
    assert np.allclose(target.n_submitted, source.n_submitted[:3])
    _exercise_sampler_after_resize(target)


def test_thompson_sampler_loads_state_into_more_arms():
    source = ThompsonSampler(arm_names=["a", "b", "c"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("b")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = ThompsonSampler(
            arm_names=["a", "b", "c", "d", "e"],
            seed=1,
            prior_alpha=2.0,
            prior_beta=3.0,
        )
        target.load_state(save_path)

    assert target.alpha.shape == (5,)
    assert np.allclose(target.alpha[:3], source.alpha)
    assert np.allclose(target.beta[:3], source.beta)
    assert np.allclose(target.alpha[3:], 2.0)
    assert np.allclose(target.beta[3:], 3.0)
    _exercise_sampler_after_resize(target)


def test_thompson_sampler_loads_state_into_fewer_arms():
    source = ThompsonSampler(arm_names=["a", "b", "c", "d", "e"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("d")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = ThompsonSampler(arm_names=["a", "b", "c"], seed=1)
        target.load_state(save_path)

    assert target.alpha.shape == (3,)
    assert np.allclose(target.alpha, source.alpha[:3])
    assert np.allclose(target.beta, source.beta[:3])
    _exercise_sampler_after_resize(target)


def test_fixed_sampler_loads_state_into_more_arms():
    source = FixedSampler(
        arm_names=["a", "b", "c"],
        prior_probs=np.array([0.2, 0.3, 0.5]),
        seed=0,
    )
    source.update("a", reward=1.0)
    source.update_cost("b", cost=0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = FixedSampler(
            arm_names=["a", "b", "c", "d", "e"],
            prior_probs=np.array([0.1, 0.2, 0.3, 0.15, 0.25]),
            seed=1,
        )
        target.load_state(save_path)

    expected_p = np.array([0.2, 0.3, 0.5, 0.15, 0.25])
    expected_p = expected_p / expected_p.sum()
    assert target.p.shape == (5,)
    assert np.allclose(target.p, expected_p)
    assert np.allclose(target.n_pulls[:3], source.n_pulls)
    assert np.allclose(target.n_pulls[3:], 0.0)
    _exercise_sampler_after_resize(target)


def test_fixed_sampler_loads_state_into_fewer_arms():
    source = FixedSampler(
        arm_names=["a", "b", "c", "d", "e"],
        prior_probs=np.array([0.1, 0.2, 0.3, 0.15, 0.25]),
        seed=0,
    )
    source.update("a", reward=1.0)
    source.update_cost("d", cost=0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = FixedSampler(
            arm_names=["a", "b", "c"],
            prior_probs=np.array([0.2, 0.3, 0.5]),
            seed=1,
        )
        target.load_state(save_path)

    expected_p = np.array([0.1, 0.2, 0.3])
    expected_p = expected_p / expected_p.sum()
    assert target.p.shape == (3,)
    assert np.allclose(target.p, expected_p)
    assert np.allclose(target.n_pulls, source.n_pulls[:3])
    _exercise_sampler_after_resize(target)


def test_asymmetric_ucb_restores_cost_range_for_cost_aware_resume():
    source = AsymmetricUCB(
        arm_names=["cheap", "expensive"],
        cost_aware_coef=0.5,
        auto_decay=None,
        exponential_base=None,
    )
    source.update("cheap", reward=1.0, baseline=0.0)
    source.update("expensive", reward=1.0, baseline=0.0)
    source.update_cost("cheap", cost=1.0)
    source.update_cost("expensive", cost=100.0)
    expected = source.posterior()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = AsymmetricUCB(
            arm_names=["cheap", "expensive"],
            cost_aware_coef=0.5,
            auto_decay=None,
            exponential_base=None,
        )
        target.load_state(save_path)

    assert target.min_cost_observed == source.min_cost_observed
    assert target.max_cost_observed == source.max_cost_observed
    assert np.allclose(target.posterior(), expected)


def test_named_bandit_state_aligns_when_model_order_changes():
    source = ThompsonSampler(arm_names=["a", "b", "c"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("c")
    source.update("c", reward=0.6, baseline=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = ThompsonSampler(arm_names=["c", "a", "d"], seed=1)
        target.load_state(save_path)

    assert np.allclose(
        target.n_submitted,
        [source.n_submitted[2], source.n_submitted[0], 0.0],
    )
    assert np.allclose(
        target.alpha,
        [source.alpha[2], source.alpha[0], target.a_prior],
    )
    assert np.allclose(target.beta, [source.beta[2], source.beta[0], target.b_prior])
    _exercise_sampler_after_resize(target)


def test_asymmetric_ucb_named_state_aligns_when_model_order_changes():
    source = AsymmetricUCB(arm_names=["a", "b", "c"], seed=0)
    source.update_submitted("a")
    source.update("a", reward=0.8, baseline=0.0)
    source.update_submitted("c")
    source.update_cost("c", cost=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = AsymmetricUCB(arm_names=["c", "a", "d"], seed=1)
        target.load_state(save_path)

    assert np.allclose(
        target.n_submitted,
        [source.n_submitted[2], source.n_submitted[0], 0.0],
    )
    assert np.allclose(target.s, [source.s[2], source.s[0], -np.inf])
    assert np.allclose(target.n_costs, [source.n_costs[2], source.n_costs[0], 0.0])
    assert np.allclose(
        target.total_costs,
        [source.total_costs[2], source.total_costs[0], 0.0],
    )
    _exercise_sampler_after_resize(target)


def test_fixed_sampler_named_state_aligns_when_model_order_changes():
    source = FixedSampler(
        arm_names=["a", "b", "c"],
        prior_probs=np.array([0.2, 0.3, 0.5]),
        seed=0,
    )
    source.update("a", reward=1.0)
    source.update_cost("c", cost=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = FixedSampler(
            arm_names=["c", "a", "d"],
            prior_probs=np.array([0.6, 0.3, 0.1]),
            seed=1,
        )
        target.load_state(save_path)

    expected_p = np.array([0.5, 0.2, 0.1])
    expected_p = expected_p / expected_p.sum()
    assert np.allclose(target.p, expected_p)
    assert np.allclose(target.n_pulls, [source.n_pulls[2], source.n_pulls[0], 0.0])
    assert np.allclose(target.n_costs, [source.n_costs[2], source.n_costs[0], 0.0])
    assert np.allclose(
        target.total_costs,
        [source.total_costs[2], source.total_costs[0], 0.0],
    )
    _exercise_sampler_after_resize(target)


def test_legacy_bandit_state_without_arm_names_uses_prefix_alignment():
    source = AsymmetricUCB(arm_names=["old-a", "old-b", "old-c"], seed=0)
    source.update_submitted("old-a")
    source.update("old-a", reward=0.8, baseline=0.0)
    source.update_submitted("old-c")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "legacy_bandit_state.pkl"
        state = _save_state_without_arm_names(source, save_path)

        target = AsymmetricUCB(
            arm_names=["new-a", "new-b", "new-c", "new-d"],
            seed=1,
        )
        target.load_state(save_path)

    assert "arm_names" not in state
    assert np.allclose(target.n_submitted[:3], source.n_submitted)
    assert np.allclose(target.n_submitted[3], 0.0)
    _exercise_sampler_after_resize(target)


def test_legacy_thompson_state_without_arm_names_uses_prefix_alignment():
    source = ThompsonSampler(arm_names=["old-a", "old-b", "old-c"], seed=0)
    source.update_submitted("old-a")
    source.update("old-a", reward=0.8, baseline=0.0)
    source.update_submitted("old-c")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "legacy_thompson_state.pkl"
        state = _save_state_without_arm_names(source, save_path)

        target = ThompsonSampler(
            arm_names=["new-a", "new-b", "new-c", "new-d"],
            seed=1,
        )
        target.load_state(save_path)

    assert "arm_names" not in state
    assert np.allclose(target.n_submitted[:3], source.n_submitted)
    assert np.allclose(target.n_submitted[3], 0.0)
    assert np.allclose(target.alpha[:3], source.alpha)
    assert np.allclose(target.beta[:3], source.beta)
    assert target.alpha[3] == target.a_prior
    assert target.beta[3] == target.b_prior
    _exercise_sampler_after_resize(target)


def test_legacy_fixed_state_without_arm_names_uses_prefix_alignment():
    source = FixedSampler(
        arm_names=["old-a", "old-b", "old-c"],
        prior_probs=np.array([0.2, 0.3, 0.5]),
        seed=0,
    )
    source.update("old-a", reward=1.0)
    source.update_cost("old-c", cost=0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "legacy_fixed_state.pkl"
        state = _save_state_without_arm_names(source, save_path)

        target = FixedSampler(
            arm_names=["new-a", "new-b", "new-c", "new-d"],
            prior_probs=np.array([0.1, 0.2, 0.3, 0.4]),
            seed=1,
        )
        target.load_state(save_path)

    expected_p = np.array([0.2, 0.3, 0.5, 0.4])
    expected_p = expected_p / expected_p.sum()
    assert "arm_names" not in state
    assert np.allclose(target.p, expected_p)
    assert np.allclose(target.n_pulls[:3], source.n_pulls)
    assert np.allclose(target.n_pulls[3], 0.0)
    assert np.allclose(target.n_costs[:3], source.n_costs)
    assert np.allclose(target.total_costs[:3], source.total_costs)
    _exercise_sampler_after_resize(target)


def test_asymmetric_ucb_recomputes_observation_range_when_dropping_arms():
    source = AsymmetricUCB(
        arm_names=["a", "b", "c", "d"],
        auto_decay=None,
        asymmetric_scaling=False,
        exponential_base=None,
        shift_by_baseline=False,
        shift_by_parent=False,
    )
    source.update("a", reward=1.0)
    source.update("d", reward=100.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = AsymmetricUCB(
            arm_names=["a", "b", "c"],
            auto_decay=None,
            asymmetric_scaling=False,
            exponential_base=None,
            shift_by_baseline=False,
            shift_by_parent=False,
        )
        target.load_state(save_path)

    assert target._obs_max == 1.0
    assert target._obs_min == 1.0


def test_thompson_sampler_recomputes_observation_range_when_dropping_arms():
    source = ThompsonSampler(
        arm_names=["a", "b", "c", "d"],
        auto_decay=None,
        asymmetric_scaling=False,
        exponential_base=None,
        shift_by_baseline=False,
        shift_by_parent=False,
    )
    source.update("a", reward=1.0)
    source.update("d", reward=100.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bandit_state.pkl"
        source.save_state(save_path)

        target = ThompsonSampler(
            arm_names=["a", "b", "c"],
            auto_decay=None,
            asymmetric_scaling=False,
            exponential_base=None,
            shift_by_baseline=False,
            shift_by_parent=False,
        )
        target.load_state(save_path)

    assert target._obs_max == 1.0
    assert target._obs_min == 1.0


def test_asymmetric_ucb_print_summary_preserves_local_model_name():
    arm_name = (
        "local/example-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
    )
    bandit = AsymmetricUCB(
        arm_names=[arm_name],
        exploration_coef=2.0,
        epsilon=0.1,
        auto_decay=0.95,
    )

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    bandit.print_summary(console=console)
    output = buffer.getvalue()

    assert "local/example-mod" in output
    assert "api.example.test" not in output
    assert "CUSTOM_API_KEY" not in output


def test_asymmetric_ucb_print_summary_preserves_openrouter_prefix():
    arm_name = "openrouter/example-model"
    bandit = AsymmetricUCB(
        arm_names=[arm_name],
        exploration_coef=2.0,
        epsilon=0.1,
        auto_decay=0.95,
    )

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    bandit.print_summary(console=console)
    output = buffer.getvalue()

    assert "openrouter/exampl" in output


def test_asymmetric_ucb_print_summary_matches_standard_summary_width():
    bandit = AsymmetricUCB(
        arm_names=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
        exploration_coef=2.0,
        epsilon=0.1,
        auto_decay=0.95,
    )

    bandit.set_baseline_score(0.5)
    bandit.update_submitted("gpt-4o")
    bandit.update("gpt-4o", reward=0.8, baseline=0.5)
    bandit.update_submitted("gpt-4o-mini")
    bandit.update("gpt-4o-mini", reward=0.6, baseline=0.5)

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    bandit.print_summary(console=console)
    output_lines = buffer.getvalue().splitlines()

    assert max(len(line) for line in output_lines) == 120


def test_asymmetric_ucb_print_summary_omits_div_and_log_mean_columns():
    bandit = AsymmetricUCB(
        arm_names=["gpt-4o"],
        exploration_coef=2.0,
        epsilon=0.1,
        auto_decay=0.95,
    )

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    bandit.print_summary(console=console)
    output = buffer.getvalue()

    assert "│ div " not in output
    assert "log mean" not in output


if __name__ == "__main__":
    test_asymmetric_ucb_persistence()
    test_thompson_sampler_persistence()
    test_fixed_sampler_persistence()
    print("\n" + "=" * 80)
    print("🎉 All bandit persistence tests passed!")
    print("=" * 80)
