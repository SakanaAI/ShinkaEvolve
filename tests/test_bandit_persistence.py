"""
Test script to verify bandit state persistence works correctly.
"""

from io import StringIO
import numpy as np
from pathlib import Path
import tempfile
from rich.console import Console
from shinka.llm import AsymmetricUCB, ThompsonSampler, FixedSampler


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
