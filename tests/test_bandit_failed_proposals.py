"""A model whose proposals fail loses selection weight, and gets it back."""

from shinka.llm.prioritization import AsymmetricUCB

PATCH_COST = 0.02


def test_arm_without_a_cost_report_is_not_treated_as_free():
    """Its proposal may be in flight, or may have failed before any bill."""
    bandit = AsymmetricUCB(arm_names=["billed", "unknown"], cost_aware_coef=1.0)
    for _ in range(3):
        bandit.update_submitted("billed")
        bandit.update_cost("billed", PATCH_COST)
        bandit.update_submitted("unknown")

    billed, unknown = bandit.posterior()

    assert unknown <= billed
