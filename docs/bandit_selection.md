# UCB1 Bandit LLM Selection

This interactive demo mirrors Shinka's current `AsymmetricUCB`-style model
selection behavior with synthetic observations. It is a teaching model rather
than a replay of runtime logs.

Use the slider to change `cost_aware_coef` and watch the favored model move
across the cost-versus-reward plot.

The reward-side score combines normalized utility and exploration pressure,
while the cost-side score reflects optimistic cheapness.

<div id="bandit-selection-widget"></div>

---

## Score Sketch

The demo tracks the same high-level quantities as `AsymmetricUCB`:

```text
reward_score_i = normalized_reward_i + exploration_coef * sqrt(2 log t / n_i)
cost_bonus_i = cost_exploration_coef * cost_range * sqrt(2 log t / n_cost_i)
cost_ratio_i = cost_ref / max(mean_cost_i - cost_bonus_i, 1e-7)
cost_score_i = (cost_ratio_i / max_j cost_ratio_j) ^ cost_power
final_score_i = (1 - cost_aware_coef) * reward_score_i
              + cost_aware_coef * cost_score_i
```

Where:

- `t`: total pulls across all models.
- `n_i`: submitted/completed pulls for model `i`.
- `normalized_reward_i`: the arm's shifted, adaptively scaled reward signal.
- `cost_ref`: percentile reference across observed mean costs.
- `cost_range`: `max_cost_observed - min_cost_observed`.

Selection then follows epsilon-greedy tie handling:

```text
if unseen models exist:
    sample uniformly among unseen models
else:
    winners = argmax_i final_score_i
    p(i) = (1 - epsilon) / |winners|     if i in winners
         = epsilon / (K - |winners|)     otherwise
```

This is why the slider matters: increasing `cost_aware_coef` does not change
the exploration term itself; it changes how much the final score trusts the
cheapness side relative to reward plus exploration.

---

## Quick Start

```python
from shinka.core import EvolutionConfig


evo_config = EvolutionConfig(
    llm_models=["gpt-5-mini", "gemini-3.1-pro-preview", "gpt-5.4"],
    llm_dynamic_selection="ucb",
    llm_dynamic_selection_kwargs={
        "cost_aware_coef": 0.5,
        "exploration_coef": 1.0,
        "epsilon": 0.2,
    },
)
```

`llm_dynamic_selection="ucb"` and `llm_dynamic_selection="ucb1"` both route to
`AsymmetricUCB`.

---

## Recommended Settings

| Goal | `cost_aware_coef` | `exploration_coef` | `epsilon` | Notes |
|------|-------------------|--------------------|-----------|-------|
| Reward-first | `0.0-0.2` | `1.0` | `0.1-0.2` | favors strongest models unless cost gaps are extreme |
| Balanced default | `0.3-0.6` | `1.0` | `0.2` | good starting point for mixed quality/cost portfolios |
| Budget-sensitive | `0.7-0.9` | `0.8-1.0` | `0.1-0.2` | pushes harder toward cheap models once costs are observed |
| High uncertainty | `0.3-0.6` | `1.2-1.8` | `0.2-0.3` | explores more aggressively when model quality is unclear |

---

## Bandit Settings

All of these can be passed through `llm_dynamic_selection_kwargs`.

| Key | What it controls | When to raise it | When to lower it |
|-----|------------------|------------------|------------------|
| `cost_aware_coef` | Blend between reward-driven UCB and cost-driven cheapness. `0` is reward-only, `1` is cost-only. | When API spend matters more than absolute quality. | When stronger models are under-sampled. |
| `exploration_coef` | Size of the classic UCB exploration bonus. | When early estimates are noisy and you want broader coverage. | When the policy keeps revisiting clearly weaker models. |
| `epsilon` | Uniform exploration mass assigned to non-winning models. | When you want explicit random exploration in addition to UCB. | When selection should stay more deterministic. |
| `cost_exploration_coef` | Strength of the optimistic cheapness bonus on sparse cost data. | When you want cheaper models to get more early trials. | When cost estimates are already stable. |
| `cost_power` | Nonlinear amplification of relative cheapness after normalization. | When small cost differences should matter more. | When cost pressure is overpowering reward. |
| `cost_ref_percentile` | Cost reference percentile used in the ratio baseline. | When you want stronger pressure toward the cheaper half of the pool. | When the reference is too anchored to outliers or premium models. |

Example:

```python
evo_config = EvolutionConfig(
    llm_models=["gpt-5-mini", "gemini-3.1-pro-preview", "gpt-5.4"],
    llm_dynamic_selection="ucb",
    llm_dynamic_selection_kwargs={
        "cost_aware_coef": 0.55,
        "exploration_coef": 1.0,
        "epsilon": 0.2,
        "cost_exploration_coef": 0.1,
        "cost_power": 1.0,
        "cost_ref_percentile": 50,
    },
)
```
