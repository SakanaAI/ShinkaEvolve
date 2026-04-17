# Async Evolution Pipeline

Shinka runs evolution through `ShinkaEvolveRunner`.
Use proposal concurrency to control throughput and emulate prior sync behavior.

---

## Quick Start

```python
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.launch import LocalJobConfig
from shinka.database import DatabaseConfig


evo_config = EvolutionConfig(
    num_generations=50,
    llm_models=["gpt-5-mini"],
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(),
    max_evaluation_jobs=2,
    max_proposal_jobs=3,  # slight proposal oversubscription to keep eval workers busy
    max_db_workers=4,
)

runner.run()
```

In async contexts (for example notebooks/async apps), use:

```python
await runner.run_async()
```

---

## Concurrency Knobs

- `max_evaluation_jobs`: max concurrent evaluation jobs.
- `max_proposal_jobs`: max concurrent proposal generation jobs.
- `max_db_workers`: max async database worker threads.
- `enable_controlled_oversubscription`: adaptive controller for bounded proposal oversubscription.

`max_proposal_jobs=1` gives sequential proposal generation behavior.
All concurrency knobs live on `ShinkaEvolveRunner`.

Suitable concurrency depends on your machine. In practice, leave enough CPU capacity for the database workers, evaluation jobs, and proposal sampling jobs to run without starving each other.

When sampling/proposal generation is slower than evaluation, set
`max_proposal_jobs > max_evaluation_jobs` and enable controlled oversubscription.
This allows a small backlog of proposals to keep evaluation workers fed without
creating an unbounded queue.

---

## ShinkaEvolveRunner Parameters

```python
ShinkaEvolveRunner(
    evo_config=EvolutionConfig(...),
    job_config=JobConfig(...),
    db_config=DatabaseConfig(...),
    verbose=True,
    max_evaluation_jobs=2,
    max_proposal_jobs=3,
    max_db_workers=4,
)
```

---

## Recommended Settings

| Scale | max_evaluation_jobs | max_proposal_jobs | Notes |
|-------|-------------------|-------------------|-------|
| Sequential-like | 1-4 | 1 | sync-like proposal behavior |
| Small | 2-6 | eval + 1 | good default if eval waits on proposals |
| Medium | 5-20 | eval + 1 to eval + 2 | use adaptive oversubscription |
| Large | 20+ | eval + 2 to eval + 6 | keep bounded with caps |

---

## Controlled Oversubscription

Adaptive oversubscription uses observed proposal and evaluation timings to
compute a bounded proposal target.

Key settings on `EvolutionConfig`:

- `enable_controlled_oversubscription`
- `proposal_target_mode`
- `proposal_target_min_samples`
- `proposal_target_ratio_cap`
- `proposal_buffer_max`
- `proposal_target_hard_cap`
- `proposal_target_ewma_alpha`

### What Each Oversubscription Setting Does

Oversubscription never increases evaluation concurrency.
`max_evaluation_jobs` still caps concurrent evals.
These settings only control how many proposal/sampling jobs Shinka is willing
to keep in flight ahead of those eval workers.
| Key | What it controls | When to raise it | When to lower it |
|-----|------------------|------------------|------------------|
| `enable_controlled_oversubscription` | Master on/off switch. If `false`, proposal target stays at `max_evaluation_jobs`. Default is `false`. | Turn it on if proposals are slower than evals and workers go idle waiting for new candidates. | Leave it off for predictable sync-like behavior or easier debugging. |
| `proposal_target_mode` | How Shinka chooses the proposal target. `adaptive` uses observed timings. `fixed` uses `max_evaluation_jobs + proposal_buffer_max`. | Use `adaptive` for most runs. Use `fixed` if workload timing is stable and you want deterministic behavior. | Switch away from `fixed` if it overfills the queue; switch away from `adaptive` if you need simpler tuning. |
| `proposal_target_min_samples` | Warmup count before adaptive mode trusts observed timing ratios. Before this, Shinka only adds a small buffer. | Raise if early timings are noisy or unrepresentative. | Lower if you want the controller to react sooner. |
| `proposal_target_ratio_cap` | Upper bound on the observed `sampling_seconds / evaluation_seconds` ratio used by adaptive mode. Prevents extreme spikes from asking for too many proposals. | Raise if proposal generation is consistently much slower than eval and backlog is still too small. | Lower if one slow sample causes too much queued proposal work. |
| `proposal_buffer_max` | Max number of extra proposal jobs allowed above `max_evaluation_jobs`. Primary backlog-size knob. | Raise if eval workers go idle waiting for proposals. | Lower if memory/API pressure grows or proposal backlog gets too large. |
| `proposal_target_hard_cap` | Absolute cap on adaptive/fixed proposal target before applying `max_proposal_jobs`. Useful when `max_proposal_jobs` is high but you want a lower oversub ceiling. | Raise if the controller is hitting the cap too early. | Lower if you want a strict safety stop regardless of timing estimates. |
| `proposal_target_ewma_alpha` | Smoothing factor for timing EWMAs. Higher values react faster; lower values react more slowly but more stably. | Raise if workload phase changes quickly and the controller lags behind. | Lower if proposal target oscillates too much from noisy timings. |

### How the Limits Combine

Think of the final proposal target as:

```text
base target = max_evaluation_jobs
adaptive/fixed target = mode-specific estimate
final target = clamp(
    adaptive/fixed target,
    lower=max_evaluation_jobs,
    upper=min(
        max_evaluation_jobs + proposal_buffer_max,
        proposal_target_hard_cap or max_proposal_jobs,
        max_proposal_jobs,
    ),
)
```

Practical read:

- `max_evaluation_jobs`: eval capacity.
- `max_proposal_jobs`: hard ceiling for proposal workers.
- `proposal_buffer_max`: how far above eval capacity you can go.
- `proposal_target_hard_cap`: extra absolute stop, even if other limits are higher.
- `proposal_target_ratio_cap`: only affects adaptive mode's estimate before clamping.

### Tuning Heuristics

- Eval workers idle often: raise `proposal_buffer_max` first, then maybe `max_proposal_jobs`.
- Backlog too deep: lower `proposal_buffer_max` or `proposal_target_hard_cap`.
- Controller too jumpy: lower `proposal_target_ewma_alpha`.
- Controller too sluggish: raise `proposal_target_ewma_alpha`.
- Startup phase too conservative: lower `proposal_target_min_samples`.
- Startup phase too noisy: raise `proposal_target_min_samples`.

Example:

```python
evo_config = EvolutionConfig(
    num_generations=100,
    llm_models=["gpt-5.4-nano", "gpt-5.4-mini"],
    enable_controlled_oversubscription=True,
    proposal_target_mode="adaptive",
    proposal_target_min_samples=5,
    proposal_target_ratio_cap=2.0,
    proposal_buffer_max=2,
    proposal_target_hard_cap=7,
    proposal_target_ewma_alpha=0.3,
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(),
    max_evaluation_jobs=5,
    max_proposal_jobs=7,
    max_db_workers=4,
)
```

---

## Troubleshooting

- Too many requests: reduce `max_proposal_jobs`.
- Proposal backlog grows too much: lower `proposal_buffer_max` or `proposal_target_ratio_cap`.
- Evaluation workers idle: raise `max_proposal_jobs` modestly and keep controlled oversubscription enabled.
- Memory pressure: lower `max_proposal_jobs` and `max_evaluation_jobs`.
- DB contention: lower `max_db_workers`.
- File I/O errors: ensure `aiofiles` installed.
