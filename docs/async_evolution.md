# Async Evolution Pipeline

Shinka runs evolution through `ShinkaEvolveRunner`.
Use proposal concurrency to control throughput and emulate prior sync behavior.

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

## Recommended Settings

| Scale | max_evaluation_jobs | max_proposal_jobs | Notes |
|-------|-------------------|-------------------|-------|
| Sequential-like | 1-4 | 1 | sync-like proposal behavior |
| Small | 2-6 | eval + 1 | good default if eval waits on proposals |
| Medium | 5-20 | eval + 1 to eval + 2 | use adaptive oversubscription |
| Large | 20+ | eval + 2 to eval + 6 | keep bounded with caps |

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

## Troubleshooting

- Too many requests: reduce `max_proposal_jobs`.
- Proposal backlog grows too much: lower `proposal_buffer_max` or `proposal_target_ratio_cap`.
- Evaluation workers idle: raise `max_proposal_jobs` modestly and keep controlled oversubscription enabled.
- Memory pressure: lower `max_proposal_jobs` and `max_evaluation_jobs`.
- DB contention: lower `max_db_workers`.
- File I/O errors: ensure `aiofiles` installed.
