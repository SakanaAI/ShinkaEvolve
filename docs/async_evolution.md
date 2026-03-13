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
    max_proposal_jobs=1,  # sync-like proposal behavior
    llm_models=["gpt-5-mini"],
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(),
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

`max_proposal_jobs=1` gives sequential proposal generation behavior.

## ShinkaEvolveRunner Parameters

```python
ShinkaEvolveRunner(
    evo_config=EvolutionConfig(...),
    job_config=JobConfig(...),
    db_config=DatabaseConfig(...),
    verbose=True,
    max_evaluation_jobs=2,
    max_proposal_jobs=None,    # defaults to evo_config.max_proposal_jobs
    max_db_workers=None,       # defaults to evo_config.max_db_workers
)
```

## Recommended Settings

| Scale | max_evaluation_jobs | max_proposal_jobs |
|-------|-------------------|-------------------|
| Sequential-like | 1-4 | 1 |
| Small | <= 10 | 2-5 |
| Medium | 10-50 | 5-10 |
| Large | 50+ | 10-20 |

## Troubleshooting

- Too many requests: reduce `max_proposal_jobs`.
- Memory pressure: lower `max_proposal_jobs` and `max_evaluation_jobs`.
- DB contention: lower `max_db_workers`.
- File I/O errors: ensure `aiofiles` installed.
