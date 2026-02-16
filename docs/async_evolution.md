# Async Evolution Pipeline

Asynchronous evolution pipeline providing **5-10x speedup** through concurrent proposal generation.

## Quick Start

```python
import asyncio
from shinka import AsyncEvolutionRunner, EvolutionConfig
from shinka.launch import LocalJobConfig
from shinka.database import DatabaseConfig

async def main():
    evo_config = EvolutionConfig(
        num_generations=50,
        max_parallel_jobs=20,
        llm_models=["gpt-4o-mini"],
    )
    
    runner = AsyncEvolutionRunner(
        evo_config=evo_config,
        job_config=LocalJobConfig(),
        db_config=DatabaseConfig(db_path="evolution.sqlite"),
        max_proposal_jobs=10,  # Generate 10 proposals concurrently
        max_evaluation_jobs=10,  # Proposals to evaluate in parallel
    )
    
    await runner.run()

asyncio.run(main())
```

## Key Benefits

- **5-10x faster**: Concurrent proposal generation vs sequential
- **Better resource utilization**: Keep evaluation queues full
- **Scalable**: Handle hundreds of parallel evaluation jobs
- **Drop-in replacement**: Same configs as sync version

## Architecture

```
Main Loop ──┬── Job Monitor (check completions)
            └── Proposal Generator (concurrent tasks)
                    ├── Sample DB
                    ├── Query LLM  
                    ├── Apply Patch
                    ├── Check Novelty
                    └── Submit to Queue
```

## Configuration

### AsyncEvolutionRunner Parameters

```python
AsyncEvolutionRunner(
    evo_config=EvolutionConfig(...),      # Same as sync version
    job_config=JobConfig(...),            # Same as sync version  
    db_config=DatabaseConfig(...),        # Same as sync version
    verbose=True,                         # Enable logging
    max_evaluation_jobs=None,             # Defaults to evo_config.max_parallel_jobs
    max_proposal_jobs=10,                 # Max concurrent proposal generation
)
```

### Recommended Settings

| Scale | max_parallel_jobs | max_proposal_jobs |
|-------|------------------|-------------------|
| Small | ≤ 10             | 5                 |
| Medium| 10-50            | 10                |
| Large | 50+              | 20                |

## Migration from Sync

1. Replace `EvolutionRunner` with `AsyncEvolutionRunner`
2. Add `await` to `runner.run()`
3. Wrap in `asyncio.run(main())`
4. All other configs unchanged

## Key Components

- **AsyncEvolutionRunner**: Main orchestrator with concurrent task management
- **AsyncProgramDatabase**: Thread-safe database wrapper
- **AsyncNoveltyJudge**: Embedding similarity + single LLM call when needed
- **AsyncLLMClient**: Concurrent API calls (existing component)


## Troubleshooting

- **"Too many requests"**: Reduce `max_proposal_jobs`
- **Memory issues**: Lower `max_proposal_jobs` 
- **Rate limits**: Configure LLM client delays
- **File errors**: Install `aiofiles`

The async pipeline maintains full compatibility with existing `EvolutionConfig`, `JobConfig`, and `DatabaseConfig` while providing significant performance improvements through concurrent processing.