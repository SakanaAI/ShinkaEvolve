#!/usr/bin/env python3
import asyncio

from shinka.core import AsyncEvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    time="00:03:00",
)

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=1,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
)

task_sys_msg = """
You are optimizing a Julia program for a math task: prime counting.

Task:
- Input is a list of integers n.
- Output must be the exact number of primes <= n for each query.

Rules:
- Keep all immutable code outside EVOLVE-BLOCK markers unchanged.
- Only modify code inside EVOLVE-BLOCK regions.
- Preserve function signatures.
- Focus on algorithmic speedups while keeping outputs exact.

Hints:
- Precomputation (e.g., sieve-style methods) can help for repeated queries.
- Avoid unnecessary allocations and redundant primality checks.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.7, 0.3],
    num_generations=24,
    max_parallel_jobs=1,
    max_patch_resamples=2,
    max_patch_attempts=3,
    job_type="local",
    language="julia",
    llm_models=["gpt-5-mini"],
    llm_kwargs=dict(
        temperatures=[0.2, 0.6, 0.9],
        reasoning_efforts=["medium"],
        max_tokens=16384,
    ),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    init_program_path="initial.jl",
    results_dir="results_julia_prime_counting_async_small",
    max_novelty_attempts=1,
)


SMALL_MAX_EVAL_JOBS = 1
SMALL_MAX_PROPOSAL_JOBS = 2
SMALL_MAX_DB_WORKERS = 1


async def main():
    runner = AsyncEvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=SMALL_MAX_EVAL_JOBS,
        max_proposal_jobs=SMALL_MAX_PROPOSAL_JOBS,
        max_db_workers=SMALL_MAX_DB_WORKERS,
        verbose=True,
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
