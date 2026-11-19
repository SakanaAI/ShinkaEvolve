#!/usr/bin/env python3
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    time="00:03:00",
)

db_config = DatabaseConfig(
    num_islands=1,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
)

task_sys_msg = """
You are optimizing a Fortran program for a numerical stencil task.

Task:
- Simulate one-dimensional heat diffusion with fixed boundary values.
- Output one floating-point checksum per input case.

Rules:
- Keep all immutable code outside EVOLVE-BLOCK markers unchanged.
- Only modify code inside EVOLVE-BLOCK regions.
- Preserve module/subroutine interfaces used by the immutable CLI.
- Keep numerical answers within the evaluator tolerance.

Hints:
- Reuse buffers efficiently.
- Avoid unnecessary allocations inside repeated time steps.
- Algebraic rearrangements are useful only if they preserve double precision results.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.7, 0.3],
    num_generations=24,
    max_patch_resamples=2,
    max_patch_attempts=3,
    job_type="local",
    language="fortran",
    llm_models=["gpt-5-mini"],
    llm_kwargs=dict(
        temperatures=[0.2, 0.6, 0.9],
        reasoning_efforts=["medium"],
        max_tokens=16384,
    ),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    init_program_path="initial.f90",
    results_dir="results_fortran_heat_diffusion_async_small",
    max_novelty_attempts=1,
)


SMALL_MAX_EVAL_JOBS = 1
SMALL_MAX_PROPOSAL_JOBS = 2
SMALL_MAX_DB_WORKERS = 1


def main():
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=SMALL_MAX_EVAL_JOBS,
        max_proposal_jobs=SMALL_MAX_PROPOSAL_JOBS,
        max_db_workers=SMALL_MAX_DB_WORKERS,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
