#!/usr/bin/env python3
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    time="00:05:00",
)

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=20,
    elite_selection_ratio=0.3,
    num_archive_inspirations=2,
    num_top_k_inspirations=1,
)

task_sys_msg = """
You are an expert Wolfram Language performance engineer.

Task:
- Speed up the function `solve[]` in the Wolfram seed program.
- `solve[]` MUST return the same integer the deoptimized seed returns
  (any other value scores -1).

Rules:
- Modify only code inside EVOLVE-BLOCK markers.
- Do not touch the I/O harness below the EVOLVE-BLOCK.
- Score = baseline_time / your_time. Higher is better; no upper bound.

Hints:
- Vectorized array operations (Map, Total, MapThread, Apply) on packed arrays
  are dramatically faster than scalar For / While loops.
- Built-in functions (NumberTheory, LinearAlgebra, ...) are nearly always
  faster than hand-written equivalents.
- Compile[] with type hints can speed up tight numeric loops considerably.
- ParallelTable / ParallelSum exploit multiple cores.
- When the problem has mathematical structure, look for identities or closed
  forms that reduce asymptotic complexity (O(N^2) -> O(N log N) or better);
  that beats any micro-optimization.
"""


evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=20,
    max_patch_resamples=2,
    max_patch_attempts=3,
    job_type="local",
    language="wolfram",
    llm_models=["gemini-2.5-flash-lite"],
    llm_kwargs=dict(
        temperatures=[0.6, 0.9],
        reasoning_efforts=["disabled"],
        max_tokens=8192,
    ),
    embedding_model=None,
    init_program_path="initial.wl",
    results_dir="results_wolfram_gcd_sum",
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
