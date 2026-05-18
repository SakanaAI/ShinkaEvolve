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
You are an expert Wolfram Language programmer with a strong eye for
expression-level simplicity.

Task:
- Rewrite each of t1..t5 in the EVOLVE-BLOCK below into a smaller,
  equivalent Wolfram expression.
- Equivalence is checked by running each tN on a fixed set of numeric
  inputs and comparing the result to what the deoptimized seed returns.
  Any mismatch, timeout, or use of a disallowed head scores -1.

Rules:
- Modify only code inside EVOLVE-BLOCK markers.
- The disallowed heads are `Simplify` and `FullSimplify`. Their use
  scores -1 even when the candidate is otherwise correct.
- Score = seed_total_LeafCount / your_total_LeafCount across the whole
  EVOLVE-BLOCK (so helpers count toward your total).
- Each call to a tN runs under a tight per-call wall-time budget (~50ms).
  Closed-form rewrites are essentially free; recomputing the answer at
  call time via heavy machinery will time out.

Hints:
- Recognize algebraic identities: factor, expand-to-cubed-binomial,
  Pythagorean identity, sum-of-i*(i+1) closed form, etc.
- Inline helpers when their only call site is one expression away.
- Replace verbose `Module[..., For[...]]` with built-in aggregators
  (`Total`, `Mean`, `Sum`, `Fold`) when the loop is just an accumulator.
- Slot syntax (`#`, `&`), operator forms, and pure-function composition
  often beat named-variable Module bodies on LeafCount.
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
    llm_models=["claude-cli/sonnet"],
    llm_kwargs=dict(
        temperatures=[0.6, 0.9],
        reasoning_efforts=["disabled"],
        max_tokens=8192,
    ),
    embedding_model=None,
    init_program_path="initial.wl",
    results_dir="results_wolfram_simplify",
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
