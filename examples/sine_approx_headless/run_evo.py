#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


TASK_DIR = Path(__file__).resolve().parent

job_config = LocalJobConfig(
    eval_program_path=str(TASK_DIR / "evaluate.py"),
    time="00:03:00",
)

db_config = DatabaseConfig(
    num_islands=2,
    archive_size=8,
    num_archive_inspirations=1,
    num_top_k_inspirations=1,
)

evo_config = EvolutionConfig(
    patch_types=["full", "diff"],
    patch_type_probs=[0.5, 0.5],
    num_generations=10,
    max_patch_resamples=1,
    max_patch_attempts=1,
    job_type="local",
    language="python",
    llm_models=[
        "headless/codex@gpt-5.5?effort=high",
        "headless/claude",
    ],
    llm_dynamic_selection="fixed",
    llm_kwargs={
        "temperatures": [0.0],
        "max_tokens": 4096,
        "reasoning_efforts": ["high"],
    },
    embedding_model=None,
    init_program_path=str(TASK_DIR / "initial.py"),
    results_dir="results/sine_approx_headless_run_evo",
    max_novelty_attempts=1,
)


def main() -> None:
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=2,
        max_proposal_jobs=2,
        max_db_workers=2,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
