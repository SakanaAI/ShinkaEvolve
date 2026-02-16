#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")


parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,  # chance to migrate program to random island
    island_elitism=True,  # Island elite is protected from migration
    **parent_config,
)

search_task_sys_msg = """
You are a world-class optimization expert and algorithm engineer in game design and AI.
You have to optimize the AI used in the famous game 2048.

## Game variant
- The goal of this variant is to reach the 2,048 value with the *least amount of actions possible*.


## Problem Constraints
- The game only lasts for *2,000 steps (actions)*.
- Each `get_best_move` function call must complete *within 100ms*. Make sure that your solution is highly efficient and suitable for implementation in Python.


Your goal is to improve the performance of the program and maximize the `combined_score` by suggesting improvements.

Typical approaches involve utilizing shallow search and heuristic for evaluating how "good" the board is.

Try diverse approaches to solve the problem. Think outside the box.
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=100,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        # "gemini-2.5-pro",
        "gemini-2.5-flash",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        # "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        # reasoning_efforts=["auto", "low", "medium", "high"],
        reasoning_efforts=["auto"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_2048",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    results_data = main()
