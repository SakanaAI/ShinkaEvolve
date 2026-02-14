#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

strategy = "weighted"
if strategy == "uniform":
    # 1. Uniform from correct programs
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    # 2. Hill Climbing (Always from the Best)
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    # 3. Weighted Prioritization
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    # 5. Beam Search
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
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

search_task_sys_msg = """You are an expert mathematician specializing in circle packing problems and computational geometry. 
The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. Variable-sized circles with strategic placement (large in center, small at edges).
2. Hexagonal center arrangement combined with boundary adjustments.
3. Use `scipy.optimize.minimize` (SLSQP) for radius optimization.

*** CRITICAL OUTPUT RULES ***
1. **Output ONLY the python code.** No explanations, no markdown outside the code block.
2. The code must be enclosed in ```python and ``` tags.
3. The function signature MUST remain: `def construct_packing():`
4. The return statement MUST be exactly: `return centers, radii`
5. Ensure all imports (numpy, scipy) are explicitly included inside the code block.
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.1, 0.8, 0.1],
    num_generations=400,
    max_parallel_jobs=12,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "local-rnj-1:8b-http://localhost:11434/v1",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.2, 0.3],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=8192,
    ),
    meta_rec_interval=10,
    meta_llm_models=["local-rnj-1:8b-http://localhost:11434/v1"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    embedding_model="local-nomic-embed-text:latest-http://localhost:11434/v1",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["local-rnj-1:8b-http://localhost:11434/v1"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_cpack",
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
