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

search_task_sys_msg = """You are an expert mathematician and architect doing Toilet Layout Optimization against Constraints.

Goal: Study the design constraints for a minimum-sized toilet layout.

1. Element Definition (Keys & Index)

Key: Type (Element Category)
WC (Water Closet)
Urinal
Wash Basin (Sink)
Corridor (Passage)
Key: Color (Representation)
Blue (B): Outline of Fixtures
Red (R): Passage / Corridor Area
Index: Used to denote variations within the same Type.
Example 1: WC-1, WC-2, WC-3 (Different door positions)
Example 2: Basin-1 (No door), Basin-2, Basin-3, Basin-4 (Different door positions)

2. Absolute Constraints (Hard Constraints)

No B/B Overlap: Blue (B) elements must not overlap with other Blue (B) elements.
No B/R Overlap: Blue (B) elements must not overlap with Red (R) elements.
Type Adjacency: Elements of the same Type (WC, Urinal, or Wash Basin) must be placed adjacent to each other.
Accessibility Constraint: Do not create a Red (R) area (Corridor/Passage) that is completely enclosed by Blue (B) elements. ( $\rightarrow$ Ensure toilet fixtures are accessible from the Corridor.)
The Green Line must border the perimeter.

3. Optimization/Preference Constraints (Soft Constraints)

Minimize Total Perimeter: Minimize the sum of the perimeters of all WC, Urinal, and Wash Basin elements.
R/R Overlap Preference: Red (R) elements should preferably overlap with other Red (R) elements (i.e., consolidate the corridor space).
Corridor Count: The number of Corridors is arbitrary, but should be minimized (for area reduction).
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=400,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        # "gemini-2.5-pro",
        "gemini-2.5-flash",
        # "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        # "o4-mini",
        # "gpt-5",
        # "gpt-5-mini",
        # "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    # meta_llm_models=["gpt-5-nano"],
    meta_llm_models=["gemini-2.5-flash"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    # embedding_model="text-embedding-3-small",
    embedding_model="gemini-embedding-exp-03-07",
    code_embed_sim_threshold=0.995,
    # novelty_llm_models=["gpt-5-nano"],
    novelty_llm_models=["gemini-2.5-flash"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
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
