"""
Example script for running circle packing evolution with E2B cloud sandboxes.

This example demonstrates how to use E2B for scalable, distributed evaluation
without requiring local compute resources or HPC infrastructure.

Prerequisites:
1. Install E2B: pip install e2b>=2.2.0
2. Get E2B API key from https://e2b.dev
3. Set environment variable: export E2B_API_KEY=your_api_key_here
"""

import os
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import E2BJobConfig


def main():
    # Check for E2B API key
    if "E2B_API_KEY" not in os.environ:
        raise ValueError(
            "E2B_API_KEY environment variable not set. "
            "Get your API key at https://e2b.dev and set it with: "
            "export E2B_API_KEY=your_api_key_here"
        )

    # Configure E2B job execution
    job_config = E2BJobConfig(
        eval_program_path="examples/circle_packing/evaluate.py",
        template="base",  # E2B sandbox template
        timeout=600,  # 10 minutes timeout
        env_vars={},  # Add any required environment variables here
    )

    # Configure database with island model
    db_config = DatabaseConfig(
        num_islands=4,
        archive_size=50,
    )

    # Configure evolution
    evo_config = EvolutionConfig(
        task_sys_msg=(
            "You are an expert mathematician specializing in circle packing problems "
            "and computational geometry. The best known result for the sum of radii "
            "when packing 26 circles in a unit square is 2.635. "
            "Be creative and try to find a new solution."
        ),
        init_program_path="examples/circle_packing/initial.py",
        job_type="e2b",  # Use E2B execution backend
        language="python",
        num_generations=10,
        max_parallel_jobs=5,  # E2B can handle many parallel jobs
        llm_models=["azure-gpt-4.1-mini"],
        results_dir="results_circle_packing_e2b",
    )

    # Run evolution
    print("Starting evolution with E2B cloud sandboxes...")
    print(f"Results will be saved to: {evo_config.results_dir}")

    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )

    runner.run()

    print(f"\nEvolution completed! Results saved to: {evo_config.results_dir}")
    print("View results with: shinka_visualize --db_path " f"{evo_config.results_dir}/evolution.db")


if __name__ == "__main__":
    main()
