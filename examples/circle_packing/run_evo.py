#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
import yaml


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py")
    db_config = DatabaseConfig(**config["db_config"])

    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    args = parser.parse_args()
    results_data = main(args.config_path)
