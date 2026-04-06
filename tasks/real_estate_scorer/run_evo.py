#!/usr/bin/env python3
"""Run ShinkaEvolve on the real estate scorer task."""

import argparse

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

search_task_sys_msg = """You are an expert data scientist specializing in real estate valuation and investment analysis for the Buenos Aires apartment market.

Your task is to evolve a Python scoring function `score_listing(listing)` that takes a dict with these keys:
- price_usd: apartment price in USD
- size_m2: apartment size in square meters
- neighborhood: neighborhood name (string)
- location_score: desirability score 1-5 (Palermo/Recoleta=5, Belgrano=4, Almagro/Villa Crespo=3, Boedo=2, Mataderos=1)
- amenity_count: number of amenities (0-10)
- floor: floor number (0-24)
- age_years: building age in years (0-79)

The function must return a float investment score. The fitness metric is Spearman rank correlation between your scores and the ground-truth price_per_m2 ranking on a held-out test set.

Key directions to explore:
1. Price per square meter is the core signal, but location and amenities add value
2. Non-linear feature interactions (e.g., location * price_per_m2)
3. Polynomial or logarithmic transforms of features
4. Feature engineering combining multiple inputs
5. Penalize older buildings or reward higher floors
6. Consider neighborhood-specific pricing models

Be creative. The baseline linear model achieves ~0.83 Spearman correlation."""


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="00:02:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs", 2),
        max_proposal_jobs=config.get("max_proposal_jobs", 2),
        max_db_workers=config.get("max_db_workers", 2),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka.yaml")
    args = parser.parse_args()
    main(args.config_path)
