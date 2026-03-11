#!/usr/bin/env python3
import argparse
import asyncio
import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """You are optimizing code converted from an existing codebase.
Preserve the task contract and keep changes focused on the intended EVOLVE-BLOCK regions.
Do not break evaluation outputs, result file schemas, or imports required by the task snapshot."""


async def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="05:00:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config["max_evaluation_jobs"],
        max_proposal_jobs=config["max_proposal_jobs"],
        max_db_workers=config["max_db_workers"],
        debug=False,
        verbose=True,
    )
    await runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config_path))
