#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig, OmegaConf
from shinka.env import load_shinka_dotenv
from shinka.core import ShinkaEvolveRunner


def run_with_cfg(cfg: DictConfig) -> None:
    load_shinka_dotenv()

    print("Experiment configurations:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)
    max_evaluation_jobs = int(cfg.get("max_evaluation_jobs", 4))
    max_proposal_jobs = int(cfg.get("max_proposal_jobs", 6))
    max_db_workers = int(cfg.get("max_db_workers", 2))

    evo_runner = ShinkaEvolveRunner(
        evo_config=evo_cfg,
        job_config=job_cfg,
        db_config=db_cfg,
        verbose=cfg.verbose,
        max_evaluation_jobs=max_evaluation_jobs,
        max_proposal_jobs=max_proposal_jobs,
        max_db_workers=max_db_workers,
    )
    evo_runner.run()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_with_cfg(cfg)


if __name__ == "__main__":
    main()
