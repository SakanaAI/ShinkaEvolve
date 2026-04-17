from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from shinka.configs import config_root
import shinka.launch_hydra as launch_hydra


class _DummyRunner:
    last_kwargs = None
    run_calls = 0

    def __init__(self, **kwargs):
        _DummyRunner.last_kwargs = kwargs

    def run(self):
        _DummyRunner.run_calls += 1


def test_launch_hydra_uses_async_runner(monkeypatch):
    monkeypatch.setattr(launch_hydra, "ShinkaEvolveRunner", _DummyRunner)
    monkeypatch.setattr(launch_hydra.hydra.utils, "instantiate", lambda cfg: cfg)

    cfg = OmegaConf.create(
        {
            "verbose": False,
            "max_evaluation_jobs": 7,
            "max_proposal_jobs": 3,
            "max_db_workers": 5,
            "job_config": {"eval_program_path": "evaluate.py"},
            "db_config": {"num_islands": 2},
            "evo_config": {},
        }
    )

    launch_hydra.run_with_cfg(cfg)

    assert _DummyRunner.run_calls == 1
    assert _DummyRunner.last_kwargs is not None
    assert _DummyRunner.last_kwargs["max_evaluation_jobs"] == 7
    assert _DummyRunner.last_kwargs["max_proposal_jobs"] == 3
    assert _DummyRunner.last_kwargs["max_db_workers"] == 5
    assert _DummyRunner.last_kwargs.get("banner_style", "full") == "full"


def test_default_launch_config_uses_neutral_shared_defaults():
    with config_root() as cfgs_root:
        with initialize_config_dir(version_base=None, config_dir=str(cfgs_root)):
            cfg = compose(config_name="config")

    assert cfg.variant_suffix == "_default"
    assert cfg.exp_name == "shinka_circle_packing"
    assert cfg.max_evaluation_jobs == 4
    assert cfg.max_proposal_jobs == 6
    assert cfg.max_db_workers == 2
    assert cfg.evo_config.num_generations == 50
    assert cfg.evo_config.max_patch_attempts == 1
    assert cfg.evo_config.llm_models == [
        "gpt-5-mini",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
    ]
    assert cfg.evo_config.llm_dynamic_selection == "ucb"
    assert cfg.evo_config.llm_dynamic_selection_kwargs.cost_aware_coef == 0.5
    assert cfg.evo_config.meta_rec_interval == 10
    assert cfg.evo_config.code_embed_sim_threshold == 0.99
    assert cfg.evo_config.enable_controlled_oversubscription is False
    assert cfg.db_config.num_islands == 2
    assert cfg.db_config.archive_size == 40
    assert cfg.db_config.num_archive_inspirations == 1
    assert cfg.db_config.num_top_k_inspirations == 1
    assert cfg.db_config.migration_rate == 0.0
    assert cfg.db_config.parent_selection_strategy == "weighted"
