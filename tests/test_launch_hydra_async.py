from __future__ import annotations

from omegaconf import OmegaConf

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
            "job_config": {"eval_program_path": "evaluate.py"},
            "db_config": {"num_islands": 2},
            "evo_config": {
                "max_proposal_jobs": 3,
                "max_db_workers": 5,
            },
        }
    )

    launch_hydra.run_with_cfg(cfg)

    assert _DummyRunner.run_calls == 1
    assert _DummyRunner.last_kwargs is not None
    assert _DummyRunner.last_kwargs["max_evaluation_jobs"] == 7
    assert _DummyRunner.last_kwargs["max_proposal_jobs"] == 3
    assert _DummyRunner.last_kwargs["max_db_workers"] == 5
