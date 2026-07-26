"""Regression coverage for optional meta / novelty / prompt LLM features.

Users may:
- disable all three auxiliary features
- enable any of them with Headless subscription agents (text/scratch mode)
- enable any of them with direct provider APIs
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import (
    AsyncRunningJob,
    PersistedProgramEvent,
    ShinkaEvolveRunner,
    _llm_client_kwargs_for_text_requests,
)
from shinka.core.config import EvolutionConfig
from shinka.core.sampler import PromptSampler
from shinka.database import DatabaseConfig, Program
from shinka.launch import LocalJobConfig
from shinka.llm.rate_limit import estimate_minimum_request_demand


def _make_seed_repo(tmp_path: Path) -> Path:
    seed = tmp_path / "seed_repo"
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    return seed


def _build_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evo_overrides: dict | None = None,
) -> ShinkaEvolveRunner:
    monkeypatch.setattr(
        "shinka.core.async_runner._validate_evo_config_model_env_access",
        lambda _config: None,
    )
    seed = _make_seed_repo(tmp_path)
    overrides = {
        "seed_repo_path": str(seed),
        "llm_models": ["headless/cursor@composer-2.5"],
        "llm_dynamic_selection": None,
        "meta_rec_interval": None,
        "meta_llm_models": None,
        "novelty_llm_models": None,
        "evolve_prompts": False,
        "prompt_llm_models": None,
        "embedding_model": None,
        "num_generations": 2,
        "results_dir": str(tmp_path / "results"),
    }
    if evo_overrides:
        overrides.update(evo_overrides)
    return ShinkaEvolveRunner(
        evo_config=EvolutionConfig(**overrides),
        job_config=LocalJobConfig(),
        db_config=DatabaseConfig(),
        verbose=False,
    )


def test_runner_constructs_with_meta_novelty_and_prompt_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _build_runner(tmp_path, monkeypatch)

    assert runner.meta_summarizer is None
    assert runner.novelty_judge is None
    assert runner.prompt_llm is None
    assert runner.prompt_evolver is None
    assert runner.prompt_db is None
    assert runner.embedding_client is None
    assert runner.minimum_request_demand == {}

    prompt_text, prompt_id = runner._get_current_system_prompt()
    assert prompt_id is None
    assert prompt_text == runner.evo_config.task_sys_msg


def test_side_effects_and_prompt_sampling_work_when_aux_features_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _build_runner(tmp_path, monkeypatch)

    class _DB:
        def __init__(self):
            self.maintenance_calls = 0

        async def run_program_maintenance_async(self, program, verbose=False):
            self.maintenance_calls += 1

        async def get_async(self, program_id):
            return None

    runner.async_db = _DB()
    runner.llm_selection = None

    async def _noop_async(*_args, **_kwargs):
        return None

    runner._update_best_solution_async = _noop_async
    runner._persist_program_metadata_async = _noop_async
    runner._log_program_to_wandb = lambda *_args, **_kwargs: None
    runner.wandb_logger = SimpleNamespace(log_program=lambda *_a, **_k: None)

    program = Program(
        id="prog-1",
        code="print(1)\n",
        language="python",
        generation=1,
        correct=True,
        combined_score=1.0,
        metadata={"model_name": "headless/cursor@composer-2.5"},
    )
    job = AsyncRunningJob(
        job_id="job-1",
        generation=1,
        exec_fname="unused.py",
        results_dir=str(tmp_path / "results" / "gen_1"),
        start_time=0.0,
        proposal_started_at=0.0,
        evaluation_submitted_at=0.0,
        evaluation_started_at=0.0,
        code_diff="",
        parent_id=None,
        archive_insp_ids=[],
        top_k_insp_ids=[],
        code_embedding=[],
        meta_patch_data={},
    )
    event = PersistedProgramEvent(
        job=job,
        program=program,
        evaluation_finished_at=1.0,
        postprocess_started_at=1.0,
        postprocess_finished_at=1.5,
    )

    asyncio.run(runner._apply_persisted_program_side_effects(event))
    assert runner.async_db.maintenance_calls == 1

    sampler = PromptSampler(
        task_sys_msg="Improve the program.",
        language="python",
        patch_types=["diff"],
        patch_type_probs=[1.0],
    )
    sys_msg, user_msg, patch_type = sampler.sample(
        parent=program,
        archive_inspirations=[],
        top_k_inspirations=[],
        meta_recommendations=None,
    )
    assert "Improve the program." in sys_msg
    assert user_msg
    assert patch_type == "diff"


def test_estimate_demand_skips_disabled_meta():
    evo = SimpleNamespace(
        num_generations=40,
        meta_rec_interval=None,
        meta_llm_models=["gemini-3.6-flash"],
        embedding_model=None,
    )
    assert estimate_minimum_request_demand(evo) == {}

    evo.meta_rec_interval = 10
    evo.meta_llm_models = None
    assert estimate_minimum_request_demand(evo) == {}


def test_aux_llm_kwargs_support_api_or_headless(tmp_path: Path):
    evo_config = EvolutionConfig(
        llm_models=["headless/cursor@composer-2.5"],
        seed_repo_path=str(tmp_path / "seed"),
        headless_cleanup_grace_seconds=15.0,
    )
    results_dir = tmp_path / "results"

    api_kwargs = _llm_client_kwargs_for_text_requests(
        {"temperatures": [0.0], "max_tokens": [1024]},
        results_dir,
        request_class="meta",
        model_names=["gemini-3.6-flash"],
        evo_config=evo_config,
    )
    assert api_kwargs["temperatures"] == [0.0]
    assert api_kwargs["max_tokens"] == [1024]
    assert api_kwargs["headless_work_dir"] == str(results_dir)
    assert "headless_response_mode" not in api_kwargs

    headless_kwargs = _llm_client_kwargs_for_text_requests(
        {"temperatures": [0.0]},
        results_dir,
        request_class="novelty",
        model_names=["headless/codex@gpt-5.6-luna?effort=medium"],
        evo_config=evo_config,
    )
    assert headless_kwargs["headless_response_mode"] == "text"
    assert headless_kwargs["headless_output_mode"] == "usage"
    assert headless_kwargs["headless_cleanup_grace_seconds"] == 15.0
    assert Path(headless_kwargs["headless_work_dir"]).name == "novelty"


def test_runner_enables_api_or_headless_aux_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    api_runner = _build_runner(
        tmp_path / "api",
        monkeypatch,
        evo_overrides={
            "meta_rec_interval": 10,
            "meta_llm_models": ["gemini-3.6-flash"],
            "meta_llm_kwargs": {"temperatures": [0.0]},
            "novelty_llm_models": ["gemini-3.6-flash"],
            "evolve_prompts": True,
            "prompt_evolution_interval": 5,
            "prompt_llm_models": ["gemini-3.6-flash"],
        },
    )
    assert api_runner.meta_summarizer is not None
    assert api_runner.novelty_judge is not None
    assert api_runner.prompt_llm is not None
    assert api_runner.meta_summarizer.async_llm_client.headless_response_mode is None
    assert api_runner.novelty_judge.async_llm_client.headless_response_mode is None
    assert api_runner.prompt_llm.headless_response_mode is None

    headless_runner = _build_runner(
        tmp_path / "headless",
        monkeypatch,
        evo_overrides={
            "meta_rec_interval": 10,
            "meta_llm_models": ["headless/cursor@composer-2.5"],
            "novelty_llm_models": ["headless/cursor@composer-2.5"],
            "evolve_prompts": True,
            "prompt_evolution_interval": 5,
            "prompt_llm_models": ["headless/cursor@composer-2.5"],
        },
    )
    assert headless_runner.meta_summarizer is not None
    assert headless_runner.novelty_judge is not None
    assert headless_runner.prompt_llm is not None
    assert (
        headless_runner.meta_summarizer.async_llm_client.headless_response_mode
        == "text"
    )
    assert (
        headless_runner.novelty_judge.async_llm_client.headless_response_mode == "text"
    )
    assert headless_runner.prompt_llm.headless_response_mode == "text"
