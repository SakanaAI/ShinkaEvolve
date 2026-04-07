from types import SimpleNamespace

from shinka.core.async_runner import ShinkaEvolveRunner


def _build_runner(**overrides):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.max_evaluation_jobs = overrides.get("max_evaluation_jobs", 5)
    runner.max_proposal_jobs = overrides.get("max_proposal_jobs", 7)
    runner.active_proposal_tasks = {}
    runner.running_jobs = []
    runner._sampling_seconds_ewma = overrides.get("sampling_ewma")
    runner._evaluation_seconds_ewma = overrides.get("evaluation_ewma")
    runner._proposal_timing_samples = overrides.get("timing_samples", 0)
    runner._last_proposal_target_log = None
    runner.evo_config = SimpleNamespace(
        enable_controlled_oversubscription=overrides.get(
            "enable_controlled_oversubscription", False
        ),
        proposal_target_mode=overrides.get("proposal_target_mode", "adaptive"),
        proposal_target_min_samples=overrides.get("proposal_target_min_samples", 5),
        proposal_target_ratio_cap=overrides.get("proposal_target_ratio_cap", 2.0),
        proposal_buffer_max=overrides.get("proposal_buffer_max", 2),
        proposal_target_hard_cap=overrides.get("proposal_target_hard_cap"),
        proposal_target_ewma_alpha=overrides.get("proposal_target_ewma_alpha", 0.3),
    )
    return runner


def test_compute_proposal_pipeline_target_defaults_to_small_buffer_before_samples():
    runner = _build_runner(
        enable_controlled_oversubscription=True,
        timing_samples=0,
        proposal_buffer_max=2,
    )

    target = runner._compute_proposal_pipeline_target()

    assert target == 6


def test_compute_proposal_pipeline_target_adapts_and_clamps():
    runner = _build_runner(
        enable_controlled_oversubscription=True,
        sampling_ewma=120.0,
        evaluation_ewma=60.0,
        timing_samples=8,
        proposal_buffer_max=2,
        max_evaluation_jobs=5,
        max_proposal_jobs=9,
    )

    target = runner._compute_proposal_pipeline_target()

    assert target == 7


def test_compute_proposal_pipeline_target_respects_disable_flag():
    runner = _build_runner(
        enable_controlled_oversubscription=False,
        sampling_ewma=200.0,
        evaluation_ewma=20.0,
        timing_samples=10,
    )

    target = runner._compute_proposal_pipeline_target()

    assert target == 5


def test_compute_proposal_pipeline_target_ignores_invalid_hard_cap_below_eval_capacity():
    runner = _build_runner(
        enable_controlled_oversubscription=True,
        sampling_ewma=120.0,
        evaluation_ewma=60.0,
        timing_samples=8,
        proposal_buffer_max=2,
        max_evaluation_jobs=10,
        max_proposal_jobs=14,
        proposal_target_hard_cap=7,
    )

    target = runner._compute_proposal_pipeline_target()

    assert target == 12


def test_record_oversubscription_timing_sample_uses_ewma():
    runner = _build_runner()

    runner._record_oversubscription_timing_sample(
        {"sampling_seconds": 100.0, "evaluation_seconds": 50.0}
    )
    runner._record_oversubscription_timing_sample(
        {"sampling_seconds": 200.0, "evaluation_seconds": 100.0}
    )

    assert runner._proposal_timing_samples == 2
    assert round(runner._sampling_seconds_ewma, 2) == 130.0
    assert round(runner._evaluation_seconds_ewma, 2) == 65.0
