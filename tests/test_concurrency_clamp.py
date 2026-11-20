"""The runner's worker clamp: each pool is limited by what its workers cost."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

import shinka.core.async_runner as async_runner
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import (
    JobConfig,
    LocalJobConfig,
    SlurmCondaJobConfig,
    SlurmDockerJobConfig,
)

RUNNER_LOGGER = "shinka.core.async_runner"
ASYNC_DOCS = Path(__file__).resolve().parents[1] / "docs" / "async_evolution.md"


@pytest.fixture
def set_memory_gb(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    def _set(memory_gb: float) -> None:
        monkeypatch.setattr(
            async_runner.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(total=memory_gb * 1024**3),
        )

    return _set


def _runner(job_config: JobConfig | None = None) -> ShinkaEvolveRunner:
    runner = object.__new__(ShinkaEvolveRunner)
    runner.job_config = job_config or LocalJobConfig()
    runner.verbose = False
    return runner


def _clamp_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == RUNNER_LOGGER and record.levelno >= logging.WARNING
    ]


@pytest.mark.parametrize(
    ("cpu_count", "memory_gb", "requested"),
    [
        (10, 64, (8, 8, 4)),  # the benchmarked request: was (7, 4, 3)
        (10, 64, (5, 7, 4)),  # docs/async_evolution.md example: was (5, 5, 4)
        (8, 16, (4, 6, 2)),  # Hydra async default: was (4, 4, 2)
        (4, 16, (4, 6, 2)),  # same on a 4-core machine: was (3, 1, 1)
        (2, 7, (2, 3, 4)),  # README quick start on a 2-core runner: was (1, 1, 1)
        (10, 64, (2, 8, 4)),  # proposals above half the cores: was (2, 5, 4)
    ],
)
def test_request_that_fits_the_machine_is_granted_unchanged(
    cpu_count: int,
    memory_gb: float,
    requested: tuple[int, int, int],
    set_memory_gb: Callable[[float], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_memory_gb(memory_gb)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(*requested, cpu_count)

    assert granted == requested
    assert _clamp_warnings(caplog) == []


def test_local_evaluations_are_capped_at_the_core_count_and_nothing_else_shrinks(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(64)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(16, 16, 4, 10)

    assert granted == (10, 16, 4)
    warnings = _clamp_warnings(caplog)
    assert len(warnings) == 1
    assert "max_evaluation_jobs" in warnings[0]
    assert "requested 16" in warnings[0]
    assert "granted 10" in warnings[0]
    assert "10 CPU cores" in warnings[0]


def test_local_evaluations_are_capped_by_memory(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(2)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(8, 8, 4, 10)

    assert granted == (4, 8, 4)
    warnings = _clamp_warnings(caplog)
    assert len(warnings) == 1
    assert "max_evaluation_jobs" in warnings[0]
    assert "requested 8" in warnings[0]
    assert "granted 4" in warnings[0]
    assert "2.0 GB installed" in warnings[0]


def test_core_limit_is_the_reason_when_cores_and_memory_allow_the_same(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(4)  # eight evaluations by memory, eight by cores

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(9, 9, 4, 8)

    assert granted == (8, 9, 4)
    assert "8 CPU cores" in _clamp_warnings(caplog)[0]


def test_unknown_memory_is_treated_as_eight_gb_and_reported_as_assumed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fail() -> None:
        raise OSError("no memory information")

    monkeypatch.setattr(async_runner.psutil, "virtual_memory", _fail)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(32, 32, 4, 64)

    assert granted == (16, 32, 4)
    warnings = _clamp_warnings(caplog)
    assert len(warnings) == 1
    assert "8.0 GB assumed" in warnings[0]
    assert "installed" not in warnings[0]


def test_verbose_run_logs_the_granted_settings(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(64)
    runner = _runner()
    runner.verbose = True

    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        granted = runner._validate_concurrency_settings(16, 12, 4, 10)

    assert granted == (10, 12, 4)
    messages = [record.getMessage() for record in caplog.records]
    assert any("Evaluation jobs: 10" in message for message in messages)
    assert any("Proposal jobs: 12" in message for message in messages)
    assert any("DB workers: 4" in message for message in messages)
    assert len(_clamp_warnings(caplog)) == 1


def test_db_workers_are_capped_at_eight(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(64)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(2, 2, 16, 32)

    assert granted == (2, 2, 8)
    warnings = _clamp_warnings(caplog)
    assert len(warnings) == 1
    assert "max_db_workers" in warnings[0]
    assert "requested 16" in warnings[0]
    assert "granted 8" in warnings[0]


def test_every_clamped_pool_gets_its_own_warning(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(64)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(12, 40, 9, 4)

    assert granted == (4, 40, 8)
    warnings = _clamp_warnings(caplog)
    assert len(warnings) == 2
    assert "max_evaluation_jobs" in warnings[0] and "requested 12" in warnings[0]
    assert "max_db_workers" in warnings[1] and "requested 9" in warnings[1]


@pytest.mark.parametrize(
    "job_config",
    [SlurmCondaJobConfig(), SlurmDockerJobConfig()],
    ids=["conda", "docker"],
)
def test_slurm_evaluations_are_not_limited_by_the_submitting_machine(
    job_config: JobConfig,
    set_memory_gb: Callable[[float], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A 4-core, 8 GB login node submitting 50 cluster jobs: was (3, 1, 1).
    set_memory_gb(8)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner(job_config)._validate_concurrency_settings(50, 50, 4, 4)

    assert granted == (50, 50, 4)
    assert _clamp_warnings(caplog) == []


@pytest.mark.parametrize(
    ("cpu_count", "memory_gb", "requested", "granted", "warnings"),
    [
        (
            10,
            64,
            (16, 16, 4),
            (10, 16, 4),
            [
                "Concurrency clamp: max_evaluation_jobs requested 16, granted 10 "
                "(one CPU-bound evaluation per core, 10 CPU cores)"
            ],
        ),
        (
            10,
            2,
            (8, 8, 4),
            (4, 8, 4),
            [
                "Concurrency clamp: max_evaluation_jobs requested 8, granted 4 "
                "(0.5 GB of RAM assumed per evaluation, 2.0 GB installed)"
            ],
        ),
        (
            32,
            64,
            (2, 2, 16),
            (2, 2, 8),
            [
                "Concurrency clamp: max_db_workers requested 16, granted 8 "
                "(at most 8 database workers are supported)"
            ],
        ),
    ],
    ids=["cores", "memory", "db-workers"],
)
def test_clamp_warning_reads_as_documented(
    cpu_count: int,
    memory_gb: float,
    requested: tuple[int, int, int],
    granted: tuple[int, int, int],
    warnings: list[str],
    set_memory_gb: Callable[[float], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_memory_gb(memory_gb)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        result = _runner()._validate_concurrency_settings(*requested, cpu_count)

    assert result == granted
    assert _clamp_warnings(caplog) == warnings


def test_async_docs_quote_the_clamp_warning_as_the_runner_logs_it(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(64)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        _runner()._validate_concurrency_settings(16, 16, 4, 10)

    [warning] = _clamp_warnings(caplog)
    assert warning in ASYNC_DOCS.read_text(encoding="utf-8").splitlines()


def test_unknown_memory_warning_says_the_size_is_assumed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fail() -> None:
        raise OSError("no memory information")

    monkeypatch.setattr(async_runner.psutil, "virtual_memory", _fail)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        _runner()._validate_concurrency_settings(32, 32, 4, 64)

    assert _clamp_warnings(caplog) == [
        "Concurrency clamp: max_evaluation_jobs requested 32, granted 16 "
        "(0.5 GB of RAM assumed per evaluation, "
        "8.0 GB assumed, memory size unknown)"
    ]


def test_a_machine_with_less_memory_than_one_evaluation_still_runs_one(
    set_memory_gb: Callable[[float], None], caplog: pytest.LogCaptureFixture
) -> None:
    set_memory_gb(0.3)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        granted = _runner()._validate_concurrency_settings(4, 4, 2, 4)

    assert granted == (1, 4, 2)
    assert _clamp_warnings(caplog) == [
        "Concurrency clamp: max_evaluation_jobs requested 4, granted 1 "
        "(0.5 GB of RAM assumed per evaluation, 0.3 GB installed)"
    ]


def _built_runner(
    results_dir: Path,
    job_config: JobConfig,
    job_type: str,
    requested: tuple[int, int, int],
) -> ShinkaEvolveRunner:
    return ShinkaEvolveRunner(
        EvolutionConfig(
            results_dir=str(results_dir),
            num_generations=1,
            job_type=job_type,
            embedding_model=None,
            llm_models=[],
            llm_dynamic_selection=None,
            init_program_path=None,
        ),
        job_config,
        DatabaseConfig(num_islands=1),
        verbose=False,
        banner_style="minimal",
        max_evaluation_jobs=requested[0],
        max_proposal_jobs=requested[1],
        max_db_workers=requested[2],
    )


def test_local_runner_is_built_with_the_granted_worker_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    set_memory_gb: Callable[[float], None],
) -> None:
    set_memory_gb(16)
    monkeypatch.setattr(async_runner.os, "cpu_count", lambda: 4)
    job_config = LocalJobConfig()

    runner = _built_runner(tmp_path, job_config, "local", (12, 40, 9))

    try:
        assert runner.max_evaluation_jobs == 4
        assert runner.max_proposal_jobs == 40
        assert runner.max_db_workers == 8
        assert runner.evaluation_slot_pool.capacity == 4
        assert runner.sampling_slot_pool.capacity == 40
        assert runner.postprocess_slot_pool.capacity == 8
        # The numeric-thread cap follows the granted evaluations, not the request.
        assert job_config.numeric_threads_per_job == 1
    finally:
        runner.scheduler.shutdown()


def test_slurm_runner_is_built_with_the_requested_evaluation_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    set_memory_gb: Callable[[float], None],
) -> None:
    set_memory_gb(8)
    monkeypatch.setattr(async_runner.os, "cpu_count", lambda: 4)
    job_config = SlurmCondaJobConfig()

    runner = _built_runner(tmp_path, job_config, "slurm_conda", (50, 50, 4))

    try:
        assert runner.max_evaluation_jobs == 50
        assert runner.max_proposal_jobs == 50
        assert runner.max_db_workers == 4
        assert runner.evaluation_slot_pool.capacity == 50
        assert job_config.numeric_threads_per_job is None
    finally:
        runner.scheduler.shutdown()
