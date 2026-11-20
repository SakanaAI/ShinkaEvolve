"""The runner's worker clamp: each pool is limited by what its workers cost."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Callable

import pytest

import shinka.core.async_runner as async_runner
from shinka.core import ShinkaEvolveRunner
from shinka.launch import (
    JobConfig,
    LocalJobConfig,
    SlurmCondaJobConfig,
    SlurmDockerJobConfig,
)

RUNNER_LOGGER = "shinka.core.async_runner"


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
    assert "2.0 GB" in warnings[0]


def test_unknown_memory_is_treated_as_eight_gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail() -> None:
        raise OSError("no memory information")

    monkeypatch.setattr(async_runner.psutil, "virtual_memory", _fail)

    granted = _runner()._validate_concurrency_settings(32, 32, 4, 64)

    assert granted == (16, 32, 4)


def test_db_workers_are_capped_at_the_database_layer_limit(
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
