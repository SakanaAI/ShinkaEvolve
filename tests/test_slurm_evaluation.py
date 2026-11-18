import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from shinka.launch import JobScheduler, SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.launch import slurm
from shinka.launch.slurm_evaluation import monitor_evaluation


@pytest.fixture(params=["slurm_conda", "slurm_docker"])
def scheduler(request, monkeypatch: pytest.MonkeyPatch):
    backend = request.param
    config = (
        SlurmCondaJobConfig() if backend == "slurm_conda" else SlurmDockerJobConfig()
    )
    monkeypatch.setattr(
        f"shinka.launch.scheduler.submit_{backend}", Mock(return_value="123")
    )
    instance = JobScheduler(backend, config)
    yield instance
    instance.shutdown()


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    (tmp_path / "correct.json").write_text('{"correct": true}')
    (tmp_path / "metrics.json").write_text('{"combined_score": 42}')
    return tmp_path


def mock_commands(monkeypatch: pytest.MonkeyPatch, accounting: list[str]) -> Mock:
    responses = iter(accounting)

    def run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        assert cmd[0] == "sacct"
        assert "--allocations" in cmd
        assert "--parsable2" in cmd
        assert cmd[cmd.index("--jobs") + 1] == "123"
        return subprocess.CompletedProcess(cmd, 0, stdout=next(responses))

    command = Mock(side_effect=run)
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", Mock())
    return command


@pytest.mark.parametrize(
    "record",
    ["123|FAILED|1:0", "123|TIMEOUT|0:15", "123|COMPLETED|1:0", "123|COMPLETED|0:9"],
)
def test_sync_slurm_rejects_partial_artifacts(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
) -> None:
    mock_commands(monkeypatch, [record])

    with pytest.raises(RuntimeError, match="123.*(FAILED|TIMEOUT|COMPLETED)"):
        scheduler.run("main.py", str(artifacts))


def test_sync_slurm_accepts_successful_allocation(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_commands(monkeypatch, ["123|COMPLETED|0:0\n"])

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["correct"]["correct"] is True
    assert results["metrics"]["combined_score"] == 42
    assert [call.args[0][0] for call in command.call_args_list] == ["squeue", "sacct"]


def test_sync_slurm_waits_for_accounting_visibility(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_commands(monkeypatch, ["", "123|RUNNING|0:0", "123|COMPLETED|0:0"])

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 42
    assert command.call_count == 4


@pytest.mark.parametrize(
    "record",
    [
        "",
        "123.batch|COMPLETED|0:0",
        "456|COMPLETED|0:0",
        "123|COMPLETED",
        "123|COMPLETED|0:0\n123|FAILED|1:0",
        "123||0:0",
        "123|RUNNING|0:0",
    ],
)
def test_sync_slurm_requires_matching_accounting_record(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
) -> None:
    command = mock_commands(monkeypatch, [record] * 6)

    with pytest.raises(RuntimeError, match="accounting.*123"):
        scheduler.run("main.py", str(artifacts))

    assert command.call_count == 7


def test_async_slurm_retrieval_keeps_existing_contract(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_commands(monkeypatch, [])

    results = scheduler.get_job_results("123", str(artifacts))

    assert results is not None and results["metrics"]["combined_score"] == 42
    assert command.call_count == 1


@pytest.mark.parametrize("exit_code", [0, 1, -9])
def test_local_fallback_uses_actual_exit_status(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    process = Mock()
    process.poll.side_effect = [None, exit_code]
    job: dict[str, Mock | None] = {"popen": None}
    monkeypatch.setitem(slurm.LOCAL_JOBS, "local-test", job)
    command = Mock(side_effect=AssertionError("Local jobs must not query Slurm"))
    monkeypatch.setattr(slurm.subprocess, "run", command)
    sleep = Mock(side_effect=lambda _: job.update(popen=process))
    monkeypatch.setattr(slurm.time, "sleep", sleep)

    if exit_code == 0:
        assert (
            monitor_evaluation("local-test", str(artifacts))["metrics"][
                "combined_score"
            ]
            == 42
        )
    else:
        with pytest.raises(RuntimeError, match=f"status {exit_code}"):
            monitor_evaluation("local-test", str(artifacts))
    assert sleep.call_count == 2
    command.assert_not_called()


def test_unknown_local_fallback_fails_closed(artifacts: Path) -> None:
    with pytest.raises(RuntimeError, match="Unknown local fallback"):
        monitor_evaluation("local-unknown", str(artifacts))


def test_unavailable_accounting_has_finite_retries(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=""),
            *[subprocess.CalledProcessError(1, "sacct")] * 6,
        ]
    )
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="accounting.*123"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 7
