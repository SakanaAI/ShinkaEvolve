import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from shinka.launch import JobScheduler, SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.launch import slurm


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


def mock_status(
    monkeypatch: pytest.MonkeyPatch,
    accounting: str | Exception,
    controller: str | Exception,
) -> Mock:
    def run(cmd, **kwargs):
        assert kwargs["timeout"] == 30
        assert not kwargs.get("shell")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="COMPLETED")
        if cmd[0] == "scontrol":
            assert cmd == ["scontrol", "--oneliner", "show", "job", "123"]
            output = controller
        else:
            assert cmd[0] == "sacct"
            output = accounting
        if isinstance(output, Exception):
            raise output
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    command = Mock(side_effect=run)
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", Mock())
    return command


@pytest.mark.parametrize(
    "accounting",
    [
        "",
        "123|RUNNING|0:0",
        FileNotFoundError("sacct unavailable"),
        subprocess.TimeoutExpired("sacct", 30),
        subprocess.CalledProcessError(1, "sacct", stderr="Accounting disabled"),
    ],
)
def test_nonaccounting_cluster_accepts_verified_controller_success(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    accounting: str | Exception,
) -> None:
    command = mock_status(
        monkeypatch,
        accounting,
        "JobId=123 JobName=seed UserId=user(1000) JobState=COMPLETED "
        "Reason=None ExitCode=0:0 WorkDir=/tmp/work\n",
    )

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["correct"]["correct"] is True
    assert results["metrics"]["combined_score"] == 42
    assert [call.args[0][0] for call in command.call_args_list] == [
        "squeue",
        "sacct",
        "scontrol",
    ]


@pytest.mark.parametrize(
    "record",
    [
        "JobId=123 JobState=FAILED ExitCode=1:0",
        "JobId=123 JobState=TIMEOUT ExitCode=0:15",
        "JobId=123 JobState=COMPLETED ExitCode=1:0",
        "JobId=123 JobState=COMPLETED ExitCode=0:9",
    ],
)
def test_controller_failure_rejects_partial_artifacts(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
) -> None:
    command = mock_status(monkeypatch, "", record)

    with pytest.raises(RuntimeError, match="123.*(FAILED|TIMEOUT|COMPLETED)"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 3


@pytest.mark.parametrize(
    "record",
    [
        "",
        "JobId=456 JobState=COMPLETED ExitCode=0:0",
        "JobId=123.batch JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobState=COMPLETED",
        "JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobState=COMPLETED ExitCode=",
        "JobId=123 JobState= ExitCode=0:0",
        "JobId=123 JobState=RUNNING ExitCode=0:0",
        "JobId=123 JobState=COMPLETED ExitCode=0:0\nJobId=123 JobState=FAILED ExitCode=1:0",
        "JobId=123 JobState=FAILED JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobId=456 JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobState=COMPLETED ExitCode=1:0 ExitCode=0:0",
    ],
)
def test_unverified_controller_status_fails_closed(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
) -> None:
    command = mock_status(monkeypatch, "", record)

    with pytest.raises(RuntimeError, match="123"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 13


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("scontrol", 30),
        subprocess.CalledProcessError(1, "scontrol", stderr="Invalid job id specified"),
    ],
)
def test_controller_command_failures_are_bounded(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    command = mock_status(monkeypatch, "", error)

    with pytest.raises(RuntimeError, match="123.*scontrol"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 13


@pytest.mark.parametrize("state", ["FAILED", "TIMEOUT"])
def test_controller_cannot_override_confirmed_accounting_failure(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    command = mock_status(
        monkeypatch,
        f"123|{state}|1:0",
        "JobId=123 JobState=COMPLETED ExitCode=0:0",
    )

    with pytest.raises(RuntimeError, match=state):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 2


def test_controller_can_recover_within_terminal_status_grace(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout="COMPLETED"),
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.TimeoutExpired("scontrol", 30),
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess(
                [], 0, stdout="JobId=123 JobState=COMPLETED ExitCode=0:0"
            ),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", sleep)

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 42
    sleep.assert_called_once_with(10)
    assert command.call_count == 5
