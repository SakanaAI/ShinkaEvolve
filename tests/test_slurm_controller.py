import subprocess
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

from shinka.launch import JobScheduler, SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.launch import slurm


JOB_NAME = f"shinka-seed-{UUID(int=0).hex}"


@pytest.fixture(params=["slurm_conda", "slurm_docker"])
def scheduler(request, monkeypatch: pytest.MonkeyPatch):
    backend = request.param
    monkeypatch.setattr("shinka.launch.scheduler.uuid4", lambda: UUID(int=0))
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
    results = tmp_path / "results"
    results.mkdir()
    execution = tmp_path / f"results.{JOB_NAME}.execution-0"
    execution.mkdir()
    (execution / "correct.json").write_text('{"correct": true}')
    (execution / "metrics.json").write_text('{"combined_score": 42}')
    execution.with_name(f"{execution.name}.receipt").write_bytes(b"complete\n")
    return results


@pytest.mark.parametrize("visible_on_poll", [1, 2])
def test_slurm_receipt_can_appear_during_terminal_verification(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_on_poll: int,
) -> None:
    receipt = artifacts.with_name(f"{artifacts.name}.{JOB_NAME}.execution-0.receipt")
    receipt.unlink()
    polls = 0

    def run(command, **kwargs):
        nonlocal polls
        if command[0] == "sacct":
            polls += 1
            if polls == visible_on_poll:
                receipt.write_bytes(b"complete\n")
            output = f"123|{JOB_NAME}|COMPLETED|0:0"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("shinka.launch.slurm_evaluation.time.sleep", Mock())

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 42
    assert polls == visible_on_poll


@pytest.mark.parametrize("queue_reports_restart", [False, True])
def test_old_receipt_cannot_win_while_current_receipt_is_invisible(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    queue_reports_restart: bool,
) -> None:
    current = artifacts.with_name(f"{artifacts.name}.{JOB_NAME}.execution-1")
    current.mkdir()
    (current / "correct.json").write_text('{"correct": true}')
    (current / "metrics.json").write_text('{"combined_score": 23}')
    polls = 0

    def run(command, **kwargs):
        nonlocal polls
        if command[0] == "squeue":
            output = "COMPLETED|1" if queue_reports_restart else ""
        elif command[0] == "sacct":
            polls += 1
            if polls == 2:
                current.with_name(f"{current.name}.receipt").write_bytes(b"complete\n")
            output = f"123|{JOB_NAME}|COMPLETED|0:0"
        else:
            assert command[0] == "scontrol"
            output = (
                f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0 Restarts=1"
            )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("shinka.launch.slurm_evaluation.time.sleep", Mock())

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 23
    assert polls == 2


@pytest.mark.parametrize("controller_available", [False, True])
def test_requeued_accounting_success_checks_current_controller(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_available: bool,
) -> None:
    execution = artifacts.with_name(f"{artifacts.name}.{JOB_NAME}.execution-1")
    execution.mkdir()
    (execution / "correct.json").write_text('{"correct": true}')
    (execution / "metrics.json").write_text('{"combined_score": 23}')
    execution.with_name(f"{execution.name}.receipt").write_bytes(b"complete\n")
    controller = (
        f"JobId=123 JobName={JOB_NAME} JobState=FAILED ExitCode=1:0 Restarts=1"
        if controller_available
        else FileNotFoundError("controller unavailable")
    )
    mock_status(monkeypatch, f"123|{JOB_NAME}|COMPLETED|0:0", controller)

    if controller_available:
        with pytest.raises(RuntimeError, match="controller: state=FAILED"):
            scheduler.run("main.py", str(artifacts))
    else:
        results, _ = scheduler.run("main.py", str(artifacts))
        assert results["metrics"]["combined_score"] == 23


def mock_status(
    monkeypatch: pytest.MonkeyPatch,
    accounting: str | Exception,
    controller: str | Exception,
) -> Mock:
    def run(cmd, **kwargs):
        assert kwargs["timeout"] == 30
        assert not kwargs.get("shell")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="COMPLETED|0")
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
        f"123|{JOB_NAME}|RUNNING|0:0",
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
        f"JobId=123 JobName={JOB_NAME} UserId=user(1000) JobState=COMPLETED "
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
        f"JobId=123 JobName={JOB_NAME} JobState=FAILED ExitCode=1:0",
        f"JobId=123 JobName={JOB_NAME} JobState=TIMEOUT ExitCode=0:15",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=1:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:9",
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
        f"JobId=456 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobName=previous-allocation JobState=COMPLETED ExitCode=0:0",
        "JobId=123 JobState=COMPLETED ExitCode=0:0",
        f"JobId=123.batch JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED",
        "JobState=COMPLETED ExitCode=0:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=",
        f"JobId=123 JobName={JOB_NAME} JobState= ExitCode=0:0",
        f"JobId=123 JobName={JOB_NAME} JobState=RUNNING ExitCode=0:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0\nJobId=123 JobName={JOB_NAME} JobState=FAILED ExitCode=1:0",
        f"JobId=123 JobName={JOB_NAME} JobState=FAILED JobState=COMPLETED ExitCode=0:0",
        f"JobId=123 JobId=456 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=1:0 ExitCode=0:0",
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
        f"123|{JOB_NAME}|{state}|1:0",
        f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0",
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
                [],
                0,
                stdout=f"JobId=123 JobName={JOB_NAME} JobState=COMPLETED ExitCode=0:0",
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


def test_recycled_job_id_uses_current_controller_failure(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit = getattr(slurm, "submit_" + scheduler.job_type.removeprefix("slurm_"))
    mocked_submit = Mock(spec=submit, return_value="123")
    monkeypatch.setattr(
        f"shinka.launch.scheduler.submit_{scheduler.job_type}", mocked_submit
    )

    def run(cmd, **kwargs):
        if cmd[0] == "squeue":
            stdout = "FAILED"
        elif cmd[0] == "sacct":
            stdout = (
                "123|previous-allocation|COMPLETED|0:0"
                if "JobName" in cmd[-1]
                else "123|COMPLETED|0:0"
            )
        else:
            assert cmd[0] == "scontrol"
            current_name = mocked_submit.call_args.kwargs["job_name"]
            stdout = f"JobId=123 JobName={current_name} JobState=FAILED ExitCode=1:0"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout)

    monkeypatch.setattr(slurm.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="controller.*FAILED"):
        scheduler.run("main.py", str(artifacts))


def test_recycled_job_id_waits_for_current_accounting(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout="COMPLETED|0"),
            subprocess.CompletedProcess(
                [], 0, stdout="123|previous-allocation|COMPLETED|0:0"
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout="JobId=123 JobName=previous-allocation JobState=COMPLETED ExitCode=0:0",
            ),
            subprocess.CompletedProcess([], 0, stdout=f"123|{JOB_NAME}|COMPLETED|0:0"),
        ]
    )
    monkeypatch.setattr(slurm.subprocess, "run", command)
    sleep = Mock()
    monkeypatch.setattr(slurm.time, "sleep", sleep)

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 42
    assert [call.args[0][0] for call in command.call_args_list] == [
        "squeue",
        "sacct",
        "scontrol",
        "sacct",
    ]
    sleep.assert_called_once_with(10)


def test_recycled_queue_id_filters_by_submitted_name(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_status(monkeypatch, f"123|{JOB_NAME}|COMPLETED|0:0", "")

    scheduler.run("main.py", str(artifacts))

    assert command.call_args_list[0].args[0][:4] == [
        "squeue",
        "-j",
        "123",
        f"--name={JOB_NAME}",
    ]
