import subprocess
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

from shinka.launch import JobScheduler, SlurmCondaJobConfig, SlurmDockerJobConfig
from shinka.launch import slurm
from shinka.launch.slurm_evaluation import monitor_evaluation


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
    (tmp_path / "correct.json").write_text('{"correct": true}')
    (tmp_path / "metrics.json").write_text('{"combined_score": 42}')
    return tmp_path


def mock_commands(monkeypatch: pytest.MonkeyPatch, accounting: list[str]) -> Mock:
    responses = iter(accounting)

    def run(cmd, **kwargs):
        if cmd[0] in {"squeue", "scontrol"}:
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
    [
        f"123|{JOB_NAME}|FAILED|1:0",
        f"123|{JOB_NAME}|TIMEOUT|0:15",
        f"123|{JOB_NAME}|COMPLETED|1:0",
        f"123|{JOB_NAME}|COMPLETED|0:9",
    ],
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
    command = mock_commands(monkeypatch, [f"123|{JOB_NAME}|COMPLETED|0:0\n"])

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["correct"]["correct"] is True
    assert results["metrics"]["combined_score"] == 42
    assert [call.args[0][0] for call in command.call_args_list] == ["squeue", "sacct"]


def test_sync_slurm_waits_for_accounting_visibility(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_commands(
        monkeypatch,
        ["", f"123|{JOB_NAME}|RUNNING|0:0", f"123|{JOB_NAME}|COMPLETED|0:0"],
    )

    results, _ = scheduler.run("main.py", str(artifacts))

    assert results["metrics"]["combined_score"] == 42
    assert command.call_count == 6


@pytest.mark.parametrize(
    "record",
    [
        "",
        f"123.batch|{JOB_NAME}|COMPLETED|0:0",
        f"456|{JOB_NAME}|COMPLETED|0:0",
        f"123|{JOB_NAME}|COMPLETED",
        f"123|{JOB_NAME}|COMPLETED|0:0\n123|{JOB_NAME}|FAILED|1:0",
        f"123|{JOB_NAME}||0:0",
        f"123|{JOB_NAME}|RUNNING|0:0",
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

    assert command.call_count == 13


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
            monitor_evaluation("local-test", str(artifacts), job_name=JOB_NAME)[
                "metrics"
            ]["combined_score"]
            == 42
        )
    else:
        with pytest.raises(RuntimeError, match=f"status {exit_code}"):
            monitor_evaluation("local-test", str(artifacts), job_name=JOB_NAME)
    assert sleep.call_count == 2
    command.assert_not_called()


def test_unknown_local_fallback_fails_closed(artifacts: Path) -> None:
    with pytest.raises(RuntimeError, match="Unknown local fallback"):
        monitor_evaluation("local-unknown", str(artifacts), job_name=JOB_NAME)


def test_unavailable_accounting_has_finite_retries(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=""),
            *[
                subprocess.CalledProcessError(1, "sacct"),
                subprocess.CompletedProcess([], 0, stdout=""),
            ]
            * 6,
        ]
    )
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="accounting.*123"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 13


@pytest.mark.parametrize(
    "failure", ["exit", "timeout", "missing-command", "other-invalid"]
)
def test_status_lookup_errors_are_bounded(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    attempts = 0

    def run(cmd, **kwargs):
        nonlocal attempts
        attempts += 1
        assert attempts <= 4, "Status retries exceeded the scratch guard"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 30)
        if failure == "missing-command":
            raise FileNotFoundError("squeue unavailable")
        if failure == "other-invalid":
            raise subprocess.CalledProcessError(
                1, cmd, stderr="Invalid job id string: 123"
            )
        raise subprocess.CalledProcessError(1, cmd, stderr="Controller unavailable")

    monkeypatch.setattr(slurm.subprocess, "run", run)
    monkeypatch.setattr(slurm.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="status.*123.*3"):
        scheduler.run("main.py", str(artifacts))
    assert attempts == 3


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_retired_job_reaches_authoritative_accounting(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    attempts = 0

    def run(cmd, **kwargs):
        nonlocal attempts
        attempts += 1
        assert attempts <= 4, "Retired job was retried instead of checked"
        if cmd[0] == "squeue":
            raise subprocess.CalledProcessError(
                1, cmd, stderr="slurm_load_jobs error: Invalid job id specified\n"
            )
        assert cmd[0] == "sacct"
        return subprocess.CompletedProcess(cmd, 0, stdout=f"123|{JOB_NAME}|{state}|0:0")

    monkeypatch.setattr(slurm.subprocess, "run", run)
    monkeypatch.setattr(slurm.time, "sleep", Mock())

    if state == "COMPLETED":
        assert (
            scheduler.run("main.py", str(artifacts))[0]["metrics"]["combined_score"]
            == 42
        )
    else:
        with pytest.raises(RuntimeError, match=state):
            scheduler.run("main.py", str(artifacts))
    assert attempts == 2


def test_active_jobs_reset_lookup_errors_without_a_queue_deadline(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = subprocess.CalledProcessError(
        1, "squeue", stderr="Controller unavailable"
    )
    pending = subprocess.CompletedProcess([], 0, stdout="PENDING")
    running = subprocess.CompletedProcess([], 0, stdout="RUNNING")
    command = Mock(
        side_effect=[
            failure,
            failure,
            *[pending] * 100,
            failure,
            failure,
            running,
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess([], 0, stdout=f"123|{JOB_NAME}|COMPLETED|0:0"),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", sleep)
    scheduler.config.time = "00:00:01"

    result, _ = scheduler.run("main.py", str(artifacts))

    assert result["metrics"]["combined_score"] == 42
    assert sleep.call_count == 105


def test_status_and_accounting_commands_have_execution_deadlines(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = mock_commands(monkeypatch, [f"123|{JOB_NAME}|COMPLETED|0:0"])

    scheduler.run("main.py", str(artifacts))

    assert [call.kwargs.get("timeout") for call in command.call_args_list] == [30, 30]


def test_accounting_command_timeouts_exhaust_grace_period(
    scheduler: JobScheduler, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=""),
            *[
                subprocess.TimeoutExpired("sacct", 30),
                subprocess.CompletedProcess([], 0, stdout=""),
            ]
            * 6,
        ]
    )
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr(slurm.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="accounting.*123"):
        scheduler.run("main.py", str(artifacts))
    assert command.call_count == 13


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
def test_sync_slurm_submission_has_execution_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    command = Mock(side_effect=subprocess.TimeoutExpired("sbatch", 30))
    monkeypatch.setattr(slurm.subprocess, "run", command)
    monkeypatch.setattr("shinka.launch.scheduler.uuid4", lambda: UUID(int=0))
    config = (
        SlurmCondaJobConfig()
        if backend == "slurm_conda"
        else SlurmDockerJobConfig(image_tar_path="/tmp/image.tar")
    )
    instance = JobScheduler(backend, config)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            instance.run("main.py", str(tmp_path))
    finally:
        instance.shutdown()

    command.assert_called_once()
    assert command.call_args.args[0][:2] == ["sbatch", f"--job-name={JOB_NAME}"]
    assert command.call_args.kwargs.get("timeout") == 30


@pytest.mark.parametrize("backend", ["slurm_conda", "slurm_docker"])
def test_async_submission_keeps_existing_timeout_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    command = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout="Submitted batch job 123\n"
        )
    )
    monkeypatch.setattr(slurm.subprocess, "run", command)
    config = (
        SlurmCondaJobConfig()
        if backend == "slurm_conda"
        else SlurmDockerJobConfig(image_tar_path="/tmp/image.tar")
    )
    instance = JobScheduler(backend, config)
    try:
        assert instance.submit_async("main.py", str(tmp_path)) == "123"
    finally:
        instance.shutdown()
    assert len(command.call_args.args[0]) == 2
    assert command.call_args.kwargs.get("timeout") is None


@pytest.mark.parametrize("stalled", ["pull", "save"])
def test_sync_cold_docker_cache_timeout_defers_to_batch_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stalled: str
) -> None:
    monkeypatch.setattr(slurm, "DOCKER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(slurm, "CACHE_MANIFEST", tmp_path / "manifest.json")
    calls: list[str] = []

    def run(cmd, **kwargs):
        assert kwargs.get("timeout") == 30
        if cmd[0] == "sbatch":
            calls.append("sbatch")
            assert "docker pull ubuntu:latest" in Path(cmd[-1]).read_text()
            raise subprocess.TimeoutExpired(cmd, 30)
        assert cmd[0] == "docker"
        action = cmd[1]
        calls.append(action)
        if action == "save":
            Path(cmd[-1]).write_text("incomplete tar")
        if action == stalled:
            raise subprocess.TimeoutExpired(cmd, 30)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(slurm.subprocess, "run", run)
    instance = JobScheduler("slurm_docker", SlurmDockerJobConfig())
    try:
        with pytest.raises(subprocess.TimeoutExpired, match="sbatch"):
            instance.run("main.py", str(tmp_path / "results"))
    finally:
        instance.shutdown()

    expected_docker_calls = ["pull"] if stalled == "pull" else ["pull", "save"]
    assert calls == [*expected_docker_calls, "sbatch"]
    assert not (tmp_path / "ubuntu_latest.tar").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_async_cold_docker_cache_installs_image_without_seed_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slurm, "DOCKER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(slurm, "CACHE_MANIFEST", tmp_path / "manifest.json")

    def run(cmd, **kwargs):
        assert kwargs.get("timeout") is None
        if cmd[:2] == ["docker", "save"]:
            Path(cmd[-1]).write_text("complete tar")
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 123")

    monkeypatch.setattr(slurm.subprocess, "run", run)
    instance = JobScheduler("slurm_docker", SlurmDockerJobConfig())
    try:
        assert instance.submit_async("main.py", str(tmp_path / "results")) == "123"
    finally:
        instance.shutdown()

    assert (tmp_path / "ubuntu_latest.tar").read_text() == "complete tar"
    assert slurm.load_cache_manifest() == {"ubuntu:latest": "ubuntu_latest.tar"}


def test_async_docker_submission_can_proceed_with_readonly_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slurm, "DOCKER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(slurm, "CACHE_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(
        slurm.tempfile, "TemporaryDirectory", Mock(side_effect=PermissionError())
    )
    run = Mock(
        return_value=subprocess.CompletedProcess([], 0, stdout="Submitted job 123")
    )
    monkeypatch.setattr(slurm.subprocess, "run", run)
    instance = JobScheduler("slurm_docker", SlurmDockerJobConfig())
    try:
        assert instance.submit_async("main.py", str(tmp_path / "results")) == "123"
    finally:
        instance.shutdown()

    assert [call.args[0][0] for call in run.call_args_list] == ["docker", "sbatch"]


@pytest.mark.parametrize("terminal_state", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_suspended_job_waits_until_terminal_state(
    scheduler: JobScheduler,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
) -> None:
    states = iter(["SUSPENDED"] * 8 + ["RUNNING", terminal_state])
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd[0])
        if cmd[0] == "squeue":
            assert "--states=all" in cmd
            assert "--format=%T" in cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=next(states))
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"123|{JOB_NAME}|{terminal_state}|0:0"
        )

    sleep = Mock()
    monkeypatch.setattr(slurm.subprocess, "run", run)
    monkeypatch.setattr(slurm.time, "sleep", sleep)

    if terminal_state == "COMPLETED":
        assert (
            scheduler.run("main.py", str(artifacts))[0]["metrics"]["combined_score"]
            == 42
        )
    else:
        with pytest.raises(RuntimeError, match=terminal_state):
            scheduler.run("main.py", str(artifacts))
    assert commands == ["squeue"] * 10 + ["sacct"]
    assert sleep.call_count == 9
