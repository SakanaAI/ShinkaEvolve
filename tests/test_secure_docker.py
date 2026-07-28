from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from shinka.launch import JobScheduler, SecureDockerJobConfig
from shinka.launch.secure_docker import (
    SecureDockerError,
    SecureDockerProcess,
    _validate_read_only_tree,
    build_create_argv,
    submit,
    validate_pinned_image,
)

IMAGE = "registry.example/shinka/evaluator@sha256:" + "a" * 64


def _value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _tar_response(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, value in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def test_pinned_image_rejects_mutable_references() -> None:
    assert validate_pinned_image(IMAGE) == IMAGE
    with pytest.raises(SecureDockerError, match="immutable image digest"):
        validate_pinned_image("python:3.12")


def test_preflight_rejects_images_with_writable_docker_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_checked(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        assert argv[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b'[{"Config": {"Volumes": {"/data": {}}}}]',
        )

    monkeypatch.setattr("shinka.launch.secure_docker._run_checked", fake_run_checked)
    with pytest.raises(SecureDockerError, match="cannot declare Docker volumes"):
        from shinka.launch.secure_docker import _preflight

        _preflight(
            executable="docker",
            image=IMAGE,
            require_rootless=False,
            allow_rootful_dedicated_vm=False,
        )


def test_preflight_requires_seccomp_support(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_checked(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=b'[{"Config": {}}]',
            )
        assert argv[:2] == ["docker", "info"]
        return subprocess.CompletedProcess(argv, 0, stdout=b'["rootless"]')

    monkeypatch.setattr("shinka.launch.secure_docker._run_checked", fake_run_checked)
    with pytest.raises(SecureDockerError, match="default seccomp"):
        from shinka.launch.secure_docker import _preflight

        _preflight(
            executable="docker",
            image=IMAGE,
            require_rootless=False,
            allow_rootful_dedicated_vm=False,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_evaluator_tree_rejects_special_files(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "service.pipe")
    with pytest.raises(SecureDockerError, match="FIFO or socket"):
        _validate_read_only_tree(tmp_path, label="evaluator_root")


@pytest.mark.parametrize(
    "suffix",
    [
        ".py",
        ".go",
        ".jl",
        ".f90",
        ".sv",
        ".rs",
        ".swift",
        ".cpp",
        ".cc",
        ".cxx",
        ".cu",
        ".json",
        ".wl",
        ".f95",
        ".f03",
        ".f08",
    ],
)
def test_create_command_preserves_single_file_candidate_contract(
    tmp_path: Path, suffix: str
) -> None:
    evaluator_root = tmp_path / "task"
    evaluator_root.mkdir()
    evaluator = evaluator_root / "evaluate.py"
    evaluator.write_text("# trusted evaluator\n", encoding="utf-8")
    candidate = tmp_path / f"main{suffix}"
    candidate.write_text("candidate\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()

    argv = build_create_argv(
        executable="docker",
        container_name="shinka-secure-eval-0123456789abcdef",
        image=IMAGE,
        evaluator_root=evaluator_root,
        runtime_root=runtime_root,
        eval_relative_path=Path("evaluate.py"),
        candidate_path=candidate,
        extra_cmd_args={"case_count": 10},
        eval_environment={"SHINKA_EVAL_VERBOSE": "0"},
        sandbox_user="1000:1000",
        memory_bytes=512 * 1024 * 1024,
        cpus=1.5,
        pids_limit=64,
        open_files_limit=128,
        max_output_bytes=1024,
        timeout_seconds=300,
        tmpfs_bytes=64 * 1024 * 1024,
        result_tmpfs_bytes=16 * 1024 * 1024,
        python_executable="python",
    )

    assert ["--network", "none"] == [
        "--network",
        _value_after(argv, "--network"),
    ]
    assert "--no-healthcheck" in argv
    assert _value_after(argv, "--pull") == "never"
    assert "--rm" in argv
    assert "--read-only" in argv
    assert ["--cap-drop", "ALL"] == ["--cap-drop", _value_after(argv, "--cap-drop")]
    assert ["--security-opt", "no-new-privileges:true"] == [
        "--security-opt",
        _value_after(argv, "--security-opt"),
    ]
    assert _value_after(argv, "--user") == "1000:1000"
    assert _value_after(argv, "--memory") == str(512 * 1024 * 1024)
    assert _value_after(argv, "--memory-swap") == str(512 * 1024 * 1024)
    assert _value_after(argv, "--cpus") == "1.5"
    assert _value_after(argv, "--pids-limit") == "64"
    assert _value_after(argv, "--log-driver") == "none"
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert f"src={evaluator_root},dst=/workspace/evaluator,readonly" in mounts[0]
    assert f"src={runtime_root},dst=/workspace/runtime,readonly" in mounts[1]
    assert (
        f"src={candidate},dst=/workspace/candidate/main{suffix},readonly" in mounts[2]
    )
    assert all("/workspace/results" not in mount for mount in mounts)
    assert any(
        value.startswith("/workspace/results:rw,nosuid,nodev,noexec,size=")
        for value in argv
    )
    assert argv[argv.index(IMAGE) :] == [
        IMAGE,
        "/workspace/runtime/secure_runner.py",
        "--evaluator",
        "/workspace/evaluator/evaluate.py",
        "--program-path",
        f"/workspace/candidate/main{suffix}",
        "--results-dir",
        "/workspace/results",
        "--max-output-bytes",
        "1024",
        "--timeout-seconds",
        "300",
        "--max-result-file-bytes",
        str(8 * 1024 * 1024),
        "--",
        "--case_count",
        "10",
    ]


def test_secure_config_requires_an_image() -> None:
    with pytest.raises(ValueError, match="image is required"):
        SecureDockerJobConfig()
    with pytest.raises(ValueError, match="time must be positive"):
        SecureDockerJobConfig(image=IMAGE, time="00:00:00")


def test_submit_uses_create_inspect_start_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = tmp_path / "evaluate.py"
    candidate = tmp_path / "main.go"
    results = tmp_path / "results"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    candidate.write_text("package main\n", encoding="utf-8")
    created_commands: list[list[str]] = []
    start_commands: list[list[str]] = []
    removed: list[str] = []

    class FakePopen:
        pid = 123
        returncode = 0

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            start_commands.append(argv)
            self.stdout = io.BytesIO(
                _tar_response(
                    {
                        "meta.json": json.dumps(
                            {
                                "returncode": 0,
                                "output_limited": False,
                                "timed_out": False,
                            }
                        ).encode(),
                        "stdout.log": b"evaluator output\n",
                        "stderr.log": b"",
                        "metrics.json": b'{"score": 1}',
                        "correct.json": b'{"correct": true}',
                    }
                )
            )
            self.stderr = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.returncode = -9

    def fake_run_checked(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        created_commands.append(argv)
        assert argv[1] == "create"
        return subprocess.CompletedProcess(argv, 0, stdout=b"container-id\n")

    monkeypatch.setattr(
        "shinka.launch.secure_docker._docker_command", lambda _: "docker"
    )
    monkeypatch.setattr("shinka.launch.secure_docker._preflight", lambda **_: None)
    monkeypatch.setattr("shinka.launch.secure_docker._run_checked", fake_run_checked)
    monkeypatch.setattr(
        "shinka.launch.secure_docker._verify_container", lambda **_: None
    )
    monkeypatch.setattr(
        "shinka.launch.secure_docker._remove_container",
        lambda _, cid: removed.append(cid),
    )
    monkeypatch.setattr("shinka.launch.secure_docker.subprocess.Popen", FakePopen)

    process = submit(
        log_dir=str(results),
        program_path=str(candidate),
        eval_program_path=str(evaluator),
        evaluator_root=None,
        image=IMAGE,
        container_executable="docker",
        extra_cmd_args={},
        eval_environment={"SHINKA_EVAL_VERBOSE": "1"},
        sandbox_user="1000:1000",
        memory_bytes=512 * 1024 * 1024,
        cpus=1.0,
        pids_limit=64,
        open_files_limit=128,
        max_output_bytes=1024,
        timeout_seconds=300,
        tmpfs_bytes=64 * 1024 * 1024,
        result_tmpfs_bytes=16 * 1024 * 1024,
        python_executable="python",
        require_rootless=True,
        allow_rootful_dedicated_vm=False,
    )

    process.cleanup_logging()

    assert created_commands and created_commands[0][:2] == ["docker", "create"]
    assert start_commands == [["docker", "start", "--attach", "container-id"]]
    assert removed == ["container-id"]
    assert (results / "job_log.out").read_text(encoding="utf-8") == "evaluator output\n"
    assert json.loads((results / "metrics.json").read_text(encoding="utf-8")) == {
        "score": 1
    }


def test_secure_runner_preserves_evaluator_cli_contract(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluate.py"
    candidate = tmp_path / "main.f90"
    results = tmp_path / "results"
    evaluator.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--program_path', required=True)
parser.add_argument('--results_dir', required=True)
parser.add_argument('--marker', required=True)
args = parser.parse_args()
assert Path(args.program_path).name == 'main.f90'
assert args.marker == 'kept'
Path(args.results_dir).mkdir(parents=True, exist_ok=True)
Path(args.results_dir, 'metrics.json').write_text(json.dumps({'score': 3}))
Path(args.results_dir, 'correct.json').write_text(json.dumps({'correct': True}))
print('evaluator stdout')
""",
        encoding="utf-8",
    )
    candidate.write_text("program main\nend program main\n", encoding="utf-8")
    secure_runner = Path(__file__).parents[1] / "shinka/launch/secure_runner.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(secure_runner),
            "--evaluator",
            str(evaluator),
            "--program-path",
            str(candidate),
            "--results-dir",
            str(results),
            "--max-output-bytes",
            "1024",
            "--timeout-seconds",
            "30",
            "--max-result-file-bytes",
            "1024",
            "--",
            "--marker",
            "kept",
        ],
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        entries = {
            member.name: archive.extractfile(member).read()
            for member in archive
            if member.isfile()
        }

    assert json.loads(entries["meta.json"]) == {
        "output_limited": False,
        "returncode": 0,
        "timed_out": False,
    }
    assert entries["stdout.log"] == b"evaluator stdout\n"
    assert json.loads(entries["metrics.json"]) == {"score": 3}
    assert json.loads(entries["correct.json"]) == {"correct": True}


def test_secure_runner_enforces_its_own_wall_time(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluate.py"
    candidate = tmp_path / "main.py"
    results = tmp_path / "results"
    evaluator.write_text(
        "import time\ntime.sleep(5)\n",
        encoding="utf-8",
    )
    candidate.write_text("print('candidate')\n", encoding="utf-8")
    secure_runner = Path(__file__).parents[1] / "shinka/launch/secure_runner.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(secure_runner),
            "--evaluator",
            str(evaluator),
            "--program-path",
            str(candidate),
            "--results-dir",
            str(results),
            "--max-output-bytes",
            "1024",
            "--timeout-seconds",
            "1",
            "--max-result-file-bytes",
            "1024",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        metadata = json.loads(archive.extractfile("meta.json").read())

    assert metadata["timed_out"] is True
    assert metadata["returncode"] != 0


def test_host_rejects_nonregular_runtime_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for name, value in {
            "meta.json": b'{"returncode": 0, "output_limited": false}',
            "stdout.log": b"",
            "stderr.log": b"",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
        link = tarfile.TarInfo("metrics.json")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)

    class FinishedProcess:
        pid = 321
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO(archive_buffer.getvalue())
            self.stderr = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.returncode = -9

    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(
        "shinka.launch.secure_docker._remove_container", lambda *_: None
    )
    process = SecureDockerProcess(
        process=FinishedProcess(),  # type: ignore[arg-type]
        executable="docker",
        container_id="container-id",
        stdout_file=(results / "job_log.out").open("w", encoding="utf-8"),
        stderr_file=(results / "job_log.err").open("w", encoding="utf-8"),
        max_output_bytes=1024,
        results_dir=results,
    )

    process.cleanup_logging()

    assert not (results / "metrics.json").exists()
    assert "rejected runtime response" in (results / "job_log.err").read_text(
        encoding="utf-8"
    )


def test_host_validates_result_json_before_promoting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FinishedProcess:
        pid = 321
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO(
                _tar_response(
                    {
                        "meta.json": b'{"returncode": 0}',
                        "stdout.log": b"",
                        "stderr.log": b"",
                        "metrics.json": b"[]",
                    }
                )
            )
            self.stderr = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.returncode = -9

    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(
        "shinka.launch.secure_docker._remove_container", lambda *_: None
    )
    process = SecureDockerProcess(
        process=FinishedProcess(),  # type: ignore[arg-type]
        executable="docker",
        container_id="container-id",
        stdout_file=(results / "job_log.out").open("w", encoding="utf-8"),
        stderr_file=(results / "job_log.err").open("w", encoding="utf-8"),
        max_output_bytes=1024,
        results_dir=results,
    )

    process.cleanup_logging()

    assert not (results / "metrics.json").exists()
    assert "rejected runtime response" in (results / "job_log.err").read_text(
        encoding="utf-8"
    )


def test_scheduler_submits_secure_container_with_legacy_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    process = object()

    def fake_submit(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return process

    monkeypatch.setattr("shinka.launch.scheduler.submit_secure_docker", fake_submit)
    scheduler = JobScheduler(
        job_type="secure_docker",
        config=SecureDockerJobConfig(
            image=IMAGE,
            evaluator_root="/trusted/task",
            extra_cmd_args={"seed": 42},
            numeric_threads_per_job=2,
        ),
    )

    submitted = scheduler.submit_async("/candidate/main.go", "/results/gen_1")

    assert submitted is process
    assert captured["program_path"] == "/candidate/main.go"
    assert captured["eval_program_path"] == "evaluate.py"
    assert captured["evaluator_root"] == "/trusted/task"
    assert captured["extra_cmd_args"] == {"seed": 42}
    assert captured["result_tmpfs_bytes"] == 64 * 1024 * 1024
    assert captured["timeout_seconds"] == 300
    assert captured["eval_environment"] == {
        "SHINKA_EVAL_VERBOSE": "1",
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MKL_DYNAMIC": "FALSE",
        "NUMEXPR_NUM_THREADS": "2",
        "NUMEXPR_MAX_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "BLIS_NUM_THREADS": "2",
        "GOTO_NUM_THREADS": "2",
    }
