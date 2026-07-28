"""Hardened Docker execution for Shinka's existing single-file evaluators.

This module deliberately preserves the ``evaluate.py --program_path --results_dir``
contract. It confines that evaluator *and* its one candidate file to one
networkless container. The host result directory is never mounted into that
container: a trusted in-container wrapper exports bounded result data over the
Docker attach stream instead.

This is host containment, not a confidentiality or score-integrity boundary
between an in-process candidate and its evaluator. A stronger boundary would
require a different evaluator API.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TextIO

_PINNED_IMAGE_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_SAFE_USER_RE = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")
_SAFE_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MAX_RESULT_FILE_BYTES = 8 * 1024 * 1024
_MAX_META_BYTES = 64 * 1024
_TRANSPORT_OVERHEAD_BYTES = 1024 * 1024
_RESULT_FILES = ("metrics.json", "correct.json")
_LOG_FILES = ("job_log.out", "job_log.err")
_ARCHIVE_FILES = ("meta.json", "stdout.log", "stderr.log", *_RESULT_FILES)


class SecureDockerError(RuntimeError):
    """Raised when secure Docker preflight or containment fails."""


def validate_pinned_image(image: str) -> str:
    """Require an immutable image digest; tags are never a secure reference."""
    if not isinstance(image, str):
        raise SecureDockerError("secure_docker.image must be a string")
    normalized = image.strip()
    if not _PINNED_IMAGE_RE.fullmatch(normalized):
        raise SecureDockerError(
            "secure_docker.image must be an immutable image digest "
            "(for example, registry.example/evaluator@sha256:<64-hex-digest>)"
        )
    return normalized


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SecureDockerError(f"{name} must be a positive integer")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SecureDockerError(f"{name} must be a positive number")
    return float(value)


def _safe_existing_file(value: str | Path, *, label: str) -> Path:
    source = Path(value).expanduser()
    if source.is_symlink():
        raise SecureDockerError(f"{label} cannot be a symlink")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise SecureDockerError(f"{label} does not exist: {source}") from exc
    if not resolved.is_file():
        raise SecureDockerError(f"{label} must be a regular file: {resolved}")
    return resolved


def _safe_existing_directory(value: str | Path, *, label: str) -> Path:
    source = Path(value).expanduser()
    if source.is_symlink():
        raise SecureDockerError(f"{label} cannot be a symlink")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise SecureDockerError(f"{label} does not exist: {source}") from exc
    if not resolved.is_dir():
        raise SecureDockerError(f"{label} must be a directory: {resolved}")
    return resolved


def _validate_read_only_tree(root: Path, *, label: str) -> None:
    """Reject special files and nested mounts that could escape a read-only bind."""
    try:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            if directory_path != root and directory_path.is_mount():
                raise SecureDockerError(
                    f"{label} cannot contain nested mount points: {directory_path}"
                )
            for name in (*directory_names, *file_names):
                path = directory_path / name
                if path.is_mount():
                    raise SecureDockerError(
                        f"{label} cannot contain nested mount points: {path}"
                    )
                mode = path.lstat().st_mode
                if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
                    raise SecureDockerError(
                        f"{label} cannot contain FIFO or socket files: {path}"
                    )
                if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
                    raise SecureDockerError(
                        f"{label} cannot contain device files: {path}"
                    )
    except OSError as exc:
        raise SecureDockerError(f"could not inspect {label}") from exc


def _safe_results_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    requested.mkdir(parents=True, exist_ok=True)
    if requested.is_symlink():
        raise SecureDockerError("results directory cannot be a symlink")
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise SecureDockerError("results directory must be a directory")
    if resolved == Path(resolved.anchor):
        raise SecureDockerError("results directory cannot be the filesystem root")
    for name in (*_LOG_FILES, *_RESULT_FILES):
        try:
            (resolved / name).lstat()
        except FileNotFoundError:
            continue
        raise SecureDockerError(
            f"results directory already contains {name}; secure_docker requires "
            "a fresh per-evaluation directory"
        )
    return resolved


def _open_private_text_output(results_dir: Path, name: str) -> TextIO:
    """Create a host output file before the container is allowed to start."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(results_dir / name, flags, 0o600)
    except OSError as exc:
        raise SecureDockerError(f"could not safely create {name}") from exc
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def _open_private_binary_output(results_dir: Path, name: str) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(results_dir / name, flags, 0o600)
    except OSError as exc:
        raise SecureDockerError(f"could not safely create {name}") from exc
    return os.fdopen(descriptor, "wb")


def _remove_private_output(results_dir: Path, name: str) -> None:
    path = results_dir / name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(metadata.st_mode):
        try:
            path.unlink()
        except OSError:
            pass


def _mount(source: Path, target: str, *, read_only: bool) -> list[str]:
    source_text = str(source)
    if "," in source_text or "\x00" in source_text:
        raise SecureDockerError("container mount source contains an unsafe character")
    destination = PurePosixPath(target)
    if not destination.is_absolute() or ".." in destination.parts:
        raise SecureDockerError(
            "container mount target must be absolute and normalized"
        )
    if "," in str(destination):
        raise SecureDockerError("container mount target contains an unsafe character")
    specification = f"type=bind,src={source},dst={destination}"
    if read_only:
        specification += ",readonly"
    return ["--mount", specification]


def _default_sandbox_user() -> str:
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        raise SecureDockerError(
            "secure_docker requires sandbox_user on platforms without POSIX user IDs"
        )
    uid = os.getuid()
    gid = os.getgid()
    if uid == 0 or gid == 0:
        raise SecureDockerError(
            "secure_docker refuses to run as root; configure a non-root sandbox_user"
        )
    return f"{uid}:{gid}"


def _validate_sandbox_user(value: str | None) -> str:
    if value is not None and not isinstance(value, str):
        raise SecureDockerError("sandbox_user must be a non-root numeric UID:GID pair")
    user = _default_sandbox_user() if value is None else value.strip()
    if not _SAFE_USER_RE.fullmatch(user):
        raise SecureDockerError(
            "sandbox_user must be a non-root numeric UID:GID pair, "
            "for example 1000:1000"
        )
    return user


def _docker_command(executable: str) -> str:
    command = executable.strip()
    if not command:
        raise SecureDockerError("container_executable cannot be empty")
    if shutil.which(command) is None:
        raise SecureDockerError(
            f"secure_docker requires a Docker-compatible executable: {command}"
        )
    return command


def _run_checked(
    argv: list[str],
    *,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SecureDockerError("secure Docker preflight could not run") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        detail = f": {diagnostic[-500:]}" if diagnostic else ""
        raise SecureDockerError(f"secure Docker command was rejected{detail}")
    return completed


def _preflight(
    *,
    executable: str,
    image: str,
    require_rootless: bool,
    allow_rootful_dedicated_vm: bool,
) -> None:
    inspected_image = _run_checked([executable, "image", "inspect", image])
    try:
        image_data = json.loads(inspected_image.stdout)
        image_config = image_data[0].get("Config") or {}
    except (IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SecureDockerError("Docker returned an invalid image inspection") from exc
    if image_config.get("Volumes"):
        raise SecureDockerError(
            "secure_docker images cannot declare Docker volumes; use bounded tmpfs "
            "storage instead"
        )
    info = _run_checked([executable, "info", "--format", "{{json .SecurityOptions}}"])
    try:
        options = json.loads(info.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecureDockerError("Docker returned invalid security options") from exc
    if not isinstance(options, list):
        raise SecureDockerError("Docker returned invalid security options")
    normalized_options = [str(option).lower() for option in options]
    if not any("seccomp" in option for option in normalized_options):
        raise SecureDockerError(
            "secure_docker requires Docker's default seccomp profile"
        )
    if (
        platform.system() == "Linux"
        and require_rootless
        and not allow_rootful_dedicated_vm
        and not any("rootless" in option for option in normalized_options)
    ):
        raise SecureDockerError(
            "secure_docker requires a rootless Docker engine on Linux. "
            "Set allow_rootful_dedicated_vm=true only for a dedicated container VM."
        )


def _safe_eval_environment(values: Mapping[str, str]) -> list[str]:
    result: list[str] = []
    for name, value in sorted(values.items()):
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not _SAFE_ENV_NAME_RE.fullmatch(name)
            or "\x00" in value
        ):
            raise SecureDockerError("invalid evaluation environment entry")
        result.extend(["--env", f"{name}={value}"])
    return result


def build_create_argv(
    *,
    executable: str,
    container_name: str,
    image: str,
    evaluator_root: Path,
    runtime_root: Path,
    eval_relative_path: PurePosixPath,
    candidate_path: Path,
    extra_cmd_args: Mapping[str, Any],
    eval_environment: Mapping[str, str],
    sandbox_user: str,
    memory_bytes: int,
    cpus: float,
    pids_limit: int,
    open_files_limit: int,
    max_output_bytes: int,
    timeout_seconds: int,
    tmpfs_bytes: int,
    result_tmpfs_bytes: int,
    python_executable: str,
) -> list[str]:
    """Build a shell-free, policy-fixed Docker ``create`` command."""
    if not container_name.startswith("shinka-secure-eval-"):
        raise SecureDockerError("secure Docker container name has an invalid prefix")
    if not python_executable or "/" in python_executable:
        raise SecureDockerError("python_executable must be a simple executable name")

    candidate_name = candidate_path.name
    if (
        not candidate_name
        or candidate_name != Path(candidate_name).name
        or "," in candidate_name
        or "\x00" in candidate_name
    ):
        raise SecureDockerError("candidate filename is unsafe")

    evaluator_path = PurePosixPath("/workspace/evaluator") / eval_relative_path
    candidate_target = PurePosixPath("/workspace/candidate") / candidate_name
    result_target = PurePosixPath("/workspace/results")

    argv = [
        executable,
        "create",
        "--pull",
        "never",
        "--rm",
        "--name",
        container_name,
        "--init",
        "--network",
        "none",
        "--no-healthcheck",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        str(memory_bytes),
        "--memory-swap",
        str(memory_bytes),
        "--cpus",
        str(cpus),
        "--ulimit",
        f"nofile={open_files_limit}:{open_files_limit}",
        "--user",
        sandbox_user,
        "--workdir",
        "/workspace/evaluator",
        "--entrypoint",
        python_executable,
        "--log-driver",
        "none",
        "--label",
        "io.shinka.managed=true",
        "--label",
        "io.shinka.role=single-file-evaluation",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={tmpfs_bytes},mode=1777",
        "--tmpfs",
        "/run:rw,nosuid,nodev,noexec,size=16m,mode=755",
        "--tmpfs",
        f"/workspace/results:rw,nosuid,nodev,noexec,size={result_tmpfs_bytes},mode=1777",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    argv.extend(_safe_eval_environment(eval_environment))
    argv.extend(_mount(evaluator_root, "/workspace/evaluator", read_only=True))
    argv.extend(_mount(runtime_root, "/workspace/runtime", read_only=True))
    argv.extend(_mount(candidate_path, str(candidate_target), read_only=True))
    argv.extend(
        [
            image,
            "/workspace/runtime/secure_runner.py",
            "--evaluator",
            str(evaluator_path),
            "--program-path",
            str(candidate_target),
            "--results-dir",
            str(result_target),
            "--max-output-bytes",
            str(max_output_bytes),
            "--timeout-seconds",
            str(timeout_seconds),
            "--max-result-file-bytes",
            str(_MAX_RESULT_FILE_BYTES),
            "--",
        ]
    )
    for key, value in extra_cmd_args.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise SecureDockerError(
                "extra_cmd_args keys must be non-empty argument names"
            )
        argv.extend([f"--{key}", str(value)])
    return argv


def _inspect_container(executable: str, container_id: str) -> Mapping[str, Any]:
    raw = _run_checked([executable, "inspect", container_id]).stdout
    try:
        parsed = json.loads(raw)
        return parsed[0]
    except (IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SecureDockerError(
            "Docker returned an invalid container inspection"
        ) from exc


def _verify_container(
    *,
    executable: str,
    container_id: str,
    image: str,
    sandbox_user: str,
    memory_bytes: int,
    cpus: float,
    pids_limit: int,
    open_files_limit: int,
) -> None:
    data = _inspect_container(executable, container_id)
    config = data.get("Config") or {}
    host = data.get("HostConfig") or {}
    labels = config.get("Labels") or {}
    if labels.get("io.shinka.managed") != "true":
        raise SecureDockerError("secure container is missing its ownership label")
    if str(config.get("Image", "")) != image:
        raise SecureDockerError("secure container image differs from the pinned image")
    if str(config.get("User", "")) != sandbox_user:
        raise SecureDockerError("secure container user differs from the launch policy")
    if not host.get("AutoRemove"):
        raise SecureDockerError(
            "secure container must be removed automatically after evaluation"
        )
    if (host.get("RestartPolicy") or {}).get("Name") not in ("", "no"):
        raise SecureDockerError("secure container unexpectedly has a restart policy")
    if host.get("Privileged") or host.get("NetworkMode") != "none":
        raise SecureDockerError(
            "secure container has an unsafe privilege or network mode"
        )
    if host.get("PidMode") == "host" or host.get("IpcMode") == "host":
        raise SecureDockerError("secure container shares a host namespace")
    if host.get("UsernsMode") == "host" or host.get("CgroupnsMode") == "host":
        raise SecureDockerError("secure container shares an unsafe host namespace")
    if host.get("Devices") or host.get("DeviceRequests"):
        raise SecureDockerError("secure container unexpectedly has device access")
    mounts = data.get("Mounts") or []
    if not isinstance(mounts, list) or any(
        isinstance(mount, dict) and mount.get("Type") == "volume" for mount in mounts
    ):
        raise SecureDockerError("secure container unexpectedly has a Docker volume")
    if host.get("Binds"):
        raise SecureDockerError("secure container unexpectedly has legacy bind mounts")
    if set(host.get("CapAdd") or []):
        raise SecureDockerError("secure container unexpectedly adds capabilities")
    if "ALL" not in set(host.get("CapDrop") or []):
        raise SecureDockerError("secure container does not drop all capabilities")
    options = {str(value) for value in host.get("SecurityOpt") or []}
    if not any(value.startswith("no-new-privileges") for value in options):
        raise SecureDockerError("secure container lacks no-new-privileges")
    if not bool(host.get("ReadonlyRootfs")):
        raise SecureDockerError("secure container root filesystem is writable")
    if int(host.get("PidsLimit") or 0) != pids_limit:
        raise SecureDockerError(
            "secure container PID limit differs from the launch policy"
        )
    if int(host.get("Memory") or 0) != memory_bytes:
        raise SecureDockerError(
            "secure container memory limit differs from the launch policy"
        )
    if int(host.get("MemorySwap") or 0) != memory_bytes:
        raise SecureDockerError(
            "secure container swap limit differs from the launch policy"
        )
    if int(host.get("NanoCpus") or 0) != int(cpus * 1_000_000_000):
        raise SecureDockerError(
            "secure container CPU limit differs from the launch policy"
        )
    expected_limit = {("nofile", open_files_limit, open_files_limit)}
    actual_limit = {
        (str(item.get("Name")), int(item.get("Soft", 0)), int(item.get("Hard", 0)))
        for item in host.get("Ulimits") or []
    }
    if actual_limit != expected_limit:
        raise SecureDockerError(
            "secure container file limit differs from the launch policy"
        )
    if (host.get("LogConfig") or {}).get("Type") != "none":
        raise SecureDockerError("secure container unexpectedly persists Docker logs")


def _remove_container(executable: str, container_id: str) -> None:
    try:
        subprocess.run(
            [executable, "rm", "--force", container_id],
            capture_output=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Do not mask the original evaluator result with best-effort cleanup.
        pass


class SecureDockerProcess:
    """A Docker-attached process compatible with the local scheduler monitor."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        executable: str,
        container_id: str,
        stdout_file: TextIO,
        stderr_file: TextIO,
        max_output_bytes: int,
        results_dir: Path,
    ) -> None:
        self.process = process
        self.executable = executable
        self.container_id = container_id
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.max_output_bytes = max_output_bytes
        self.results_dir = results_dir
        self._max_transport_bytes = (
            max_output_bytes + 2 * _MAX_RESULT_FILE_BYTES + _TRANSPORT_OVERHEAD_BYTES
        )
        self._transport_bytes = 0
        self._transport_lock = threading.Lock()
        self._remove_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._transport_limited = threading.Event()
        self._removed = False
        self._cleaned = False
        # These handles span the attached process lifetime and are closed in
        # ``cleanup_logging`` rather than a lexical context-manager scope.
        self._response_file = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        self._docker_stderr_file = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+b"
        )
        self._threads = (
            threading.Thread(
                target=self._capture_transport,
                args=(process.stdout, self._response_file),
                daemon=True,
            ),
            threading.Thread(
                target=self._capture_transport,
                args=(process.stderr, self._docker_stderr_file),
                daemon=True,
            ),
        )
        for thread in self._threads:
            thread.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.process, name)

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    @property
    def output_limited(self) -> bool:
        return self._transport_limited.is_set()

    def _capture_transport(self, pipe: BinaryIO | None, file_handle: BinaryIO) -> None:
        if pipe is None:
            return
        try:
            while True:
                chunk = pipe.read(64 * 1024)
                if not chunk:
                    return
                with self._transport_lock:
                    remaining = self._max_transport_bytes - self._transport_bytes
                    accepted = chunk[: max(0, remaining)]
                    self._transport_bytes += len(accepted)
                    exceeded = len(chunk) > len(accepted)
                if accepted:
                    file_handle.write(accepted)
                    file_handle.flush()
                if exceeded:
                    self._transport_limited.set()
                    self.kill()
                    return
        finally:
            pipe.close()

    def _remove_once(self) -> None:
        with self._remove_lock:
            if self._removed:
                return
            self._removed = True
        _remove_container(self.executable, self.container_id)

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def kill(self) -> None:
        self._remove_once()
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass

    def _write_stderr(self, message: str) -> None:
        self.stderr_file.write(message.rstrip() + "\n")
        self.stderr_file.flush()

    def _copy_archive_log(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> None:
        source = archive.extractfile(member)
        if source is None:
            raise SecureDockerError(f"secure runtime archive lacks {member.name}")
        destination = (
            self.stdout_file if member.name == "stdout.log" else self.stderr_file
        )
        remaining = member.size
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                raise SecureDockerError(
                    f"secure runtime archive truncates {member.name}"
                )
            destination.write(chunk.decode("utf-8", errors="replace"))
            remaining -= len(chunk)
        destination.flush()

    def _copy_archive_result(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> None:
        source = archive.extractfile(member)
        if source is None:
            raise SecureDockerError(f"secure runtime archive lacks {member.name}")
        payload = source.read(member.size + 1)
        if len(payload) != member.size:
            raise SecureDockerError(f"secure runtime archive truncates {member.name}")
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecureDockerError(
                f"secure runtime result {member.name} is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise SecureDockerError(
                f"secure runtime result {member.name} must be a JSON object"
            )
        destination = _open_private_binary_output(self.results_dir, member.name)
        try:
            destination.write(payload)
        except Exception:
            destination.close()
            _remove_private_output(self.results_dir, member.name)
            raise
        else:
            destination.close()

    def _read_archive_meta(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> Mapping[str, Any]:
        source = archive.extractfile(member)
        if source is None:
            raise SecureDockerError("secure runtime archive lacks meta.json")
        payload = source.read(_MAX_META_BYTES + 1)
        if len(payload) > _MAX_META_BYTES:
            raise SecureDockerError("secure runtime metadata exceeds its limit")
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecureDockerError("secure runtime metadata is invalid") from exc
        if not isinstance(parsed, dict):
            raise SecureDockerError("secure runtime metadata must be an object")
        return parsed

    def _decode_response(self) -> None:
        self._response_file.flush()
        self._response_file.seek(0)
        seen: set[str] = set()
        log_bytes = 0
        metadata: Mapping[str, Any] | None = None
        try:
            with tarfile.open(fileobj=self._response_file, mode="r:") as archive:
                for member in archive:
                    if member.name not in _ARCHIVE_FILES or member.name in seen:
                        raise SecureDockerError(
                            "secure runtime archive has an invalid entry"
                        )
                    if not member.isfile():
                        raise SecureDockerError(
                            "secure runtime archive entry is not a file"
                        )
                    seen.add(member.name)
                    if member.name == "meta.json":
                        metadata = self._read_archive_meta(archive, member)
                    elif member.name in ("stdout.log", "stderr.log"):
                        if (
                            member.size > self.max_output_bytes
                            or log_bytes + member.size > self.max_output_bytes
                        ):
                            raise SecureDockerError(
                                "secure runtime log output exceeds its limit"
                            )
                        self._copy_archive_log(archive, member)
                        log_bytes += member.size
                    else:
                        if member.size > _MAX_RESULT_FILE_BYTES:
                            raise SecureDockerError(
                                f"secure runtime result {member.name} exceeds its limit"
                            )
                        self._copy_archive_result(archive, member)
            if metadata is None:
                raise SecureDockerError("secure runtime did not emit metadata")
        except (OSError, RecursionError, SecureDockerError, tarfile.TarError) as exc:
            for name in _RESULT_FILES:
                _remove_private_output(self.results_dir, name)
            self._write_stderr(f"secure_docker: rejected runtime response: {exc}")
            return

        if metadata.get("output_limited"):
            self._write_stderr(
                "secure_docker: evaluation output exceeded its configured limit."
            )
        if metadata.get("timed_out"):
            self._write_stderr(
                "secure_docker: evaluation exceeded its configured wall-time limit."
            )
        return_code = metadata.get("returncode")
        if isinstance(return_code, int) and return_code != 0:
            self._write_stderr(
                f"secure_docker: evaluator exited with return code {return_code}."
            )
        wrapper_error = metadata.get("error")
        if isinstance(wrapper_error, str) and wrapper_error:
            self._write_stderr(f"secure_docker: runtime error: {wrapper_error}")

    def _append_docker_stderr(self) -> None:
        self._docker_stderr_file.flush()
        self._docker_stderr_file.seek(0)
        data = self._docker_stderr_file.read()
        if data:
            self._write_stderr(
                "secure_docker transport stderr:\n"
                + data.decode("utf-8", errors="replace")
            )

    def cleanup_logging(self) -> None:
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True

        for thread in self._threads:
            thread.join(timeout=2.0)
        # ``docker start --attach`` should return only after the container exits,
        # but remove it before parsing output so a malformed attach never leaves
        # candidate code running while host files are being promoted.
        self._remove_once()
        try:
            if self.output_limited:
                self._write_stderr(
                    "secure_docker: runtime transport exceeded its configured limit; "
                    "container was terminated."
                )
            else:
                self._decode_response()
            self._append_docker_stderr()
        finally:
            for file_handle in (self._response_file, self._docker_stderr_file):
                try:
                    file_handle.close()
                except OSError:
                    pass
            for file_handle in (self.stdout_file, self.stderr_file):
                try:
                    file_handle.close()
                except OSError:
                    pass


def submit(
    *,
    log_dir: str,
    program_path: str,
    eval_program_path: str,
    evaluator_root: str | None,
    image: str,
    container_executable: str,
    extra_cmd_args: Mapping[str, Any],
    eval_environment: Mapping[str, str],
    sandbox_user: str | None,
    memory_bytes: int,
    cpus: float,
    pids_limit: int,
    open_files_limit: int,
    max_output_bytes: int,
    timeout_seconds: int,
    tmpfs_bytes: int,
    result_tmpfs_bytes: int,
    python_executable: str,
    require_rootless: bool,
    allow_rootful_dedicated_vm: bool,
) -> SecureDockerProcess:
    """Launch one hardened evaluation container without invoking a shell."""
    executable = _docker_command(container_executable)
    immutable_image = validate_pinned_image(image)
    candidate = _safe_existing_file(program_path, label="candidate program")
    if candidate.is_mount():
        raise SecureDockerError("candidate program cannot be a mount point")
    evaluator = _safe_existing_file(eval_program_path, label="evaluation script")
    root = _safe_existing_directory(
        evaluator_root or str(evaluator.parent), label="evaluator_root"
    )
    if root == Path(root.anchor):
        raise SecureDockerError("evaluator_root cannot be the filesystem root")
    runtime_root = _safe_existing_directory(
        Path(__file__).parent, label="secure runtime"
    )
    _validate_read_only_tree(root, label="evaluator_root")
    _validate_read_only_tree(runtime_root, label="secure runtime")
    try:
        relative_evaluator = PurePosixPath(evaluator.relative_to(root).as_posix())
    except ValueError as exc:
        raise SecureDockerError(
            "evaluation script must be contained by evaluator_root"
        ) from exc
    results = _safe_results_directory(log_dir)
    user = _validate_sandbox_user(sandbox_user)
    memory = _validate_positive_int("memory_bytes", memory_bytes)
    cpu_limit = _validate_positive_float("cpus", cpus)
    pids = _validate_positive_int("pids_limit", pids_limit)
    open_files = _validate_positive_int("open_files_limit", open_files_limit)
    output_limit = _validate_positive_int("max_output_bytes", max_output_bytes)
    runtime_timeout = _validate_positive_int("timeout_seconds", timeout_seconds)
    tmpfs_limit = _validate_positive_int("tmpfs_bytes", tmpfs_bytes)
    result_tmpfs_limit = _validate_positive_int(
        "result_tmpfs_bytes", result_tmpfs_bytes
    )
    if result_tmpfs_limit < _MAX_RESULT_FILE_BYTES:
        raise SecureDockerError(
            "result_tmpfs_bytes must accommodate at least one bounded result file"
        )

    _preflight(
        executable=executable,
        image=immutable_image,
        require_rootless=require_rootless,
        allow_rootful_dedicated_vm=allow_rootful_dedicated_vm,
    )
    name = f"shinka-secure-eval-{uuid.uuid4().hex[:20]}"
    create_argv = build_create_argv(
        executable=executable,
        container_name=name,
        image=immutable_image,
        evaluator_root=root,
        runtime_root=runtime_root,
        eval_relative_path=relative_evaluator,
        candidate_path=candidate,
        extra_cmd_args=extra_cmd_args,
        eval_environment=eval_environment,
        sandbox_user=user,
        memory_bytes=memory,
        cpus=cpu_limit,
        pids_limit=pids,
        open_files_limit=open_files,
        max_output_bytes=output_limit,
        timeout_seconds=runtime_timeout,
        tmpfs_bytes=tmpfs_limit,
        result_tmpfs_bytes=result_tmpfs_limit,
        python_executable=python_executable,
    )
    created = _run_checked(create_argv, timeout=120.0)
    container_id = created.stdout.decode("utf-8", errors="replace").strip()
    if not container_id:
        raise SecureDockerError("Docker did not return a container ID")

    stdout_file: TextIO | None = None
    stderr_file: TextIO | None = None
    try:
        _verify_container(
            executable=executable,
            container_id=container_id,
            image=immutable_image,
            sandbox_user=user,
            memory_bytes=memory,
            cpus=cpu_limit,
            pids_limit=pids,
            open_files_limit=open_files,
        )
        # The container never receives this host directory. Opening these files
        # before ``docker start`` also avoids symlink races on error paths.
        stdout_file = _open_private_text_output(results, "job_log.out")
        stderr_file = _open_private_text_output(results, "job_log.err")
        process = subprocess.Popen(
            [executable, "start", "--attach", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        return SecureDockerProcess(
            process=process,
            executable=executable,
            container_id=container_id,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            max_output_bytes=output_limit,
            results_dir=results,
        )
    except Exception:
        for file_handle in (stdout_file, stderr_file):
            if file_handle is not None:
                try:
                    file_handle.close()
                except OSError:
                    pass
        for output_name in _LOG_FILES:
            _remove_private_output(results, output_name)
        _remove_container(executable, container_id)
        raise
