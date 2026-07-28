"""Trusted in-container wrapper for secure single-file evaluation.

It preserves the evaluator's public CLI, captures its output with a fixed
budget, and emits a bounded tar response for the host launcher. This file uses
only the Python standard library because it is mounted into the evaluator image
at runtime.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Optional

_CHUNK_SIZE = 64 * 1024
_RESULT_FILES = ("metrics.json", "correct.json")


class _OutputBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self.limited = False
        self.lock = threading.Lock()

    def take(self, data: bytes) -> tuple[bytes, bool]:
        with self.lock:
            remaining = self.limit - self.used
            accepted = data[: max(0, remaining)]
            self.used += len(accepted)
            exceeded = len(data) > len(accepted)
            if exceeded:
                self.limited = True
            return accepted, exceeded


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (PermissionError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _capture_stream(
    pipe: Optional[BinaryIO],
    destination: BinaryIO,
    budget: _OutputBudget,
    process: subprocess.Popen[bytes],
) -> None:
    if pipe is None:
        return
    try:
        while True:
            data = pipe.read(_CHUNK_SIZE)
            if not data:
                return
            accepted, exceeded = budget.take(data)
            if accepted:
                destination.write(accepted)
            if exceeded:
                _terminate_process_group(process)
                return
    except (OSError, ValueError):
        return
    finally:
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(value))


def _add_open_file(
    archive: tarfile.TarFile,
    name: str,
    file_handle: BinaryIO,
    size: int,
) -> None:
    file_handle.seek(0)
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    archive.addfile(info, file_handle)


def _read_regular_result(path: Path, max_bytes: int) -> Optional[bytes]:
    try:
        initial = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(initial.st_mode) or initial.st_size > max_bytes:
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as result_file:
        current = os.fstat(result_file.fileno())
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != initial.st_dev
            or current.st_ino != initial.st_ino
            or current.st_size > max_bytes
        ):
            return None
        value = result_file.read(max_bytes + 1)
        return value if len(value) <= max_bytes else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--program-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-output-bytes", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--max-result-file-bytes", required=True, type=int)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        args.max_output_bytes <= 0
        or args.timeout_seconds <= 0
        or args.max_result_file_bytes <= 0
    ):
        parser.error("output limits must be positive")
    if args.extra_args[:1] == ["--"]:
        args.extra_args = args.extra_args[1:]
    return args


def main() -> int:
    args = _parse_args()
    metadata: dict[str, object] = {
        "returncode": 127,
        "output_limited": False,
        "timed_out": False,
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_capture,
        tempfile.TemporaryFile(mode="w+b") as stderr_capture,
    ):
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    args.evaluator,
                    "--program_path",
                    args.program_path,
                    "--results_dir",
                    args.results_dir,
                    *args.extra_args,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            budget = _OutputBudget(args.max_output_bytes)
            threads = (
                threading.Thread(
                    target=_capture_stream,
                    args=(process.stdout, stdout_capture, budget, process),
                    daemon=True,
                ),
                threading.Thread(
                    target=_capture_stream,
                    args=(process.stderr, stderr_capture, budget, process),
                    daemon=True,
                ),
            )
            for thread in threads:
                thread.start()
            try:
                return_code = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                return_code = process.wait()
                metadata["timed_out"] = True
            # A legacy evaluator can leave descendants behind. They must not
            # keep the capture pipes open or race result export after their
            # immediate parent has exited.
            _terminate_process_group(process)
            for thread in threads:
                thread.join(timeout=1.0)
            if any(thread.is_alive() for thread in threads):
                # An adversarial child can leave a pipe open after escaping the
                # evaluator process group. Do not let that block the wrapper
                # indefinitely; Docker tears down the remaining container
                # processes when this wrapper exits.
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except (OSError, ValueError):
                            pass
                metadata["error"] = "evaluator left an output pipe open"
                metadata["output_limited"] = True
            metadata["returncode"] = return_code
            metadata["output_limited"] = (
                bool(metadata["output_limited"]) or budget.limited
            )
        except OSError as exc:
            metadata["error"] = str(exc)
        finally:
            if process is not None:
                _terminate_process_group(process)

        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            _add_bytes(
                archive,
                "meta.json",
                json.dumps(metadata, sort_keys=True).encode("utf-8"),
            )
            _add_open_file(
                archive,
                "stdout.log",
                stdout_capture,
                stdout_capture.tell(),
            )
            _add_open_file(
                archive,
                "stderr.log",
                stderr_capture,
                stderr_capture.tell(),
            )
            results_dir = Path(args.results_dir)
            for name in _RESULT_FILES:
                value = _read_regular_result(
                    results_dir / name, args.max_result_file_bytes
                )
                if value is not None:
                    _add_bytes(archive, name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
