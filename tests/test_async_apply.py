import asyncio
from pathlib import Path

import pytest

from shinka.edit import async_apply


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.kill_called = False
        self.wait_called = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.kill_called = True

    async def wait(self) -> None:
        self.wait_called = True


def test_run_validation_subprocess_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    async def fake_create_subprocess_exec(
        *args: str,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> _FakeProcess:
        recorded["args"] = args
        recorded["stdout"] = stdout
        recorded["stderr"] = stderr
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(
        async_apply.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    is_valid, error = asyncio.run(
        async_apply._run_validation_subprocess(
            "python",
            "-m",
            "py_compile",
            "candidate.py",
            timeout=7,
        )
    )

    assert is_valid is True
    assert error is None
    assert recorded["args"] == ("python", "-m", "py_compile", "candidate.py")
    assert recorded["stdout"] == asyncio.subprocess.PIPE
    assert recorded["stderr"] == asyncio.subprocess.PIPE


def test_run_validation_subprocess_returns_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(
        *args: str,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> _FakeProcess:
        return _FakeProcess(returncode=1, stderr=b"syntax error")

    monkeypatch.setattr(
        async_apply.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    is_valid, error = asyncio.run(
        async_apply._run_validation_subprocess(
            "g++", "-fsyntax-only", "bad.cpp", timeout=5
        )
    )

    assert is_valid is False
    assert error == "syntax error"


def test_run_validation_subprocess_timeout_kills_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProcess()

    async def fake_create_subprocess_exec(
        *args: str,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> _FakeProcess:
        return proc

    async def fake_wait_for(awaitable: object, timeout: int) -> object:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        del timeout
        raise asyncio.TimeoutError

    monkeypatch.setattr(
        async_apply.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(async_apply.asyncio, "wait_for", fake_wait_for)

    is_valid, error = asyncio.run(
        async_apply._run_validation_subprocess("swiftc", "candidate.swift", timeout=3)
    )

    assert is_valid is False
    assert error == "Validation timeout after 3s"
    assert proc.kill_called is True
    assert proc.wait_called is True


def test_validate_code_async_python_delegates_to_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        return True, None

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.py"), language="python", timeout=11
        )
    )

    assert is_valid is True
    assert error is None
    assert recorded["args"] == (
        "python",
        "-m",
        "py_compile",
        str(tmp_path / "candidate.py"),
    )
    assert recorded["timeout"] == 11


def test_validate_code_async_fortran_delegates_to_gfortran(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        return True, None

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.f90"), language="f95", timeout=17
        )
    )

    assert is_valid is True
    assert error is None
    assert recorded["args"] == (
        "gfortran",
        "-fsyntax-only",
        str(tmp_path / "candidate.f90"),
    )
    assert recorded["timeout"] == 17


def test_validate_code_async_json_delegates_to_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        return False, "bad json"

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.json"), language="json", timeout=13
        )
    )

    assert is_valid is False
    assert error == "bad json"
    assert recorded["args"] == ("jsonschema", str(tmp_path / "candidate.json"))
    assert recorded["timeout"] == 13


def test_validate_code_async_go_uses_read_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        raise AssertionError(f"unexpected compiler validation: {args}")

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fail_helper)
    candidate = tmp_path / "candidate.go"
    candidate.write_text("package main\nfunc main() {}\n", encoding="utf-8")

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="go", timeout=19)
    )

    assert is_valid is True
    assert error is None


def test_validate_code_async_wolfram_uses_wolframscript_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wolfram path must route through the shared wolframscript helpers
    so it honors WOLFRAMSCRIPT_BIN and the WSL bash-c wrap, and must
    escape the candidate path before embedding it in a Wolfram string."""
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        return True, None

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/opt/Wolfram/wolframscript",
    )
    monkeypatch.setattr("shinka.utils.wolfram.is_wsl", lambda: False)

    # A code_path containing a backslash and a quote must be escaped, not
    # passed raw into the f-string.
    candidate = tmp_path / 'odd"name\\file.wl'
    candidate.write_text(
        "(* EVOLVE-BLOCK-START *)\n(* EVOLVE-BLOCK-END *)", encoding="utf-8"
    )

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(candidate),
            language="wolfram",
            timeout=17,
        )
    )

    assert is_valid is True
    assert error is None
    args = recorded["args"]
    assert isinstance(args, tuple)
    assert args[0] == "/opt/Wolfram/wolframscript"
    assert "-code" in args
    code_arg = args[args.index("-code") + 1]
    # Backslash and quote both escaped — no raw " or unescaped \ from the path.
    assert '\\"' in code_arg
    assert "\\\\" in code_arg
    assert recorded["timeout"] == 17
