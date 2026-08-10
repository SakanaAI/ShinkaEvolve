import asyncio
import shutil
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


def test_validate_code_async_rust_delegates_to_rustc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rust validation must stay on stable-channel rustc flags.

    `-Zparse-only` is nightly-only, so a stable toolchain rejects the flag
    before parsing and reports every candidate invalid.
    """
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        return True, None

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.rs"), language="rust", timeout=23
        )
    )

    assert is_valid is True
    assert error is None
    assert recorded["timeout"] == 23

    args = recorded["args"]
    assert isinstance(args, tuple)
    assert args[0] == "rustc"
    assert args[-1] == str(tmp_path / "candidate.rs")
    assert "--edition" in args
    assert args[args.index("--edition") + 1] == "2021"
    assert "--crate-type=lib" in args
    assert "--emit=dep-info" in args
    assert "--out-dir" in args
    # No nightly-gated flag may be reintroduced: any `-Z` option makes stable
    # rustc exit before it parses the candidate.
    assert not any(arg.startswith("-Z") for arg in args)


def test_validate_code_async_rust_discards_dep_info_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """dep-info output goes to a temporary directory that is then removed.

    Without an explicit `--out-dir` rustc writes the dependency file into the
    working directory, which would leave artifacts beside the evolved program.
    """
    observed: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        out_dir = Path(args[args.index("--out-dir") + 1])
        observed["out_dir"] = out_dir
        observed["existed_during_call"] = out_dir.is_dir()
        return True, None

    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fake_helper)

    asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.rs"), language="rust", timeout=5
        )
    )

    out_dir = observed["out_dir"]
    assert isinstance(out_dir, Path)
    assert observed["existed_during_call"] is True
    assert out_dir.is_dir() is False
    assert out_dir != tmp_path
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is not installed")
def test_validate_code_async_rust_accepts_valid_program_on_real_rustc(
    tmp_path: Path,
) -> None:
    """End-to-end guard against the nightly-flag regression."""
    candidate = tmp_path / "candidate.rs"
    candidate.write_text(
        "pub fn collatz_steps(n: u64) -> u32 {\n"
        "    let mut steps = 0u32;\n"
        "    let mut value = n;\n"
        "    while value != 1 {\n"
        "        value = if value % 2 == 0 { value / 2 } else { 3 * value + 1 };\n"
        "        steps += 1;\n"
        "    }\n"
        "    steps\n"
        "}\n",
        encoding="utf-8",
    )

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is True, error
    assert error is None


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is not installed")
def test_validate_code_async_rust_rejects_broken_program_on_real_rustc(
    tmp_path: Path,
) -> None:
    """A syntax error must be reported as a rust error, not a flag error."""
    candidate = tmp_path / "candidate.rs"
    candidate.write_text("pub fn broken( -> { let mut\n", encoding="utf-8")

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is False
    assert error is not None
    assert "nightly" not in error
    assert "error" in error.lower()


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
