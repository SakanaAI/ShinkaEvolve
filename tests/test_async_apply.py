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
        self.communicate_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
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
    assert recorded["stdout"] == asyncio.subprocess.DEVNULL
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
    assert proc.communicate_calls == 1


def test_run_validation_subprocess_cancellation_kills_process(
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
        del awaitable, timeout
        raise asyncio.CancelledError

    monkeypatch.setattr(
        async_apply.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(async_apply.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            async_apply._run_validation_subprocess(
                "rustfmt", "candidate.rs", timeout=3
            )
        )

    assert proc.kill_called is True
    assert proc.communicate_calls == 1


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


def test_validate_code_async_rust_delegates_to_rustfmt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rust validation must use stable, non-mutating rustfmt flags.

    `-Zparse-only` is nightly-only, so a stable toolchain rejects the flag
    before parsing and reports every candidate invalid.
    """
    recorded: dict[str, object] = {}

    async def fake_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        recorded["args"] = args
        recorded["timeout"] = timeout
        config_path = Path(args[args.index("--config-path") + 1])
        recorded["config_path"] = config_path
        recorded["config_existed"] = config_path.is_file()
        recorded["config_contents"] = config_path.read_text(encoding="utf-8")
        return True, None

    monkeypatch.setattr(async_apply.shutil, "which", lambda _name: "/usr/bin/rustfmt")
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
    assert args[0] == "rustfmt"
    assert args[-2] == "--"
    assert args[-1] == str(tmp_path / "candidate.rs")
    assert "--config-path" in args
    assert "--emit" in args
    assert args[args.index("--emit") + 1] == "stdout"
    assert "--edition" in args
    assert args[args.index("--edition") + 1] == "2015"
    assert "--config" in args
    assert args[args.index("--config") + 1] == "skip_children=true"
    # No nightly-gated flag may be reintroduced: stable Rust tooling rejects
    # `-Z` options before it parses the candidate.
    assert not any(arg.startswith("-Z") for arg in args)
    assert recorded["config_existed"] is True
    assert recorded["config_contents"] == ""
    config_path = recorded["config_path"]
    assert isinstance(config_path, Path)
    assert config_path.exists() is False


def test_validate_code_async_rust_reports_missing_rustfmt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A minimal Rust toolchain gets an actionable dependency error."""

    async def fail_helper(*args: str, timeout: int) -> tuple[bool, str | None]:
        raise AssertionError(f"unexpected validator invocation: {args}")

    monkeypatch.setattr(async_apply.shutil, "which", lambda _name: None)
    monkeypatch.setattr(async_apply, "_run_validation_subprocess", fail_helper)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(tmp_path / "candidate.rs"), language="rust", timeout=5
        )
    )

    assert is_valid is False
    assert error == (
        "Rust validation requires rustfmt on PATH. Install it with "
        "`rustup component add rustfmt`."
    )


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_does_not_read_host_files(
    tmp_path: Path,
) -> None:
    """Syntax validation must not read files through macros or modules."""
    secret = tmp_path / "secret.txt"
    secret.write_text("host-secret-must-not-leak !!!", encoding="utf-8")
    candidate = tmp_path / "candidate.rs"
    candidate.write_text(
        f'#[path = r"{secret}"]\n'
        "mod secret;\n"
        f'compile_error!(include_str!(r"{secret}"));\n',
        encoding="utf-8",
    )

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is True, error
    assert error is None
    assert candidate.read_text(encoding="utf-8") == (
        f'#[path = r"{secret}"]\n'
        "mod secret;\n"
        f'compile_error!(include_str!(r"{secret}"));\n'
    )


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_accepts_valid_program_on_real_rustfmt(
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


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_rejects_broken_program_on_real_rustfmt(
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


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_rejects_option_shaped_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    candidate = Path("--version")
    candidate.write_text("pub fn broken( -> {\n", encoding="utf-8")

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is False
    assert error is not None
    assert "error" in error.lower()


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_ignores_project_config_that_disables_parsing(
    tmp_path: Path,
) -> None:
    """An ambient rustfmt.toml cannot make malformed Rust pass validation."""
    (tmp_path / "rustfmt.toml").write_text(
        "disable_all_formatting = true\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.rs"
    candidate.write_text("pub fn broken( -> {\n", encoding="utf-8")

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is False
    assert error is not None
    assert "error" in error.lower()


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_accepts_type_errors(
    tmp_path: Path,
) -> None:
    """Validation remains syntax-only and does not run the type checker."""
    candidate = tmp_path / "candidate.rs"
    candidate.write_text(
        'pub fn wrong_type() -> u32 { "not a number" }\n',
        encoding="utf-8",
    )

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(candidate), language="rust", timeout=60)
    )

    assert is_valid is True, error
    assert error is None


@pytest.mark.skipif(shutil.which("rustfmt") is None, reason="rustfmt is not installed")
def test_validate_code_async_rust_honors_configured_edition(
    tmp_path: Path,
) -> None:
    """Projects can override the default for edition-sensitive syntax."""
    candidate = tmp_path / "candidate.rs"
    candidate.write_text("pub async fn modern() {}\n", encoding="utf-8")

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(
            str(candidate),
            language="rust",
            timeout=60,
            rust_edition="2021",
        )
    )

    assert is_valid is True, error
    assert error is None


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
