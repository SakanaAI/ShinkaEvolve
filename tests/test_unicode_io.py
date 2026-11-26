import asyncio
import builtins
from pathlib import Path
import sys

import pandas as pd
import pytest

from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.edit import async_apply
from shinka.edit import summary as summary_module
from shinka.launch import local
from shinka.utils import general as general_module
from shinka.utils import load_df as load_df_module


class _FakeAsyncFile:
    def __init__(self, path: str, mode: str, encoding: str | None) -> None:
        self._path = path
        self._mode = mode
        self._encoding = encoding or "cp1252"
        self._handle = None

    async def __aenter__(self):
        self._handle = builtins.open(
            self._path,
            self._mode,
            encoding=self._encoding,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        if self._handle is not None:
            self._handle.close()

    async def read(self) -> str:
        assert self._handle is not None
        return self._handle.read()

    async def write(self, content: str) -> int:
        assert self._handle is not None
        return self._handle.write(content)


class _FakeAiofiles:
    @staticmethod
    def open(path: str, mode: str, encoding: str | None = None):
        return _FakeAsyncFile(path, mode, encoding)


def _open_with_cp1252_default(
    file,
    mode="r",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    closefd=True,
    opener=None,
):
    return builtins.open(
        file,
        mode,
        buffering=buffering,
        encoding=encoding or "cp1252",
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
    )


def test_read_file_async_reads_utf8_when_aiofiles_defaults_to_cp1252(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = 'print("smart quote ”")\n'
    file_path = tmp_path / "candidate.py"
    file_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(async_apply, "aiofiles", _FakeAiofiles)

    result = asyncio.run(async_apply.read_file_async(str(file_path)))

    assert result == content


def test_write_file_async_writes_utf8_when_aiofiles_defaults_to_cp1252(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = 'print("Han 漢")\n'
    file_path = tmp_path / "candidate.py"
    monkeypatch.setattr(async_apply, "aiofiles", _FakeAiofiles)

    written = asyncio.run(async_apply.write_file_async(str(file_path), content))

    assert written is True
    assert file_path.read_text(encoding="utf-8") == content


def test_read_file_async_fallback_uses_utf8_with_path_methods(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_read_text = Path.read_text
    content = 'print("smart quote ”")\n'
    file_path = tmp_path / "candidate.py"
    file_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(async_apply, "aiofiles", None)

    def read_text_with_cp1252_default(self, encoding=None, errors=None):
        return original_read_text(
            self,
            encoding=encoding or "cp1252",
            errors=errors,
        )

    monkeypatch.setattr(Path, "read_text", read_text_with_cp1252_default)

    result = asyncio.run(async_apply.read_file_async(str(file_path)))

    assert result == content


def test_write_file_async_fallback_uses_utf8_with_path_methods(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_write_text = Path.write_text
    content = 'print("Han 漢")\n'
    file_path = tmp_path / "candidate.py"
    monkeypatch.setattr(async_apply, "aiofiles", None)

    def write_text_with_cp1252_default(
        self,
        data,
        encoding=None,
        errors=None,
        newline=None,
    ):
        return original_write_text(
            self,
            data,
            encoding=encoding or "cp1252",
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", write_text_with_cp1252_default)

    written = asyncio.run(async_apply.write_file_async(str(file_path), content))

    assert written is True
    assert file_path.read_text(encoding="utf-8") == content


def test_validate_code_async_reads_utf8_for_other_languages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = 'print("smart quote ”")\n'
    file_path = tmp_path / "candidate.lua"
    file_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(async_apply, "aiofiles", _FakeAiofiles)

    is_valid, error = asyncio.run(
        async_apply.validate_code_async(str(file_path), language="lua")
    )

    assert is_valid is True
    assert error is None


def test_summarize_diff_reads_utf8_patch(monkeypatch, tmp_path: Path) -> None:
    diff_path = tmp_path / "candidate.patch"
    diff_path.write_text(
        """--- a/original.py
+++ b/original.py
@@ -1 +1 @@
-print("plain")
+print("smart quote ”")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(summary_module, "open", _open_with_cp1252_default, raising=False)

    summary = summary_module.summarize_diff(str(diff_path))

    assert summary == {"original.py": {"added": 0, "deleted": 0, "modified": 1}}


def test_store_best_path_writes_utf8_artifacts(monkeypatch, tmp_path: Path) -> None:
    original_write_text = Path.write_text

    def write_text_with_cp1252_default(
        self,
        data,
        encoding=None,
        errors=None,
        newline=None,
    ):
        return original_write_text(
            self,
            data,
            encoding=encoding or "cp1252",
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", write_text_with_cp1252_default)

    df = pd.DataFrame(
        [
            {
                "id": "prog-1",
                "parent_id": None,
                "generation": 1,
                "correct": True,
                "combined_score": 1.0,
                "code_diff": """--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-print("plain")
+print("Han 漢")
""",
                "code": 'print("Han 漢")\n',
                "patch_name": "unicode-patch",
            }
        ]
    )

    load_df_module.store_best_path(df, str(tmp_path))

    assert (tmp_path / "best_path" / "patches" / "patch_0.patch").read_text(
        encoding="utf-8"
    ).endswith('print("Han 漢")\n')
    assert (tmp_path / "best_path" / "code" / "main_0.py").read_text(
        encoding="utf-8"
    ) == 'print("Han 漢")\n'


def test_async_runner_read_file_async_uses_utf8(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = 'print("smart quote ”")\n'
    file_path = tmp_path / "candidate.py"
    file_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "shinka.core.async_runner.open",
        _open_with_cp1252_default,
        raising=False,
    )
    runner = object.__new__(ShinkaEvolveRunner)

    result = asyncio.run(runner._read_file_async(str(file_path)))

    assert result == content


def test_async_novelty_judge_reads_code_file_as_utf8(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = 'print("smart quote ”")\n'
    file_path = tmp_path / "candidate.py"
    file_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "shinka.core.async_novelty_judge.open",
        _open_with_cp1252_default,
        raising=False,
    )
    judge = AsyncNoveltyJudge(sync_novelty_judge=None)

    assert judge._read_code_file(str(file_path)) == content


@pytest.mark.parametrize(
    ("encoding", "stdout", "stderr"),
    [
        ("utf-8", "score → 0.5\n", 'ValueError: bad value "漢"\n'),
        ("cp1252", "score ± 0.5\n", "ValueError: bad value ”\n"),
    ],
    ids=["utf8", "legacy-locale"],
)
def test_load_result_logs_prefers_utf8_with_locale_fallback(
    monkeypatch,
    tmp_path: Path,
    encoding: str,
    stdout: str,
    stderr: str,
) -> None:
    original_read_text = Path.read_text
    (tmp_path / "job_log.out").write_text(stdout, encoding=encoding)
    (tmp_path / "job_log.err").write_text(stderr, encoding=encoding)

    def read_text_with_cp1252_default(self, encoding=None, errors=None):
        return original_read_text(
            self,
            encoding=encoding or "cp1252",
            errors=errors,
        )

    monkeypatch.setattr(Path, "read_text", read_text_with_cp1252_default)

    logs = general_module.load_result_logs(str(tmp_path))

    assert logs == {"stdout_log": stdout, "stderr_log": stderr}


def test_local_evaluator_captures_utf8_logs_under_cp1252(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_popen = local.subprocess.Popen

    def popen_with_cp1252_default(*args, **kwargs):
        if kwargs.get("text") and not kwargs.get("encoding"):
            kwargs["encoding"] = "cp1252"
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(local.subprocess, "Popen", popen_with_cp1252_default)
    monkeypatch.setattr(local, "open", _open_with_cp1252_default, raising=False)
    monkeypatch.setattr(local.locale, "getpreferredencoding", lambda _: "cp1252")

    cmd = [
        sys.executable,
        "-c",
        r"import sys; print('score \u2192 0.5'); "
        r"print('bad value \u201d\u6f22', file=sys.stderr); "
        r"sys.stderr.buffer.write(b'native tool \x94\n')",
    ]
    process = local.submit(str(tmp_path), cmd)
    logs = local.monitor(process, str(tmp_path))

    assert process.returncode == 0
    assert logs["stdout_log"] == "score → 0.5\n"
    assert logs["stderr_log"] == "bad value ”漢\nnative tool ”\n"


def test_local_evaluator_normalizes_stdio_override_to_utf8(tmp_path: Path) -> None:
    process = local.submit(
        str(tmp_path),
        [sys.executable, "-c", r"print('quote \u201d')"],
        env_overrides={"PYTHONIOENCODING": "cp1252:backslashreplace"},
    )

    logs = local.monitor(process, str(tmp_path))

    assert process.returncode == 0
    assert logs["stdout_log"] == "quote ”\n"
    assert (tmp_path / "job_log.out").read_text(encoding="utf-8") == "quote ”\n"
