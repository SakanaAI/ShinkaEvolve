import importlib
import os
import sys
from types import ModuleType
from pathlib import Path

import pytest


def _clear_shinka_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "shinka" or module_name.startswith("shinka."):
            sys.modules.pop(module_name, None)


def _snapshot_shinka_modules() -> dict[str, ModuleType]:
    return {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "shinka" or module_name.startswith("shinka.")
    }


def _restore_shinka_modules(snapshot: dict[str, ModuleType]) -> None:
    _clear_shinka_modules()
    sys.modules.update(snapshot)


def test_import_shinka_loads_dotenv_from_launch_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_key = "SHINKA_TEST_IMPORT_KEY"
    monkeypatch.delenv(env_key, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{env_key}=from-launch-dir\n", encoding="utf-8")

    module_snapshot = _snapshot_shinka_modules()
    try:
        _clear_shinka_modules()
        importlib.import_module("shinka")

        assert os.getenv(env_key) == "from-launch-dir"
    finally:
        _restore_shinka_modules(module_snapshot)


def test_load_shinka_dotenv_prefers_launch_directory_over_package_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shinka.env import load_shinka_dotenv

    env_key = "SHINKA_TEST_PRIORITY_KEY"
    package_root = tmp_path / "package-root"
    launch_dir = tmp_path / "launch-dir"
    package_root.mkdir()
    launch_dir.mkdir()
    (package_root / ".env").write_text(f"{env_key}=from-package\n", encoding="utf-8")
    (launch_dir / ".env").write_text(f"{env_key}=from-launch-dir\n", encoding="utf-8")
    monkeypatch.delenv(env_key, raising=False)

    load_shinka_dotenv(package_root=package_root, cwd=launch_dir)

    assert os.getenv(env_key) == "from-launch-dir"
