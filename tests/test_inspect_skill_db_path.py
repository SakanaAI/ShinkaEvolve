"""Which database the shinka-inspect skill reads for a run directory."""

import importlib.util
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.run_db_path import AmbiguousRunDatabaseError

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "shinka-inspect"
    / "scripts"
    / "inspect_best_programs.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inspect_best_programs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclass looks the module up by name while the class is built.
    registered = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if registered is None:
            del sys.modules[spec.name]
        else:
            sys.modules[spec.name] = registered
    return module


@pytest.fixture(scope="module")
def inspect_script() -> ModuleType:
    return _load_script()


def _write_db(db_path: Path, num_programs: int = 1) -> None:
    config = DatabaseConfig(db_path=str(db_path), num_islands=1)
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        for generation in range(num_programs):
            db.add(
                Program(
                    id=f"p{generation}",
                    code="pass",
                    generation=generation,
                    correct=True,
                    combined_score=float(generation),
                )
            )


@pytest.mark.parametrize(
    "db_name", ["programs.sqlite", "custom.sqlite", "run.db", "evolution_db.sqlite"]
)
def test_run_directory_resolves_to_its_database(tmp_path, inspect_script, db_name):
    _write_db(tmp_path / db_name)

    assert inspect_script._resolve_db_path(str(tmp_path)) == tmp_path / db_name


def test_empty_default_database_does_not_hide_the_legacy_one(tmp_path, inspect_script):
    _write_db(tmp_path / "programs.sqlite", num_programs=0)
    _write_db(tmp_path / "evolution_db.sqlite")

    resolved = inspect_script._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / "evolution_db.sqlite"


def test_prompt_database_is_not_taken_for_the_run_database(tmp_path, inspect_script):
    # Sorts before the run database, so a first-match glob would pick it.
    (tmp_path / "prompts.sqlite").write_bytes(b"")
    _write_db(tmp_path / "run_programs.sqlite")

    resolved = inspect_script._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / "run_programs.sqlite"


def test_several_unknown_databases_are_not_guessed_between(tmp_path, inspect_script):
    _write_db(tmp_path / "a.sqlite")
    _write_db(tmp_path / "b.sqlite")

    with pytest.raises(AmbiguousRunDatabaseError, match="a.sqlite, b.sqlite"):
        inspect_script._resolve_db_path(str(tmp_path))


@pytest.mark.parametrize("db_name", ["other.sqlite", "other.db"])
def test_database_file_is_used_as_given(tmp_path, inspect_script, db_name):
    _write_db(tmp_path / "programs.sqlite")
    _write_db(tmp_path / db_name)

    resolved = inspect_script._resolve_db_path(str(tmp_path / db_name))

    assert resolved == tmp_path / db_name


def test_directory_without_database_resolves_to_the_default_name(
    tmp_path, inspect_script
):
    assert inspect_script._resolve_db_path(str(tmp_path)) == tmp_path / "programs.sqlite"


@pytest.fixture
def script_for_old_shinka(monkeypatch) -> ModuleType:
    # Releases up to 0.0.8 have no shinka.database.run_db_path.
    monkeypatch.setitem(sys.modules, "shinka.database.run_db_path", None)
    return _load_script()


def test_shinka_without_the_shared_resolver_defaults_to_the_default_name(
    tmp_path, script_for_old_shinka
):
    resolved = script_for_old_shinka._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / "programs.sqlite"


@pytest.mark.parametrize("db_name", ["custom.sqlite", "run.db"])
def test_shinka_without_the_shared_resolver_finds_the_only_database(
    tmp_path, script_for_old_shinka, db_name
):
    # A run directory written by a newer shinka with a custom db_path. Its
    # prompt database is not a candidate.
    (tmp_path / "prompts.sqlite").write_bytes(b"")
    _write_db(tmp_path / db_name)

    resolved = script_for_old_shinka._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / db_name


def test_shinka_without_the_shared_resolver_does_not_guess_between_databases(
    tmp_path, script_for_old_shinka
):
    _write_db(tmp_path / "a.sqlite")
    _write_db(tmp_path / "b.sqlite")

    with pytest.raises(ValueError, match="a.sqlite, b.sqlite"):
        script_for_old_shinka._resolve_db_path(str(tmp_path))


def test_shinka_without_the_shared_resolver_still_finds_the_first_release_name(
    tmp_path, script_for_old_shinka
):
    _write_db(tmp_path / "evolution_db.sqlite")

    resolved = script_for_old_shinka._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / "evolution_db.sqlite"

    _write_db(tmp_path / "programs.sqlite")

    resolved = script_for_old_shinka._resolve_db_path(str(tmp_path))

    assert resolved == tmp_path / "programs.sqlite"


def test_another_import_failure_is_not_taken_for_an_old_shinka(monkeypatch):
    # The resolver exists but one of its own imports is broken.
    monkeypatch.delitem(sys.modules, "shinka.database.run_db_path")
    monkeypatch.setitem(sys.modules, "sqlite3", None)

    with pytest.raises(ModuleNotFoundError, match="sqlite3"):
        _load_script()


def test_bundle_is_built_from_a_custom_named_database(
    tmp_path, inspect_script, monkeypatch, capsys
):
    _write_db(tmp_path / "run.db", num_programs=3)
    monkeypatch.setattr(
        sys, "argv", ["inspect_best_programs.py", "--results-dir", str(tmp_path)]
    )

    assert inspect_script.main() == 0

    bundle = (tmp_path / "shinka_inspect_context.md").read_text(encoding="utf-8")
    assert f"- Source DB: `{tmp_path / 'run.db'}`" in bundle
    assert "- Total rows loaded: `3`" in bundle
    assert "selected=3" in capsys.readouterr().out


def test_bundle_is_built_from_a_first_release_file_named_none(
    tmp_path, inspect_script, monkeypatch
):
    # load_programs_to_df takes a path without a database suffix for a directory.
    _write_db(tmp_path / "None", num_programs=2)
    monkeypatch.setattr(
        sys, "argv", ["inspect_best_programs.py", "--results-dir", str(tmp_path)]
    )

    assert inspect_script.main() == 0

    bundle = (tmp_path / "shinka_inspect_context.md").read_text(encoding="utf-8")
    assert f"- Source DB: `{tmp_path / 'None'}`" in bundle
    assert "- Total rows loaded: `2`" in bundle
