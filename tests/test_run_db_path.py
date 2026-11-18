"""Which database file a run directory reads and writes."""

import logging
from contextlib import closing
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.run_db_path import resolve_run_db_path


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
                    combined_score=1.0,
                )
            )


@pytest.mark.parametrize(
    "configured",
    [None, "programs.sqlite", "evolution_db.sqlite", "custom.sqlite", "/abs/my.db"],
)
def test_new_run_always_uses_programs_sqlite(tmp_path, configured):
    assert resolve_run_db_path(tmp_path, configured) == tmp_path / "programs.sqlite"


@pytest.mark.parametrize("configured", [None, "programs.sqlite"])
def test_default_configuration_does_not_warn(tmp_path, caplog, configured):
    with caplog.at_level(logging.WARNING):
        resolve_run_db_path(tmp_path, configured)
        resolve_run_db_path(tmp_path, str(tmp_path / "programs.sqlite"))

    assert caplog.records == []


def test_custom_db_path_is_reported_as_ignored(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        resolve_run_db_path(tmp_path, "custom.sqlite")

    (record,) = caplog.records
    assert "'custom.sqlite' is ignored" in record.getMessage()
    assert str(tmp_path / "programs.sqlite") in record.getMessage()


@pytest.mark.parametrize(
    ("legacy_name", "configured"),
    [
        ("evolution_db.sqlite", "evolution_db.sqlite"),
        # The preset value was dropped from the config since the run started.
        ("evolution_db.sqlite", None),
        ("custom.sqlite", "custom.sqlite"),
        ("nested/custom.sqlite", "nested/custom.sqlite"),
    ],
)
def test_legacy_run_directory_keeps_its_database(tmp_path, legacy_name, configured):
    _write_db(tmp_path / legacy_name)

    assert resolve_run_db_path(tmp_path, configured) == tmp_path / legacy_name


def test_database_file_outside_the_configuration_is_not_adopted(tmp_path):
    _write_db(tmp_path / "other.sqlite")

    resolved = resolve_run_db_path(tmp_path, "custom.sqlite")

    assert resolved == tmp_path / "programs.sqlite"


def test_populated_programs_sqlite_wins_over_a_legacy_database(tmp_path):
    _write_db(tmp_path / "evolution_db.sqlite", num_programs=3)
    _write_db(tmp_path / "programs.sqlite")

    resolved = resolve_run_db_path(tmp_path, "evolution_db.sqlite")

    assert resolved == tmp_path / "programs.sqlite"


@pytest.mark.parametrize("leftover", ["zero-bytes", "tables-only", "not-a-database"])
def test_empty_programs_sqlite_does_not_hide_the_legacy_database(tmp_path, leftover):
    """An earlier resume attempt may have left an empty programs.sqlite."""
    _write_db(tmp_path / "evolution_db.sqlite")
    default_path = tmp_path / "programs.sqlite"
    if leftover == "zero-bytes":
        default_path.touch()
    elif leftover == "tables-only":
        _write_db(default_path, num_programs=0)
    else:
        default_path.write_text("not sqlite")

    resolved = resolve_run_db_path(tmp_path, "evolution_db.sqlite")

    assert resolved == tmp_path / "evolution_db.sqlite"
