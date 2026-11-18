"""Which database file a run directory reads and writes."""

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from shinka.configs import config_root
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.run_db_path import (
    AmbiguousRunDatabaseError,
    find_run_db_path,
    resolve_run_db_path,
)
from shinka.utils import load_programs_to_df


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


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


# -- new runs -----------------------------------------------------------------


@pytest.mark.parametrize("configured", [None, "", "programs.sqlite"])
def test_new_run_without_db_path_uses_programs_sqlite(tmp_path, caplog, configured):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, configured)

    assert resolved == tmp_path / "programs.sqlite"
    assert caplog.records == []


@pytest.mark.parametrize("configured", ["custom.sqlite", "evolution_db.sqlite", "my.db"])
def test_new_run_uses_the_configured_file_name(tmp_path, caplog, configured):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, configured)

    assert resolved == tmp_path / configured
    assert caplog.records == []


@pytest.mark.parametrize("configured", ["/abs/my.db", "nested/my.db", "../my.db"])
def test_directory_part_of_db_path_is_not_used(tmp_path, caplog, configured):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, configured)

    assert resolved == tmp_path / "my.db"
    (message,) = _warnings(caplog)
    assert "Only the file name" in message
    assert str(tmp_path / "my.db") in message


def test_path_into_the_results_dir_is_taken_as_is(tmp_path, caplog):
    """The runner writes the resolved path back into the config it was given."""
    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, str(tmp_path / "custom.sqlite"))

    assert resolved == tmp_path / "custom.sqlite"
    assert caplog.records == []


# -- resuming -----------------------------------------------------------------


@pytest.mark.parametrize("configured", [None, "programs.sqlite", "custom.sqlite"])
def test_resume_continues_in_the_configured_file(tmp_path, caplog, configured):
    name = configured or "programs.sqlite"
    _write_db(tmp_path / name)

    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, configured)

    assert resolved == tmp_path / name
    assert caplog.records == []


@pytest.mark.parametrize(
    ("existing_name", "configured"),
    [
        # db_path was added, changed or removed since the run started.
        ("programs.sqlite", "custom.sqlite"),
        ("first.sqlite", "second.sqlite"),
        ("custom.sqlite", None),
        ("custom.db", None),
        # The first releases built the path as f"{results_dir}/{db_path}".
        ("evolution_db.sqlite", None),
        ("nested/custom.sqlite", "nested/custom.sqlite"),
        ("abs/custom.sqlite", "/abs/custom.sqlite"),
        ("None", None),
    ],
)
def test_existing_programs_win_over_the_configured_name(
    tmp_path, caplog, existing_name, configured
):
    (tmp_path / existing_name).parent.mkdir(parents=True, exist_ok=True)
    _write_db(tmp_path / existing_name)

    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, configured)

    assert resolved == tmp_path / existing_name
    (message,) = _warnings(caplog)
    assert f"{tmp_path / existing_name} already holds the programs" in message


def test_configured_file_with_programs_wins_over_other_databases(tmp_path, caplog):
    _write_db(tmp_path / "programs.sqlite", num_programs=3)
    _write_db(tmp_path / "evolution_db.sqlite", num_programs=3)
    _write_db(tmp_path / "custom.sqlite")

    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, "custom.sqlite")

    assert resolved == tmp_path / "custom.sqlite"
    assert caplog.records == []


def test_programs_sqlite_wins_over_other_existing_databases(tmp_path):
    _write_db(tmp_path / "backup.sqlite", num_programs=3)
    _write_db(tmp_path / "evolution_db.sqlite", num_programs=3)
    _write_db(tmp_path / "programs.sqlite")

    resolved = resolve_run_db_path(tmp_path, "custom.sqlite")

    assert resolved == tmp_path / "programs.sqlite"


@pytest.mark.parametrize("leftover", ["zero-bytes", "tables-only"])
def test_file_without_programs_does_not_hide_the_existing_database(tmp_path, leftover):
    """An earlier resume attempt may have left an empty programs.sqlite."""
    _write_db(tmp_path / "evolution_db.sqlite")
    default_path = tmp_path / "programs.sqlite"
    if leftover == "zero-bytes":
        default_path.touch()
    else:
        _write_db(default_path, num_programs=0)

    assert resolve_run_db_path(tmp_path, None) == tmp_path / "evolution_db.sqlite"


@pytest.mark.parametrize("damage", ["not-a-database", "corrupt", "foreign-schema"])
def test_unreadable_database_is_not_swapped_for_another_one(tmp_path, caplog, damage):
    """A damaged run database must fail on open, not resume from an older copy."""
    _write_db(tmp_path / "evolution_db.sqlite")
    _write_db(tmp_path / "backup.sqlite")
    default_path = tmp_path / "programs.sqlite"
    if damage == "not-a-database":
        default_path.write_text("not sqlite")
    elif damage == "corrupt":
        _write_db(default_path)
        with open(default_path, "r+b") as db_file:
            db_file.write(b"\0" * 200)
    else:
        with closing(sqlite3.connect(default_path)) as conn:
            conn.execute("CREATE TABLE programs (title TEXT)")
            conn.execute("INSERT INTO programs VALUES ('news')")
            conn.commit()

    with caplog.at_level(logging.WARNING):
        assert resolve_run_db_path(tmp_path, None) == default_path
        assert find_run_db_path(tmp_path) == default_path
    assert caplog.records == []


def test_several_unnamed_databases_are_not_guessed_between(tmp_path):
    _write_db(tmp_path / "a.sqlite")
    _write_db(tmp_path / "b.sqlite")

    with pytest.raises(AmbiguousRunDatabaseError, match="a.sqlite, b.sqlite"):
        resolve_run_db_path(tmp_path, None)

    assert resolve_run_db_path(tmp_path, "b.sqlite") == tmp_path / "b.sqlite"


def test_prompt_database_and_other_files_are_not_run_databases(tmp_path):
    with closing(ProgramDatabase(DatabaseConfig(db_path=str(tmp_path / "x.sqlite")))):
        pass  # tables only
    (tmp_path / "prompts.sqlite").write_bytes((tmp_path / "x.sqlite").read_bytes())
    (tmp_path / "notes.db").write_text("not sqlite")
    (tmp_path / "gen_0").mkdir()

    assert resolve_run_db_path(tmp_path, None) == tmp_path / "programs.sqlite"


def test_database_set_aside_under_another_suffix_is_not_resumed(tmp_path):
    _write_db(tmp_path / "programs.sqlite")
    (tmp_path / "programs.sqlite").rename(tmp_path / "programs.sqlite.bak")

    assert resolve_run_db_path(tmp_path, None) == tmp_path / "programs.sqlite"


def test_db_path_without_database_suffix_is_honoured_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        resolved = resolve_run_db_path(tmp_path, "mydb")

    assert resolved == tmp_path / "mydb"
    (message,) = _warnings(caplog)
    assert "does not end in .sqlite or .db" in message

    _write_db(tmp_path / "mydb")
    assert resolve_run_db_path(tmp_path, "mydb") == tmp_path / "mydb"


# -- readers ------------------------------------------------------------------


def test_reader_finds_the_database_of_a_run_directory(tmp_path):
    assert find_run_db_path(tmp_path) == tmp_path / "programs.sqlite"

    _write_db(tmp_path / "custom.sqlite")
    assert find_run_db_path(tmp_path) == tmp_path / "custom.sqlite"

    _write_db(tmp_path / "programs.sqlite")
    assert find_run_db_path(tmp_path) == tmp_path / "programs.sqlite"
    assert find_run_db_path(tmp_path, "custom.sqlite") == tmp_path / "custom.sqlite"


@pytest.mark.parametrize("db_name", ["programs.sqlite", "custom.sqlite", "run.db"])
def test_load_programs_to_df_reads_a_run_directory(tmp_path, db_name):
    _write_db(tmp_path / db_name, num_programs=3)

    frame = load_programs_to_df(str(tmp_path), verbose=False)

    assert frame is not None
    assert sorted(frame["id"]) == ["p0", "p1", "p2"]


def test_load_programs_to_df_reads_the_file_it_is_given(tmp_path):
    _write_db(tmp_path / "programs.sqlite", num_programs=3)
    _write_db(tmp_path / "other.sqlite", num_programs=1)

    frame = load_programs_to_df(str(tmp_path / "other.sqlite"), verbose=False)

    assert frame is not None
    assert list(frame["id"]) == ["p0"]


def test_load_programs_to_df_reports_a_directory_without_database(tmp_path, capsys):
    assert load_programs_to_df(str(tmp_path), verbose=False) is None
    assert "Database file not found" in capsys.readouterr().out


# -- Hydra presets ------------------------------------------------------------


@pytest.mark.parametrize("preset", ["island_small", "island_medium", "island_large"])
def test_hydra_presets_accept_a_db_path_override(preset):
    with config_root() as configs:
        with initialize_config_dir(version_base=None, config_dir=str(configs)):
            default = compose(config_name="config", overrides=[f"database@_global_={preset}"])
            custom = compose(
                config_name="config",
                overrides=[f"database@_global_={preset}", "db_config.db_path=custom.sqlite"],
            )

    assert default.db_config.db_path is None
    assert custom.db_config.db_path == "custom.sqlite"
