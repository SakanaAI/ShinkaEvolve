"""Locate the program database of a run directory.

A run directory holds one program database, directly inside it: the prompt
database, the meta files and the analysis helpers all look next to it.
``DatabaseConfig.db_path`` chooses its file name, ``programs.sqlite`` by
default. Programs a directory already holds under another name win over the
configured name, so an edited configuration cannot restart a run at generation 0.
"""

import logging
import sqlite3
from contextlib import closing
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_RUN_DB_NAME = "programs.sqlite"
# The name the shipped presets and examples configured in the first releases.
LEGACY_RUN_DB_NAME = "evolution_db.sqlite"

# How the web UI and load_programs_to_df tell a database file from other files.
_DB_SUFFIXES = (".sqlite", ".db")
_PROMPT_DB_NAMES = ("prompts.sqlite", "prompts.db")


class AmbiguousRunDatabaseError(ValueError):
    """Several unrelated program databases could be the one of a run."""


def resolve_run_db_path(
    results_dir: Union[str, Path], configured_db_path: Optional[str]
) -> Path:
    """Return the database file a run in ``results_dir`` reads and writes.

    That is ``<results_dir>/<file name of db_path>``, or ``programs.sqlite``
    when ``db_path`` is unset. A directory part of ``db_path`` is not used. If
    that file has no programs but another database of the directory does, the
    run continues in the existing one and a warning names both.
    """
    results_dir = Path(results_dir)
    target = results_dir / _run_db_name(configured_db_path)
    existing = _existing_run_db(results_dir, target, configured_db_path)

    if existing is not None and existing != target:
        logger.warning(
            "%s already holds the programs of this run. Continuing in it instead "
            "of %s. Use a new results_dir to start a new run.",
            existing,
            target.name,
        )
        return existing

    if configured_db_path and not _is_inside(configured_db_path, results_dir):
        logger.warning(
            "Only the file name of DatabaseConfig.db_path=%r is used: the run "
            "stores its programs in %s. Set results_dir to move a run.",
            configured_db_path,
            target,
        )
    if target.suffix not in _DB_SUFFIXES:
        logger.warning(
            "DatabaseConfig.db_path=%r does not end in .sqlite or .db. The web UI "
            "and load_programs_to_df will not recognise %s as a database, and a "
            "resume without this db_path will not find it.",
            configured_db_path,
            target.name,
        )
    return target


def find_run_db_path(
    results_dir: Union[str, Path], db_name: str = DEFAULT_RUN_DB_NAME
) -> Path:
    """Return the database file to read for the run in ``results_dir``.

    ``<results_dir>/<db_name>`` if it has programs, else the database of the
    directory that does (a custom ``db_path`` or a first-release layout), else
    ``<results_dir>/<db_name>``.
    """
    results_dir = Path(results_dir)
    target = results_dir / db_name
    return _existing_run_db(results_dir, target, None) or target


def _run_db_name(configured_db_path: Optional[str]) -> str:
    if not configured_db_path:
        return DEFAULT_RUN_DB_NAME
    return Path(configured_db_path).name or DEFAULT_RUN_DB_NAME


def _is_inside(configured_db_path: str, results_dir: Path) -> bool:
    """True for a bare file name or a path to a file directly in the directory."""
    parent = Path(configured_db_path).parent
    return parent == Path(".") or parent.resolve() == results_dir.resolve()


def _existing_run_db(
    results_dir: Path, target: Path, configured_db_path: Optional[str]
) -> Optional[Path]:
    """The database of ``results_dir`` that holds programs, ``target`` first."""
    if _content_of(target) is not _Content.EMPTY:
        # It holds programs, or it cannot be read. Opening it reports that
        # error, which is safer than continuing in an older copy next to it.
        return target

    known = [results_dir / DEFAULT_RUN_DB_NAME]
    if configured_db_path:
        # As the first releases built it: f"{results_dir}/{db_path}"
        known.append(results_dir / str(configured_db_path).lstrip("/\\"))
    # The preset name, and the file an unset db_path produced back then.
    known += [results_dir / LEGACY_RUN_DB_NAME, results_dir / "None"]
    for known_path in known:
        if known_path != target and _has_programs(known_path):
            return known_path

    # Any other name: the run was started with a db_path the caller no longer
    # passes.
    others = [
        path
        for path in _database_files(results_dir)
        if path not in known and path != target and _has_programs(path)
    ]
    if len(others) > 1:
        raise AmbiguousRunDatabaseError(
            f"{results_dir} holds several program databases "
            f"({', '.join(path.name for path in others)}) and none is named "
            f"{target.name}. Name the one to use: set DatabaseConfig.db_path "
            "for a run, or pass the file path when loading."
        )
    return others[0] if others else None


def _database_files(results_dir: Path) -> List[Path]:
    """Files directly in the directory that are named like a database.

    A copy set aside under another suffix (``programs.sqlite.bak``) is not one.
    """
    if not results_dir.is_dir():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.suffix in _DB_SUFFIXES and path.name not in _PROMPT_DB_NAMES
    )


class _Content(Enum):
    """What a file holds, seen as the program database of a run."""

    PROGRAMS = "programs"
    # No file, or a database without a program in it.
    EMPTY = "empty"
    # Damaged, locked, not permitted, or not a program database.
    UNREADABLE = "unreadable"


def _has_programs(db_path: Path) -> bool:
    """True if the file is a program database with at least one program."""
    return _content_of(db_path) is _Content.PROGRAMS


def _content_of(db_path: Path) -> _Content:
    if not db_path.is_file():
        return _Content.EMPTY
    try:
        with closing(
            sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        ) as conn:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'programs'"
            ).fetchone()
            if has_table is None:
                return _Content.EMPTY
            program = conn.execute(
                "SELECT id, generation FROM programs LIMIT 1"
            ).fetchone()
    except sqlite3.Error:
        return _Content.UNREADABLE
    return _Content.PROGRAMS if program else _Content.EMPTY
