"""Locate the program database of a run directory.

A run directory holds one program database, directly inside it: the prompt
database, the meta files and the analysis helpers all look next to it.
``DatabaseConfig.db_path`` chooses its file name, ``programs.sqlite`` by
default. A run never starts at generation 0 next to programs it did not ask
for: it continues in ``programs.sqlite`` or a first-release layout, and stops
when the programs sit under any other name.
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
    """The programs of a run directory sit under a name nobody asked for."""


def resolve_run_db_path(
    results_dir: Union[str, Path], configured_db_path: Optional[str]
) -> Path:
    """Return the database file a run in ``results_dir`` reads and writes.

    That is ``<results_dir>/<file name of db_path>``, or ``programs.sqlite``
    when ``db_path`` is unset. A directory part of ``db_path`` is not used. If
    that file has no programs but ``programs.sqlite`` or a first-release layout
    of the directory does, the run continues there and a warning names both.
    Programs under any other name are not adopted: the run stops and lists them.
    """
    results_dir = Path(results_dir)
    target = results_dir / _run_db_name(configured_db_path)
    known = _known_run_db_paths(results_dir, configured_db_path)
    existing = _first_with_programs(target, known)

    if existing is None:
        unknown = _unknown_program_dbs(results_dir, [target, *known])
        if unknown:
            raise AmbiguousRunDatabaseError(
                f"{results_dir} holds programs in {_names(unknown)}, but the run "
                f"is configured for {target.name}. To continue that run, set "
                "DatabaseConfig.db_path to its file name. To start a new run, "
                "use a new results_dir."
            )
    elif existing != target:
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
    directory that does (a first-release layout or a custom ``db_path``), else
    ``<results_dir>/<db_name>``. Reading changes nothing, so unlike a run a
    reader takes a database under an unknown name if it is the only one.
    """
    results_dir = Path(results_dir)
    target = results_dir / db_name
    known = _known_run_db_paths(results_dir, None)
    existing = _first_with_programs(target, known)
    if existing is not None:
        return existing

    unknown = _unknown_program_dbs(results_dir, [target, *known])
    if len(unknown) > 1:
        raise AmbiguousRunDatabaseError(
            f"{results_dir} holds several program databases ({_names(unknown)}) "
            f"and none is named {target.name}. Pass the path of the file to load."
        )
    return unknown[0] if unknown else target


def _names(paths: List[Path]) -> str:
    return ", ".join(path.name for path in paths)


def _run_db_name(configured_db_path: Optional[str]) -> str:
    if not configured_db_path:
        return DEFAULT_RUN_DB_NAME
    name = Path(configured_db_path).name
    if name in ("", ".."):
        raise ValueError(
            f"DatabaseConfig.db_path={configured_db_path!r} names no file. Give "
            "the file name of the run database, or leave it unset."
        )
    if _is_prompt_db_name(name):
        raise ValueError(
            f"DatabaseConfig.db_path={configured_db_path!r} is the name of the "
            "prompt database of a run. Choose another file name."
        )
    return name


def _is_prompt_db_name(name: str) -> bool:
    # Compared without case: on many file systems both spellings are one file.
    return name.lower() in _PROMPT_DB_NAMES


def _is_inside(configured_db_path: str, results_dir: Path) -> bool:
    """True for a bare file name or a path to a file directly in the directory."""
    parent = Path(configured_db_path).parent
    return parent == Path(".") or parent.resolve() == results_dir.resolve()


def _known_run_db_paths(
    results_dir: Path, configured_db_path: Optional[str]
) -> List[Path]:
    """Where a run of this or an earlier release keeps its programs."""
    known = [results_dir / DEFAULT_RUN_DB_NAME]
    if configured_db_path:
        known += _first_release_paths(results_dir, configured_db_path)
    # The preset name, and the file an unset db_path produced back then.
    return known + [results_dir / LEGACY_RUN_DB_NAME, results_dir / "None"]


def _first_with_programs(target: Path, known: List[Path]) -> Optional[Path]:
    """``target`` if it is in use, else the first known database with programs."""
    if _content_of(target) is not _Content.EMPTY:
        # It holds programs, or it cannot be read. Opening it reports that
        # error, which is safer than continuing in an older copy next to it.
        return target
    for known_path in known:
        if known_path != target and _has_programs(known_path):
            return known_path
    return None


def _unknown_program_dbs(results_dir: Path, named: List[Path]) -> List[Path]:
    """Databases with programs in the directory, other than the ``named`` ones."""
    return [
        path
        for path in _database_files(results_dir)
        if path not in named and _has_programs(path)
    ]


def _first_release_paths(results_dir: Path, configured_db_path: str) -> List[Path]:
    """The path the first releases built, ``f"{results_dir}/{db_path}"``.

    Empty when that path leaves the directory (``..``, a drive letter): a run
    must not continue in the database of another one.
    """
    path = results_dir / str(configured_db_path).lstrip("/\\")
    inside = path.resolve().is_relative_to(results_dir.resolve())
    return [path] if inside else []


def _database_files(results_dir: Path) -> List[Path]:
    """Files directly in the directory that are named like a database.

    A copy set aside under another suffix (``programs.sqlite.bak``) is not one.
    """
    if not results_dir.is_dir():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.suffix in _DB_SUFFIXES and not _is_prompt_db_name(path.name)
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
