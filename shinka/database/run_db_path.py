"""Locate the program database of a run directory."""

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_RUN_DB_NAME = "programs.sqlite"
# The name the shipped presets and examples configured while the runner still
# stored the database under ``DatabaseConfig.db_path``.
LEGACY_RUN_DB_NAME = "evolution_db.sqlite"


def resolve_run_db_path(
    results_dir: Union[str, Path], configured_db_path: Optional[str]
) -> Path:
    """Return the database file a run in ``results_dir`` reads and writes.

    Runs store their programs in ``programs.sqlite``. The first releases wrote
    ``<results_dir>/<db_path>`` instead, so a run directory from that time is
    continued in the file it already has: opening a new, empty
    ``programs.sqlite`` next to it would restart the run at generation 0.
    """
    results_dir = Path(results_dir)
    default_path = results_dir / DEFAULT_RUN_DB_NAME
    configured = Path(configured_db_path) if configured_db_path else None
    if configured is not None and configured.name == DEFAULT_RUN_DB_NAME:
        configured = None

    if not _has_programs(default_path):
        candidates = []
        if configured is not None:
            # As the first releases built it: f"{results_dir}/{db_path}"
            candidates.append(results_dir / str(configured).lstrip("/\\"))
            candidates.append(results_dir / configured.name)
        # The preset name, and the file an unset db_path produced back then.
        candidates += [results_dir / LEGACY_RUN_DB_NAME, results_dir / "None"]
        for legacy_path in candidates:
            if _has_programs(legacy_path):
                logger.warning(
                    "Resuming the run in its existing database %s. New runs "
                    "store their programs in %s.",
                    legacy_path,
                    DEFAULT_RUN_DB_NAME,
                )
                return legacy_path

    if configured is not None:
        logger.warning(
            "DatabaseConfig.db_path=%r is ignored: the run stores its programs "
            "in %s. Remove db_path from the run configuration.",
            configured_db_path,
            default_path,
        )
    return default_path


def _has_programs(db_path: Path) -> bool:
    """True if the file is a program database with at least one program."""
    if not db_path.is_file():
        return False
    try:
        with closing(
            sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        ) as conn:
            return conn.execute("SELECT 1 FROM programs LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False
