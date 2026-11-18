import errno
import os
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4


_ATTEMPT_PREFIX = ".results-attempt-"
_RESULTS_HISTORY = "results-attempts"


def run_seed_evaluation(
    evaluate: Callable[[str, str], tuple[dict[str, Any], float]],
    program_path: str,
    results_dir: str,
) -> tuple[dict[str, Any], float]:
    """Evaluate in a fresh directory and retain each attempt's output."""
    public_results = Path(results_dir)
    attempt_id = uuid4().hex
    attempt = public_results.with_name(f"{_ATTEMPT_PREFIX}{attempt_id}")
    attempt.mkdir()

    if public_results.exists() and not public_results.is_symlink():
        history = public_results.with_name(_RESULTS_HISTORY)
        history.mkdir(exist_ok=True)
        public_results.rename(history / f"previous-{attempt_id}")

    if os.name != "nt" and _link_seed_results(attempt, public_results):
        return evaluate(program_path, str(attempt))

    # Windows can require elevated privileges, and some filesystems lack links.
    if public_results.is_symlink():
        public_results.unlink()
    try:
        return evaluate(program_path, str(attempt))
    finally:
        shutil.copytree(attempt, public_results)


def _link_seed_results(attempt: Path, public_results: Path) -> bool:
    # A relative link survives archival and keeps the evaluator's path stable.
    link = public_results.with_name(f"{attempt.name}.link")
    try:
        link.symlink_to(attempt.name, target_is_directory=True)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP}:
            raise
        return False
    try:
        link.replace(public_results)
    finally:
        link.unlink(missing_ok=True)
    return True


def copy_seed_generation(source: Path, destination: Path) -> None:
    """Export the current seed artifacts without earlier attempts."""

    def ignore(directory: str, names: list[str]) -> list[str]:
        if Path(directory) != source:
            return []
        return [
            name
            for name in names
            if name == _RESULTS_HISTORY or name.startswith(_ATTEMPT_PREFIX)
        ]

    shutil.copytree(source, destination, ignore=ignore)
