import asyncio
import logging
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4


logger = logging.getLogger(__name__)


class SeedResultAttempt:
    """Own the evaluator's destination and publish a snapshot for readers."""

    def __init__(self, results_dir: str) -> None:
        self._public_results = Path(results_dir)
        self._id = uuid4().hex
        self._history = self._public_results.parent.parent / ".seed-results-attempts"
        self.path = self._history / f"attempt-{self._id}"

    async def run(
        self,
        evaluate: Callable[[str, str], tuple[dict[str, Any], float]],
        program_path: str,
    ) -> tuple[dict[str, Any], float]:
        self.path.mkdir(parents=True)
        if self._public_results.exists() or self._public_results.is_symlink():
            self._public_results.rename(self._history / f"previous-{self._id}")

        try:
            result = await asyncio.to_thread(evaluate, program_path, str(self.path))
        except Exception:
            try:
                await self._publish_results()
            except OSError as exc:
                logger.warning(
                    "Could not publish seed results from %s: %s", self.path, exc
                )
            raise
        await self._publish_results()
        for name in ("correct.json", "metrics.json"):
            if (self.path / name).is_symlink() and not (
                self._public_results / name
            ).is_file():
                raise ValueError(f"{name} must refer to a file within the seed attempt")
        return result

    async def _publish_results(self) -> None:
        snapshot = self._history / f".publish-{self._id}"
        await asyncio.to_thread(self._copy_results, snapshot)
        # Only the live runner may replace the public path.
        snapshot.rename(self._public_results)

    def _copy_results(self, snapshot: Path) -> None:
        # A Slurm allocation can outlive status verification. Keep its original
        # directory available and copy files independently of the producer.
        self.path.mkdir(exist_ok=True)
        root = self.path.resolve()

        def ignore(directory: str, names: list[str]) -> list[str]:
            return [
                name
                for name in names
                if not _can_snapshot(Path(directory) / name, root)
            ]

        shutil.copytree(self.path, snapshot, ignore=ignore)


def _can_snapshot(path: Path, root: Path) -> bool:
    if not path.is_symlink():
        return path.is_file() or path.is_dir()
    try:
        target = path.resolve()
    except (OSError, RuntimeError):
        return False
    return target.is_relative_to(root) and target.is_file()
