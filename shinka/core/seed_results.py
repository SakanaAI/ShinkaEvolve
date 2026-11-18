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
                await self._publish_results(self.path)
            except OSError as exc:
                logger.warning(
                    "Could not publish seed results from %s: %s", self.path, exc
                )
            raise
        artifact_dir = result[0].get("_artifact_dir", self.path)
        await self._publish_results(artifact_dir)
        for name in ("correct.json", "metrics.json"):
            if (artifact_dir / name).is_symlink() and not (
                self._public_results / name
            ).is_file():
                raise ValueError(f"{name} must refer to a file within the seed attempt")
        return result

    async def _publish_results(self, artifact_dir: Path) -> None:
        snapshot = self._history / f".publish-{self._id}"
        await asyncio.to_thread(self._copy_results, snapshot, artifact_dir)
        # Only the live runner may replace the public path.
        snapshot.rename(self._public_results)

    def _copy_results(self, snapshot: Path, artifact_dir: Path) -> None:
        # A Slurm allocation can outlive status verification. Keep its original
        # directory available and copy files independently of the producer.
        self.path.mkdir(exist_ok=True)
        root = artifact_dir.resolve()
        log_names = ("job_log.out", "job_log.err")

        def ignore(directory: str, names: list[str]) -> list[str]:
            source = Path(directory)
            return [
                name
                for name in names
                if not _can_snapshot(source / name, root)
                or (
                    artifact_dir != self.path
                    and source == artifact_dir
                    and name in log_names
                )
            ]

        shutil.copytree(artifact_dir, snapshot, ignore=ignore)
        if artifact_dir != self.path:
            log_root = self.path.resolve()
            for name in log_names:
                path = self.path / name
                if path.is_file() and _can_snapshot(path, log_root):
                    shutil.copy2(path, snapshot / name)


def _can_snapshot(path: Path, root: Path) -> bool:
    if not path.is_symlink():
        return path.is_file() or path.is_dir()
    try:
        target = path.resolve()
    except (OSError, RuntimeError):
        return False
    return target.is_relative_to(root) and target.is_file()
