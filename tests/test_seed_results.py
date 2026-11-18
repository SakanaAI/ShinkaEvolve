import os
from pathlib import Path

import pytest

from shinka.core.seed_results import SeedResultAttempt
from shinka.utils.utils_hydra import load_hydra_config


@pytest.mark.parametrize("first_fails", [False, True])
def test_seed_publishes_snapshot_and_isolates_late_output(
    tmp_path: Path, first_fails: bool
) -> None:
    hydra = tmp_path / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text("task: seed")
    gen_dir = tmp_path / "gen_0"
    gen_dir.mkdir()
    public_results = gen_dir / "results"
    attempts: list[Path] = []

    def evaluate(_program, results):
        attempt = Path(results)
        attempts.append(attempt)
        config = load_hydra_config(results)
        assert config is not None and config.task == "seed"
        (attempt / "job_log.out").write_text(f"attempt {len(attempts)}")
        if first_fails and len(attempts) == 1:
            raise RuntimeError("status verification failed")
        return {}, 0.0

    if first_fails:
        with pytest.raises(RuntimeError, match="status verification failed"):
            SeedResultAttempt(str(public_results)).run(evaluate, "main.py")
    else:
        SeedResultAttempt(str(public_results)).run(evaluate, "main.py")

    public_log = public_results / "job_log.out"
    assert public_log.read_text() == "attempt 1"
    (attempts[0] / "job_log.out").write_text("late output")
    assert public_log.read_text() == "attempt 1"

    SeedResultAttempt(str(public_results)).run(evaluate, "main.py")

    first, second = attempts
    assert first != second
    assert first.parent == second.parent == tmp_path / ".seed-results-attempts"
    assert (first / "job_log.out").read_text() == "late output"
    assert public_log.read_text() == "attempt 2"
    assert sorted(path.name for path in gen_dir.iterdir()) == ["results"]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX links and FIFOs")
def test_seed_snapshot_is_independent_of_owned_links_and_special_files(
    tmp_path: Path,
) -> None:
    gen_dir = tmp_path / "gen_0"
    gen_dir.mkdir()
    public_results = gen_dir / "results"
    attempt = SeedResultAttempt(str(public_results))
    external_file = tmp_path / "external"
    external_file.write_text("external")

    def evaluate(_program, results):
        root = Path(results)
        (root / "log-target").write_text("partial stderr")
        (root / "job_log.err").symlink_to(root / "log-target")
        (root / "external-link").symlink_to(external_file)
        os.mkfifo(root / "stream")
        raise RuntimeError("status verification failed")

    with pytest.raises(RuntimeError, match="status verification failed"):
        attempt.run(evaluate, "main.py")

    (attempt.path / "log-target").write_text("late stderr")
    assert (public_results / "job_log.err").read_text() == "partial stderr"
    assert not (public_results / "job_log.err").is_symlink()
    assert sorted(path.name for path in public_results.iterdir()) == [
        "job_log.err",
        "log-target",
    ]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
def test_seed_cannot_accept_results_linked_to_an_older_attempt(tmp_path: Path) -> None:
    older_metrics = tmp_path / "older-metrics.json"
    older_metrics.write_text('{"combined_score": 42}')
    gen_dir = tmp_path / "gen_0"
    gen_dir.mkdir()
    attempt = SeedResultAttempt(str(gen_dir / "results"))

    def evaluate(_program, results):
        (Path(results) / "metrics.json").symlink_to(older_metrics)
        return {"metrics": {"combined_score": 42}}, 0.0

    with pytest.raises(ValueError, match="metrics.json.*within the seed attempt"):
        attempt.run(evaluate, "main.py")
