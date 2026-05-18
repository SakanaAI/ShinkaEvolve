from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_evaluator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "wolfram_simplify" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("wolfram_simplify_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_harness_factory(harness_output):
    """Build a subprocess.run stub for the wolfram_simplify harness.

    The harness writes its analysis JSON to the LAST positional arg of the
    wolframscript invocation; the stub does the same with the canned
    payload."""

    def fake_run(args, **kwargs):
        out_path = Path(args[-1])
        out_path.write_text(json.dumps(harness_output), encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    return fake_run


_SEED_ANALYSIS = {
    "leafCount": 174,
    "blocklisted": [],
    "parseError": None,
    "outputs": {
        "t1": {"inputs": [0.0, 1.0], "outputs": [-1.0, 0.0]},
        "t2": {"inputs": [0.0, 1.0], "outputs": [1.0, 1.0]},
        "t3": {"inputs": [0, 5], "outputs": [0, 70]},
        "t4": {"inputs": [0.0, 2.0], "outputs": [0.0, 4.0]},
        "t5": {"inputs": [[1.0, 2.0]], "outputs": [[1.5]]},
    },
}


def test_simplify_evaluator_scores_matching_candidate(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    candidate = dict(_SEED_ANALYSIS)
    candidate["leafCount"] = 60
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_harness_factory({"seed": _SEED_ANALYSIS, "candidate": candidate}),
    )
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.wl"), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is True
    assert metrics["combined_score"] == 174 / 60
    assert metrics["public"]["seed_leafcount"] == 174
    assert metrics["public"]["candidate_leafcount"] == 60


def test_simplify_evaluator_rejects_output_mismatch(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    candidate = {
        "leafCount": 50,
        "blocklisted": [],
        "parseError": None,
        "outputs": {
            "t1": {"inputs": [0.0, 1.0], "outputs": [-1.0, 42.0]},  # t1[1.0] should be 0.0
            "t2": _SEED_ANALYSIS["outputs"]["t2"],
            "t3": _SEED_ANALYSIS["outputs"]["t3"],
            "t4": _SEED_ANALYSIS["outputs"]["t4"],
            "t5": _SEED_ANALYSIS["outputs"]["t5"],
        },
    }
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_harness_factory({"seed": _SEED_ANALYSIS, "candidate": candidate}),
    )
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.wl"), str(results_dir))

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert metrics["combined_score"] == -1.0
    assert "t1" in metrics["public"]["error"]


def test_simplify_evaluator_rejects_blocklisted_head(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    candidate = {
        "leafCount": 30,
        "blocklisted": ["FullSimplify"],
        "parseError": None,
        "outputs": {},
    }
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_harness_factory({"seed": _SEED_ANALYSIS, "candidate": candidate}),
    )
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.wl"), str(results_dir))

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert metrics["combined_score"] == -1.0
    assert "FullSimplify" in metrics["public"]["error"]


def test_simplify_evaluator_int_float_crossover_accepted(monkeypatch, tmp_path):
    """A simplified candidate may legitimately return integer 1 where the
    seed's Sin/Cos expansion evaluates to a near-1.0 Real — the evaluator
    compares numerically and accepts that crossover."""
    evaluator = _load_evaluator()
    candidate = {
        "leafCount": 50,
        "blocklisted": [],
        "parseError": None,
        "outputs": {
            "t1": _SEED_ANALYSIS["outputs"]["t1"],
            "t2": {"inputs": [0.0, 1.0], "outputs": [1, 1]},  # int 1 vs seed's 1.0
            "t3": _SEED_ANALYSIS["outputs"]["t3"],
            "t4": _SEED_ANALYSIS["outputs"]["t4"],
            "t5": _SEED_ANALYSIS["outputs"]["t5"],
        },
    }
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_harness_factory({"seed": _SEED_ANALYSIS, "candidate": candidate}),
    )
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.wl"), str(results_dir))

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is True
