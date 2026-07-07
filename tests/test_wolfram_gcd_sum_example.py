from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_evaluator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "wolfram_gcd_sum" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("wolfram_gcd_sum_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_wolfram_evaluator_run_once_parses_bridge_output(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    run_calls = []

    def fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        out_path = Path(args[-1])
        out_path.write_text(
            json.dumps({"result": "336784", "time_ms": 12.5, "n": 300}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    result = evaluator._run_once(str(tmp_path / "candidate.wl"))

    assert result["result"] == "336784"
    assert result["time_ms"] == 12.5
    assert run_calls and run_calls[0][0][1] == "-file"


def _fake_run_factory(seed_time_ms, candidate_result, candidate_time_ms):
    """Build a subprocess.run stub that discriminates the calibration run
    (initial.wl) from the candidate run by program path."""

    def fake_run(args, **kwargs):
        out_path = Path(args[-1])
        program = args[2]
        if program.endswith("initial.wl"):
            payload = {"result": "336784", "time_ms": seed_time_ms, "n": 300}
        else:
            payload = {
                "result": candidate_result,
                "time_ms": candidate_time_ms,
                "n": 300,
            }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    return fake_run


def test_wolfram_evaluator_scores_correct_candidate(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_run_factory(
            seed_time_ms=100.0, candidate_result="336784", candidate_time_ms=10.0
        ),
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
    # calibrated baseline 100 ms / candidate 10 ms = 10x speedup.
    assert metrics["combined_score"] == 10.0


def test_wolfram_evaluator_rejects_wrong_answer(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        _fake_run_factory(
            seed_time_ms=100.0, candidate_result="999", candidate_time_ms=10.0
        ),
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


def test_wolfram_evaluator_records_empty_output_as_failure(monkeypatch, tmp_path):
    evaluator = _load_evaluator()

    def fake_run(args, **kwargs):
        out_path = Path(args[-1])
        program = args[2]
        if program.endswith("initial.wl"):
            out_path.write_text(
                json.dumps({"result": "336784", "time_ms": 100.0, "n": 300}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.wl"), str(results_dir))

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    assert correct == {"correct": False, "error": "empty output file"}
    assert metrics["combined_score"] == -1.0
    assert metrics["public"]["error"] == "empty output file"
