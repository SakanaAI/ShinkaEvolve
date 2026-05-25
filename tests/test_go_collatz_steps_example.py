from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_evaluator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "go_collatz_steps" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("go_collatz_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_go_evaluator_runs_candidate_and_parses_output(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    run_calls = []

    def fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        output_path = Path(args[-1])
        output_path.write_text("0\n16\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    answers, _ = evaluator._run_program(str(tmp_path / "candidate.go"), [1, 7])

    assert answers == [0, 16]
    assert run_calls
    assert run_calls[0][0][:2] == ["go", "run"]
    assert run_calls[0][0][2] == str(tmp_path / "candidate.go")


def test_go_evaluator_scores_correct_candidate(monkeypatch, tmp_path):
    evaluator = _load_evaluator()

    def fake_run(args, **kwargs):
        output_path = Path(args[-1])
        queries = [int(line) for line in Path(args[-2]).read_text().splitlines()]
        answers = evaluator._expected_steps(queries)
        output_path.write_text(
            "\n".join(str(answer) for answer in answers),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.go"), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is True
    assert metrics["public"]["accuracy"] == 1.0
    assert metrics["combined_score"] > 0.0


def test_go_evaluator_reports_nonzero_exit(monkeypatch, tmp_path):
    evaluator = _load_evaluator()

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="compile failed",
        )

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    results_dir = tmp_path / "results"
    evaluator.main(str(tmp_path / "candidate.go"), str(results_dir))

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert "Go program failed" in correct["error"]
    assert metrics["combined_score"] == 0.0
