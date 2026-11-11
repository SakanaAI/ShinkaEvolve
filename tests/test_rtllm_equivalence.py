from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest


def _load_evaluator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "rtllm" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("rtllm_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROBLEM = {
    "design_name": "toy",
    "top_module": "toy",
    "ref_module": "toy",
    "tb_module": "tb",
    "reference": "module toy(); endmodule\n",
    "testbench": "",
}


def _evaluate(monkeypatch, tmp_path, evaluator, sat_output, prove):
    """Run evaluate_rtllm with every tool stubbed except the SAT verdict."""
    monkeypatch.setattr(evaluator, "_run_testbench", lambda *a, **k: (True, "pass", 0))
    monkeypatch.setattr(evaluator, "_ppa", lambda *a, **k: (10.0, 4))
    monkeypatch.setattr(evaluator, "_reference_ppa", lambda *a, **k: (10.0, 4, None))
    monkeypatch.setattr(evaluator, "_measure_power", lambda *a, **k: None)
    monkeypatch.setattr(evaluator, "_run_yosys", lambda wd, script, timeout: sat_output(script))
    monkeypatch.setenv("RTLLM_PROVE", "1" if prove else "0")
    cand = tmp_path / "cand.sv"
    cand.write_text("module toy(); endmodule\n", encoding="utf-8")
    return evaluator.evaluate_rtllm(str(cand), PROBLEM, timeout=5)


def _sat(induction_out):
    def run(script):
        if "-tempinduct" in script:
            return 0, induction_out
        return 0, "SAT proof finished - no model found: SUCCESS!"
    return run


def test_bounded_pass_is_not_labelled_formal(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    calls = []

    def run(script):
        calls.append(script)
        return 0, "SAT proof finished - no model found: SUCCESS!"

    metrics = _evaluate(monkeypatch, tmp_path, evaluator, run, prove=False)

    assert metrics["public"]["verification"] == "bounded"
    assert metrics["public"]["equivalent"] is False
    assert metrics["combined_score"] == pytest.approx(100.0)
    assert not any("-tempinduct" in s for s in calls)


def test_induction_proof_earns_formal_label(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    out = "Induction step proven: SUCCESS!"
    metrics = _evaluate(monkeypatch, tmp_path, evaluator, _sat(out), prove=True)

    assert metrics["public"]["verification"] == "formal"
    assert metrics["public"]["equivalent"] is True


def test_induction_counterexample_rejects_candidate(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    out = "SAT temporal induction proof finished - model found for base case: FAIL!"
    metrics = _evaluate(monkeypatch, tmp_path, evaluator, _sat(out), prove=True)

    assert metrics["combined_score"] == 0.0
    assert metrics["public"]["stage"] == "equivalence"
    assert metrics["public"]["verification"] == "formal"


def test_inconclusive_induction_stays_bounded(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    out = "Reached maximum number of time steps -> proof failed."
    metrics = _evaluate(monkeypatch, tmp_path, evaluator, _sat(out), prove=True)

    assert metrics["public"]["verification"] == "bounded"
    assert metrics["combined_score"] == pytest.approx(100.0)


# --- real Yosys runs (skipped when yosys is not installed) ------------------- #

# square_wave from issue #206: with freq=0 the 32-bit reference never toggles, while
# an 8-bit counter wraps and toggles at cycle 256.
SQ_REF = """module square_wave(input clk, input [7:0] freq, output reg wave_out);
  reg [31:0] count;
  initial begin count = 0; wave_out = 0; end
  always @(posedge clk)
    if (count == freq - 1) begin count <= 0; wave_out <= ~wave_out; end
    else count <= count + 1;
endmodule
"""
SQ_CAND = SQ_REF.replace("reg [31:0] count", "reg [7:0] count").replace(
    "freq - 1", "freq - 8'd1").replace("count + 1", "count + 8'd1")

# Diverges at cycle 22 (c == 21), past the 16-cycle bound.
PULSE_REF = """module pulse(input clk, input rst_n, output reg v);
  reg [4:0] c;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin c <= 0; v <= 0; end
    else begin c <= c + 5'd1; v <= (c == 5'd20); end
endmodule
"""
PULSE_CAND = PULSE_REF.replace("(c == 5'd20)", "c[4] & c[2]")

ACC_REF = """module acc(input clk, input rst_n, input [7:0] d, output reg [7:0] q);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) q <= 0; else q <= q + d;
endmodule
"""
ACC_CAND = ACC_REF.replace("q <= q + d", "q <= d + q")

needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys is not installed")


@needs_yosys
@pytest.mark.parametrize(
    "ref, cand, top, expected",
    [
        (ACC_REF, ACC_CAND, "acc", "PROVEN"),
        (PULSE_REF, PULSE_CAND, "pulse", "DIVERGES"),
        (SQ_REF, SQ_CAND, "square_wave", "UNKNOWN"),
    ],
)
def test_late_divergence_passes_bounded_but_not_induction(tmp_path, ref, cand, top, expected):
    evaluator = _load_evaluator()
    (tmp_path / "ref.v").write_text(ref, encoding="utf-8")
    (tmp_path / "cand.v").write_text(cand, encoding="utf-8")

    bounded, _ = evaluator._bounded_io_equiv(tmp_path, top, top, 16, 120)
    proof, _ = evaluator._induction_equiv(tmp_path, top, top, 32, 300)

    assert bounded == "EQUIV_IO"
    assert proof == expected
