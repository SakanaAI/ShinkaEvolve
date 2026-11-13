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
    monkeypatch.setattr(evaluator, "_run_yosys", lambda wd, script, timeout: sat_output(wd, script))
    monkeypatch.setenv("RTLLM_PROVE", "1" if prove else "0")
    cand = tmp_path / "cand.sv"
    cand.write_text("module toy(); endmodule\n", encoding="utf-8")
    return evaluator.evaluate_rtllm(str(cand), PROBLEM, timeout=5)


def _sat(induction_out):
    def run(wd, script):
        if "-tempinduct" in script:
            (wd / "induction.log").write_text(induction_out, encoding="utf-8")
            return 0, ""
        return 0, "SAT proof finished - no model found: SUCCESS!"
    return run


def test_bounded_pass_is_not_labelled_formal(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    calls = []

    def run(wd, script):
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


def test_induction_verdict_ignores_design_output(monkeypatch, tmp_path):
    evaluator = _load_evaluator()

    def run(wd, script, timeout):
        # a candidate's $display lands in stdout; the solver's verdict is in the log
        (wd / "induction.log").write_text(
            "SAT temporal induction proof finished - model found for base case: FAIL!",
            encoding="utf-8")
        return 0, "Induction step proven: SUCCESS!"

    monkeypatch.setattr(evaluator, "_run_yosys", run)
    verdict, _ = evaluator._induction_equiv(tmp_path, "toy", "toy", 32, 5)

    assert verdict == "DIVERGES"


def test_formal_needs_the_real_clock_proof_too(monkeypatch, tmp_path):
    evaluator = _load_evaluator()

    def run(wd, script):
        if "-tempinduct" in script:
            verdict = ("SAT temporal induction proof finished - model found for base "
                       "case: FAIL!" if "clk2fflogic" in script
                       else "Induction step proven: SUCCESS!")
            (wd / "induction.log").write_text(verdict, encoding="utf-8")
            return 0, ""
        return 0, "SAT proof finished - no model found: SUCCESS!"

    metrics = _evaluate(monkeypatch, tmp_path, evaluator, run, prove=True)

    assert metrics["combined_score"] == 0.0
    assert metrics["public"]["verification"] == "formal"
    assert metrics["public"]["equivalent"] is False


def test_retries_without_nodisplay_on_older_yosys(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    scripts = []

    def run(wd, script, timeout):
        scripts.append(script)
        if "-nodisplay" in script:
            return 1, "ERROR: Command syntax error: Unknown option or option in arguments."
        return 1, "SAT proof finished - model found: FAIL!\nAssert ... did fail"

    monkeypatch.setattr(evaluator, "_run_yosys", run)
    verdict, _ = evaluator._bounded_io_equiv(tmp_path, "toy", "toy", 16, 5)

    assert verdict == "DIVERGES"
    assert len(scripts) == 2 and "-nodisplay" not in scripts[1]


def test_rejected_script_fails_loudly(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    monkeypatch.setattr(evaluator, "_run_yosys", lambda wd, script, timeout: (
        1, "ERROR: Command syntax error: Unknown option or option in arguments."))

    with pytest.raises(evaluator.YosysScriptError):
        evaluator._bounded_io_equiv(tmp_path, "toy", "toy", 16, 5)
    with pytest.raises(evaluator.YosysScriptError):
        evaluator._induction_equiv(tmp_path, "toy", "toy", 32, 5)


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
# Same late divergence, but the design prints the solver's success line itself.
PULSE_SPOOF = PULSE_CAND.replace(
    "endmodule", '  initial $display("Induction step proven: SUCCESS!");\nendmodule')

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
        (PULSE_REF, PULSE_SPOOF, "pulse", "DIVERGES"),
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


CLK_REF = """module r(input clk, input d, output reg q);
  always @(posedge clk) q <= d;
endmodule
"""
CLK_NEG = CLK_REF.replace("posedge", "negedge")


@needs_yosys
@pytest.mark.parametrize(
    "ref, cand, top, expected",
    [
        (CLK_REF, CLK_REF, "r", "PROVEN"),
        (ACC_REF, ACC_CAND, "acc", "PROVEN"),
        (CLK_REF, CLK_NEG, "r", "DIVERGES"),
    ],
    ids=["same-clock", "acc", "posedge-vs-negedge"],
)
def test_real_clock_proof_sees_clock_edges(tmp_path, ref, cand, top, expected):
    evaluator = _load_evaluator()
    (tmp_path / "ref.v").write_text(ref, encoding="utf-8")
    (tmp_path / "cand.v").write_text(cand, encoding="utf-8")

    shared, _ = evaluator._induction_equiv(tmp_path, top, top, 32, 300)
    real, _ = evaluator._induction_equiv(tmp_path, top, top, 32, 300, real_clocks=True)

    assert shared == "PROVEN"  # one shared clock can't tell posedge from negedge
    assert real == expected


BUF = """module m(input d, output q);
  assign q = d;
endmodule
"""


@needs_yosys
def test_combinational_mismatch_is_caught(tmp_path):
    # On Yosys 0.33 `-nodisplay` used to break the script, so this came back
    # INCONCLUSIVE and the candidate was scored on the testbench alone.
    evaluator = _load_evaluator()
    (tmp_path / "ref.v").write_text(BUF, encoding="utf-8")
    (tmp_path / "cand.v").write_text(BUF.replace("= d", "= ~d"), encoding="utf-8")

    bounded, _ = evaluator._bounded_io_equiv(tmp_path, "m", "m", 16, 120)

    assert bounded == "DIVERGES"
