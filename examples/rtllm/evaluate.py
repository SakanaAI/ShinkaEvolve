"""
ShinkaEvolve evaluator for RTLLM (v2.0) PPA-speedup problems.

Optimizes a Verilog design for area / logic-depth / power under a FIXED functional
spec (the RTLLM design_description), using an all-open-source flow:

    correctness  : Icarus Verilog compile + RTLLM testbench
    equivalence  : Yosys SAT formal combinational equivalence, so a candidate cannot
                   win by overfitting the finite testbench
    PPA          : Yosys synthesis to the Nangate45 standard cells (area um^2 + logic
                   depth) + OpenSTA power. Absolute numbers are tool-dependent, so the
                   evolved and reference designs are compared on the identical flow.

Fitness:
    score = 0                                                  if syntax/equivalence fails
    score = 100 * geomean(area_ref/area_cand, depth_ref/depth_cand, power_ref/power_cand)
    => the RTLLM human reference scores exactly 100; beating it scores > 100.

Called by ShinkaEvolve as:
    python evaluate.py --program_path <candidate.v> --results_dir <dir>

Environment variables:
    RTLLM_PROBLEM_FILE  JSONL with embedded spec/testbench/reference (default: example.jsonl)
    RTLLM_DESIGN        design_name, e.g. "adder_8bit" (default: first in file)
    RTLLM_TIMEOUT       per-tool timeout seconds (default: 60)
    RTLLM_POWER         set "0" to skip OpenSTA power (score falls back to area x depth)
    RTLLM_YOSYS_IMAGE   docker image for yosys (default: hdlc/yosys:latest)
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MARKERS = ("EVOLVE-BLOCK-START", "EVOLVE-BLOCK-END")
LIBERTY = Path(__file__).resolve().parent / "pdk" / "nangate45.lib"


def _ensure_liberty(workdir: Path) -> None:
    """Place the standard-cell liberty in the yosys workdir (mounted into docker)."""
    dst = workdir / "lib.lib"
    if not dst.exists() and LIBERTY.exists():
        dst.write_bytes(LIBERTY.read_bytes())


def _load_problems(jsonl_path: Path) -> List[Dict]:
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _find_problem(problems: List[Dict], design: Optional[str]) -> Dict:
    if not design:
        return problems[0]
    for p in problems:
        if p["design_name"] == design:
            return p
    available = [p["design_name"] for p in problems]
    raise ValueError(f"Design '{design}' not found. Available: {available}")


def _strip_markers(text: str) -> str:
    return "".join(
        line for line in text.splitlines(keepends=True)
        if MARKERS[0] not in line and MARKERS[1] not in line
    )


# --------------------------------------------------------------------------- #
# Yosys invocation (native if present, else the hdlc/yosys docker image).
# --------------------------------------------------------------------------- #
def _yosys_argv(workdir: Path, script: str, timeout: int, name: str) -> List[str]:
    """Docker yosys argv (used when there is no native yosys). The container
    self-terminates so a stuck SAT solve can't outlive the Python timeout."""
    image = os.environ.get("RTLLM_YOSYS_IMAGE", "hdlc/yosys:latest")
    wd = str(workdir).replace("\\", "/")
    return ["docker", "run", "--rm", "--name", name, "-v", f"{wd}:/work", "-w", "/work",
            image, "bash", "-c", f"timeout {int(timeout)} yosys -p {shlex.quote(script)}"]


def _run_yosys(workdir: Path, script: str, timeout: int) -> Tuple[int, str]:
    if shutil.which("yosys"):
        proc = subprocess.run(["yosys", "-p", script], capture_output=True, text=True,
                              timeout=timeout, cwd=str(workdir))
        return proc.returncode, proc.stdout + "\n" + proc.stderr
    name = f"rtllm_yosys_{uuid.uuid4().hex[:10]}"
    argv = _yosys_argv(workdir, script, timeout=timeout, name=name)
    try:
        # +15s lets the in-container `timeout` fire first in the normal case
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 15)
        return proc.returncode, proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)  # no orphan
        raise


# Liberty-mapped synthesis: real area (um^2) from Nangate45 cell areas, and the
# critical-path length (real-gate logic depth) as the timing proxy.
_PPA_FLOW = (
    "read_verilog -sv {f}; hierarchy -top {top} -check; synth -top {top} -flatten; "
    "design -push-copy; abc -g AND; ltp -noff; design -pop; "
    "dfflibmap -liberty lib.lib; abc -liberty lib.lib; stat -liberty lib.lib"
)


def _ppa(workdir: Path, vfile: str, top: str, timeout: int) -> Optional[Tuple[float, int]]:
    """Return (chip_area_um2, logic_depth) or None on synthesis failure."""
    _ensure_liberty(workdir)
    rc, out = _run_yosys(workdir, _PPA_FLOW.format(f=vfile, top=top), timeout)
    if rc != 0:
        return None
    m_area = re.search(r"Chip area for module .*?:\s*([0-9.]+)", out)
    m_depth = re.search(r"Longest topological path in \S+ \(length=(\d+)\)", out)
    if not m_area or not m_depth:
        return None
    return float(m_area.group(1)), int(m_depth.group(1))


# OpenSTA power (uW): write the Nangate45 netlist, then report_power with a virtual
# 1ns clock + default input activity (recorded metadata; does not gate the score).
_NETLIST_FLOW = ("read_verilog -sv {f}; synth -top {top} -flatten; "
                 "dfflibmap -liberty lib.lib; abc -liberty lib.lib; "
                 "write_verilog -noattr netlist.v")
_POWER_TCL = """read_liberty /work/lib.lib
read_verilog /work/netlist.v
link_design {top}
set ck [get_ports -quiet {{clk clock CLK Clock clk_a clk_b clkA clkB}}]
if {{[llength $ck] > 0}} {{ create_clock -name clk -period 1.0 $ck }} else {{ create_clock -name vclk -period 1.0 }}
set_power_activity -input -activity 0.2
report_power -digits 6
exit
"""


def _measure_power(workdir: Path, vfile: str, top: str, timeout: int) -> Optional[float]:
    """Return total power in uW from OpenSTA, or None on failure (never raises)."""
    try:
        rc, _ = _run_yosys(workdir, _NETLIST_FLOW.format(f=vfile, top=top), timeout)
        if rc != 0 or not (workdir / "netlist.v").exists():
            return None
        (workdir / "power.tcl").write_text(_POWER_TCL.format(top=top), encoding="utf-8")
        image = os.environ.get("RTLLM_OPENSTA_IMAGE", "opensta:local")
        wd = str(workdir).replace("\\", "/")
        # NB: opensta entrypoint is a RELATIVE path -> do NOT set -w
        proc = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{wd}:/work", image, "/work/power.tcl"],
            capture_output=True, text=True, timeout=timeout)
        m = re.search(r"Total\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)",
                      proc.stdout + proc.stderr)
        return round(float(m.group(1)) * 1e6, 3) if m else None
    except Exception:
        return None


# Bounded I/O equivalence: build an output-comparing miter and SAT-check that the
# outputs match for n cycles from reset. This compares behaviour, not state encoding,
# so it accepts equivalent restructurings while still catching real behavioural change.
_BOUNDED_FLOW = (
    "read_verilog -sv ref.v; hierarchy -top {ref}; proc; memory; flatten; "
    "rename {ref} gold; design -stash gold; "
    "read_verilog -sv cand.v; hierarchy -top {top}; proc; memory; flatten; "
    "rename {top} gate; design -stash gate; "
    "design -copy-from gold -as gold gold; design -copy-from gate -as gate gate; "
    "miter -equiv -flatten -make_assert gold gate miter; hierarchy -top miter; "
    "async2sync; dffunmap; "
    "sat -seq {n} -prove-asserts -set-init-zero -verify miter"
)


def _bounded_io_equiv(workdir: Path, top: str, ref: str, n: int, timeout: int) -> Tuple[str, str]:
    """Return ('EQUIV_IO' | 'DIVERGES' | 'INCONCLUSIVE', yosys_output)."""
    try:
        rc, out = _run_yosys(workdir, _BOUNDED_FLOW.format(ref=ref, top=top, n=n), timeout)
    except Exception:
        return "INCONCLUSIVE", "timeout"
    if rc == 0 and "did fail" not in out:
        return "EQUIV_IO", out          # outputs matched for all n cycles
    if "did fail" in out or "model found" in out:
        return "DIVERGES", out          # counterexample: outputs differ
    return "INCONCLUSIVE", out          # SAT-hard: didn't finish in budget


def _classify_compile_error(stderr: str) -> str:
    low = stderr.lower()
    if "syntax error" in low:
        return "syntax_error"
    if "unable to bind" in low:
        return "port_binding_error"
    if "unknown module type" in low:
        return "missing_module"
    if "error" in low:
        return "compile_error"
    return "unknown_compile_error"


# --------------------------------------------------------------------------- #
# Reference PPA cache (reference is constant per design; compute once).
# --------------------------------------------------------------------------- #
def _ref_cache_path() -> Path:
    return Path(__file__).resolve().parent / "problems" / ".ppa_ref_cache.json"


def _reference_ppa(problem: Dict, timeout: int) -> Optional[Tuple[float, int, Optional[float]]]:
    """Return (area_um2, depth, power_uw) for the golden reference (cached)."""
    key = problem["design_name"]
    ref_hash = hashlib.sha1(problem["reference"].encode("utf-8")).hexdigest()[:12]
    cache_file = _ref_cache_path()
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    entry = cache.get(key)
    if entry and entry.get("hash") == ref_hash and "power_uw" in entry:
        return entry["area"], entry["depth"], entry["power_uw"]

    with tempfile.TemporaryDirectory(prefix="rtllm_ref_") as tmp:
        wd = Path(tmp)
        (wd / "ref.v").write_text(problem["reference"], encoding="utf-8")
        ppa = _ppa(wd, "ref.v", problem["ref_module"], timeout)
        ref_power = (_measure_power(wd, "ref.v", problem["ref_module"], timeout)
                     if ppa is not None else None)
    if ppa is None:
        return None
    cache[key] = {"hash": ref_hash, "area": ppa[0], "depth": ppa[1], "power_uw": ref_power}
    try:
        tmp_path = cache_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_file)
    except Exception:
        pass
    return ppa[0], ppa[1], ref_power


def _run_testbench(wd: Path, candidate_file: str, tb_module: str,
                   timeout: int) -> Tuple[bool, str, int]:
    """Compile candidate + testbench with iverilog, run, and read the pass/fail banner.

    Returns (passed, stage_or_error, failures). The testbench is the cheap first
    filter; equivalence to the reference is the actual correctness gate.
    """
    sim = wd / "sim.vvp"
    comp = subprocess.run(
        ["iverilog", "-g2012", "-Wno-timescale", "-s", tb_module,
         "-o", str(sim), str(wd / candidate_file), str(wd / "testbench.v")],
        capture_output=True, text=True, timeout=timeout,
    )
    if comp.returncode != 0:
        return False, _classify_compile_error(comp.stderr), -1
    try:
        run = subprocess.run(["vvp", str(sim)], capture_output=True, text=True,
                             timeout=timeout, cwd=str(wd))
    except subprocess.TimeoutExpired:
        return False, "timeout", -1
    out = run.stdout + "\n" + run.stderr
    if "Your Design Passed" in out:
        return True, "pass", 0
    m = re.search(r"completed with\s+(\d+)\s*/", out)
    return False, "tb_fail", int(m.group(1)) if m else -1


def evaluate_rtllm(program_path: str, problem: Dict, timeout: int) -> Dict[str, Any]:
    top = problem["top_module"]
    ref_mod = problem["ref_module"]

    candidate = _strip_markers(Path(program_path).read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="rtllm_eval_") as tmp:
        wd = Path(tmp)
        (wd / "cand.v").write_text(candidate, encoding="utf-8")
        (wd / "ref.v").write_text(problem["reference"], encoding="utf-8")
        (wd / "testbench.v").write_text(problem["testbench"], encoding="utf-8")
        # testbench data files (e.g. $readmemh targets); basename only for safety
        for fn, content in (problem.get("aux_files") or {}).items():
            (wd / Path(fn).name).write_text(content, encoding="utf-8")

        # --- Gate 1: syntax + RTLLM testbench (mirror of VCS flow) ---
        tb_pass, tb_stage, failures = _run_testbench(
            wd, "cand.v", problem["tb_module"], timeout)
        if tb_stage in ("syntax_error", "compile_error", "port_binding_error",
                        "missing_module", "unknown_compile_error"):
            return {
                "combined_score": 0.0,
                "public": {"stage": "syntax", "error_class": tb_stage,
                           "tb_pass": False, "equivalent": False},
                "private": {},
                "text_feedback": f"Does not compile ({tb_stage}). Fix syntax; keep "
                                 f"module '{top}' and its port interface unchanged.",
            }

        # Correctness: the testbench is the cheap first filter; every passing
        # candidate is then checked for cycle-accurate equivalence to the reference
        # with a bounded I/O miter (DIVERGES -> reject, EQUIV_IO -> accept,
        # timeout on a hard SAT -> fall back to the testbench verdict).
        if not tb_pass:
            return {
                "combined_score": 0.0,
                "public": {"stage": "functional", "tb_pass": False,
                           "tb_failures": failures, "verification": "none"},
                "private": {},
                "text_feedback": "Fails the testbench: wrong on at least one of the "
                                 "checked input vectors. Keep the specified function.",
            }
        # Bounded I/O equivalence: compare outputs for 16 cycles from reset, 20s SAT budget.
        bnd, bnd_out = _bounded_io_equiv(wd, top, ref_mod, 16, 20)
        if bnd == "DIVERGES":
            return {
                "combined_score": 0.0,
                "public": {"stage": "equivalence", "tb_pass": True,
                           "verification": "formal", "equivalent": False},
                "private": {"equiv_tail": bnd_out[-1500:]},
                "text_feedback": "Passes the testbench but is not equivalent to the "
                                 "reference: it differs on some input sequence. Match the "
                                 "reference on every output, every cycle; do not change "
                                 "timing or remove registers.",
            }
        verification = "formal" if bnd == "EQUIV_IO" else "testbench"

        # --- PPA: candidate vs reference on the identical AIG flow ---
        cand_ppa = _ppa(wd, "cand.v", top, timeout)
        ref_ppa = _reference_ppa(problem, timeout)
        if cand_ppa is None or ref_ppa is None:
            return {
                "combined_score": 0.0,
                "public": {"stage": "synthesis", "tb_pass": tb_pass,
                           "verification": verification},
                "private": {},
                "text_feedback": "Correct, but synthesis (yosys) failed to map the "
                                 "design. Use synthesizable constructs.",
            }

        a_c, d_c = cand_ppa
        a_r, d_r, p_r = ref_ppa
        p_c = (_measure_power(wd, "cand.v", top, timeout)
               if os.environ.get("RTLLM_POWER", "1") == "1" else None)

        area_ratio = a_r / a_c if a_c else 0.0
        delay_ratio = d_r / d_c if d_c else 0.0
        power_ratio = (p_r / p_c) if (p_r and p_c) else None

        # combined_score = 100 x geomean of the per-axis improvement ratios (area,
        # logic-depth, and power when OpenSTA produced a reading). 100 = reference.
        ratios = [r for r in (area_ratio, delay_ratio, power_ratio) if r]
        prod = 1.0
        for r in ratios:
            prod *= r
        speedup = prod ** (1.0 / len(ratios)) if ratios else 0.0
        score = 100.0 * speedup

        verdict = "beats" if score > 100.0 else ("matches" if score == 100.0 else "below")
        how = "formally equivalent" if verification == "formal" else "passes the testbench"
        pwr_str = (f", power {p_c} vs {p_r} uW (x{power_ratio:.2f})"
                   if power_ratio else f", power {p_c} uW")
        feedback = (
            f"Correct ({how}). PPA vs reference: "
            f"area {a_c} vs {a_r} (x{area_ratio:.2f}), "
            f"depth {d_c} vs {d_r} (x{delay_ratio:.2f}){pwr_str}; "
            f"PPA-speedup={speedup:.3f} -> score {score:.1f} ({verdict} the human reference). "
            f"Reduce area, depth AND power together while staying correct."
        )
        return {
            "combined_score": score,
            "public": {
                "stage": "pass", "tb_pass": tb_pass, "verification": verification,
                "equivalent": verification == "formal",
                "area": a_c, "depth": d_c, "power_uw": p_c,
                "ref_area": a_r, "ref_depth": d_r, "ref_power_uw": p_r,
                "area_ratio": area_ratio, "delay_ratio": delay_ratio,
                "power_ratio": power_ratio, "speedup": speedup,
            },
            "private": {},
            "text_feedback": feedback,
        }


def _save_results(results_dir: str, metrics: dict, correct: bool, error: str):
    Path(results_dir, "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=4), encoding="utf-8")
    Path(results_dir, "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8")


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    problem_file = os.environ.get("RTLLM_PROBLEM_FILE", "example.jsonl")
    design = os.environ.get("RTLLM_DESIGN", "")
    timeout = int(os.environ.get("RTLLM_TIMEOUT", "60"))

    problem_path = Path(problem_file)
    if not problem_path.is_absolute():
        problem_path = Path(__file__).resolve().parent / problem_path

    try:
        problems = _load_problems(problem_path)
        problem = _find_problem(problems, design or None)
    except (FileNotFoundError, ValueError) as e:
        _save_results(results_dir, {"combined_score": 0.0, "public": {}, "private": {}},
                      False, str(e))
        return

    try:
        metrics = evaluate_rtllm(program_path, problem, timeout)
    except Exception as e:  # never crash the evolution loop
        _save_results(results_dir, {"combined_score": 0.0, "public": {}, "private": {}},
                      False, repr(e))
        return

    correct = metrics["public"].get("stage") == "pass"
    error = "" if metrics["combined_score"] >= 100.0 else metrics.get("text_feedback", "")
    _save_results(results_dir, metrics, correct, error)

    pub = metrics.get("public", {})
    print(f"Design: {problem['design_name']} [{problem.get('category','')}]")
    print(f"Score:  {metrics['combined_score']:.1f}  (reference = 100.0)")
    if pub.get("stage") == "pass":
        print(f"  area {pub['area']} vs ref {pub['ref_area']}  |  "
              f"depth {pub['depth']} vs ref {pub['ref_depth']}  |  "
              f"speedup {pub['speedup']:.3f}")
    else:
        print(f"  stage={pub.get('stage')}  equivalent={pub.get('equivalent')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Verilog candidate against RTLLM PPA")
    parser.add_argument("--program_path", type=str, default="initial.v")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
