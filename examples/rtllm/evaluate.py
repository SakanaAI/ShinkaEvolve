"""
ShinkaEvolve evaluator for RTLLM (v2.0) PPA-speedup problems.

Optimizes a Verilog design for area/delay under a FIXED functional spec
(the RTLLM design_description), mirroring RTLLM's flow with open-source tools:

    correctness  : Icarus Verilog compile + RTLLM testbench   (mirror of Synopsys VCS)
    equivalence  : Yosys SAT formal combinational equivalence  (reward-hack-proof gate,
                   the open analog of Synopsys Formality used by RTL-OPT)
    PPA          : Yosys synthesis mapped to the Nangate45 standard-cell liberty;
                   area = real chip area (um^2), delay = critical-path logic depth
                   (open analog of Synopsys Design Compiler area/WNS; absolute numbers
                   differ by tool, so we compare evolved-vs-reference on the same flow)

Fitness (p_speedup, kernel_bench-style but under fixed spec):
    score = 0                              if syntax/equivalence fails
    score = 100 * sqrt( area_ref/area_cand * depth_ref/depth_cand )   otherwise
    => the RTLLM human reference scores exactly 100; beating it scores > 100.

The candidate must be logically EQUIVALENT to the reference (not merely pass the
finite testbench), which closes the testbench-overfitting reward-hacking hole.

Called by ShinkaEvolve as:
    python evaluate.py --program_path <candidate.v> --results_dir <dir>

Environment variables:
    RTLLM_PROBLEM_FILE  JSONL with embedded spec/testbench/reference
                        (default: problems/rtllm_proto.jsonl)
    RTLLM_DESIGN        design_name, e.g. "adder_8bit" (default: first in file)
    RTLLM_TIMEOUT       per-tool timeout seconds (default: 60)
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
def _yosys_argv(workdir: Path, script: str, timeout: Optional[int] = None,
                name: Optional[str] = None) -> List[str]:
    if shutil.which("yosys"):
        return ["yosys", "-p", script]
    image = os.environ.get("RTLLM_YOSYS_IMAGE", "hdlc/yosys:latest")
    wd = str(workdir).replace("\\", "/")
    cmd = ["docker", "run", "--rm"]
    if name:
        cmd += ["--name", name]
    cmd += ["-v", f"{wd}:/work", "-w", "/work", image]
    if timeout:
        # self-terminate INSIDE the container: a stuck SAT solve (e.g. divider/
        # multiplier equivalence) must not outlive the Python-side timeout and
        # orphan a CPU-pinning container.
        cmd += ["bash", "-c", f"timeout {int(timeout)} yosys -p {shlex.quote(script)}"]
    else:
        cmd += ["yosys", "-p", script]
    return cmd


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


_EQUIV_FLOW = (
    "read_verilog -sv ref.v; hierarchy -top {ref}; proc; memory; opt_clean; flatten; "
    "rename {ref} gold; design -stash gold; "
    "read_verilog -sv cand.v; hierarchy -top {top}; proc; memory; opt_clean; flatten; "
    "rename {top} gate; design -stash gate; "
    "design -copy-from gold -as gold gold; design -copy-from gate -as gate gate; "
    "equiv_make gold gate equiv; hierarchy -top equiv; "
    "equiv_simple; equiv_induct; equiv_status -assert"
)


def _check_equivalence(workdir: Path, top: str, ref: str, timeout: int) -> Tuple[bool, str]:
    try:
        rc, out = _run_yosys(workdir, _EQUIV_FLOW.format(top=top, ref=ref), timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    proven = rc == 0 and "Equivalence successfully proven" in out
    return proven, out


# Bounded I/O equivalence: build an output-comparing miter and SAT-check that outputs
# match for n cycles from reset. Unlike equiv_induct this ignores the internal state
# ENCODING, so it accepts behaviorally-equivalent state-restructurings (FSM->shift-reg,
# resized counters) while still catching genuine behavioral change. async2sync converts
# $adff so the SAT solver can import the flip-flops.
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
def _ref_cache_path(problem: Dict) -> Path:
    return Path(__file__).resolve().parent / "problems" / ".ppa_ref_cache.json"


def _reference_ppa(problem: Dict, timeout: int) -> Optional[Tuple[float, int, Optional[float]]]:
    """Return (area_um2, depth, power_uw) for the golden reference (cached)."""
    key = problem["design_name"]
    ref_hash = hashlib.sha1(problem["reference"].encode("utf-8")).hexdigest()[:12]
    cache_file = _ref_cache_path(problem)
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
    """Mirror RTLLM's flow: iverilog compile candidate+testbench, run, grep banner.

    Returns (passed, stage_or_error, failures). Recorded for the science even
    though the *gate* is formal equivalence, so we can measure how often a design
    passes the finite testbench but is not actually equivalent.
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

        # --- Gate 2: correctness (UNIFORM for every design) ---
        # 1) testbench is the cheap first filter; failing it = wrong, rejected.
        # 2) every testbench-passing candidate is then checked for cycle-accurate
        #    equivalence to the reference via a bounded I/O miter:
        #       DIVERGES  -> testbench-overfit, REJECTED in-loop (the reward-hack guard)
        #       EQUIV_IO  -> behaviorally equivalent, accepted ("formal" verdict)
        #       timeout   -> SAT-hard (div/mul/deep-seq): fall back to the testbench
        #                    verdict, recorded honestly as "testbench".
        if not tb_pass:
            return {
                "combined_score": 0.0,
                "public": {"stage": "functional", "tb_pass": False,
                           "tb_failures": failures, "verification": "none"},
                "private": {},
                "text_feedback": "Fails the testbench: wrong on at least one of the "
                                 "checked input vectors. Keep the specified function.",
            }
        bnd_cycles = int(os.environ.get("RTLLM_BOUNDED_CYCLES", "16"))
        bnd_timeout = int(os.environ.get("RTLLM_BOUNDED_TIMEOUT", "20"))
        bnd, bnd_out = _bounded_io_equiv(wd, top, ref_mod, bnd_cycles, bnd_timeout)
        if bnd == "DIVERGES":
            return {
                "combined_score": 0.0,
                "public": {"stage": "equivalence", "tb_pass": True,
                           "verification": "formal", "equivalent": False},
                "private": {"equiv_tail": bnd_out[-1500:]},
                "text_feedback": "Passes the finite testbench but is NOT cycle-accurate "
                                 "equivalent to the reference (testbench-overfit): it differs "
                                 "on some input sequence. It must match the reference on every "
                                 "output, every cycle, for ALL inputs -- do not change timing, "
                                 "remove registers, or rely on inputs being held constant.",
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

        # How to combine the per-axis improvement ratios into one number:
        #   RTLLM_SCORE_MEAN="geo"   -> geometric mean  (product^(1/n)); the standard
        #       for combining ratios/speedups (SPEC etc.); scale-invariant, one bad
        #       axis (ratio<1) is fully felt, any 0 -> 0.
        #   RTLLM_SCORE_MEAN="arith" -> arithmetic mean of the ratios; a big win on one
        #       axis can mask a loss on another (more forgiving, less principled).
        def _combine(ratios):
            rs = [r for r in ratios if r]
            if not rs:
                return 0.0
            if os.environ.get("RTLLM_SCORE_MEAN", "geo") == "arith":
                return sum(rs) / len(rs)
            prod = 1.0
            for r in rs:
                prod *= r
            return prod ** (1.0 / len(rs))

        # Two speedups, both recorded; ShinkaEvolve optimizes ONE scalar (combined_score):
        #   ppa2 = area x delay (power decoupled)
        #   ppa3 = area x delay x power  (matches RTLLM's full PPA)
        speedup_ppa2 = _combine([area_ratio, delay_ratio])
        speedup_ppa3 = _combine([area_ratio, delay_ratio, power_ratio]) if power_ratio else None
        # RTLLM_SCORE_AXES: "3" -> power counts (default), "2" -> area x delay only
        axes = os.environ.get("RTLLM_SCORE_AXES", "3")
        speedup = speedup_ppa3 if (axes == "3" and speedup_ppa3) else speedup_ppa2
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
                "equivalent": verification == "formal", "score_axes": axes,
                "score_mean": os.environ.get("RTLLM_SCORE_MEAN", "geo"),
                "area": a_c, "depth": d_c, "power_uw": p_c,
                "ref_area": a_r, "ref_depth": d_r, "ref_power_uw": p_r,
                "area_ratio": area_ratio, "delay_ratio": delay_ratio,
                "power_ratio": power_ratio,
                "speedup": speedup, "speedup_ppa2": speedup_ppa2,
                "speedup_ppa3": speedup_ppa3 if power_ratio else None,
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
    problem_file = os.environ.get("RTLLM_PROBLEM_FILE", "problems/rtllm_proto.jsonl")
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
