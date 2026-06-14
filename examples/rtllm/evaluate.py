"""
ShinkaEvolve evaluator for RTLLM (v2.0) PPA-speedup problems.

Optimizes a Verilog design for area/delay under a FIXED functional spec
(the RTLLM design_description), mirroring RTLLM's flow with open-source tools:

    correctness  : Icarus Verilog compile + RTLLM testbench   (mirror of Synopsys VCS)
    equivalence  : Yosys SAT formal combinational equivalence  (reward-hack-proof gate,
                   the open analog of Synopsys Formality used by RTL-OPT)
    PPA          : Yosys -> ABC AIG mapping; area = AIG cell count, delay = logic depth
                   (open, license-free, deterministic proxy for Design Compiler area/WNS)

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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MARKERS = ("EVOLVE-BLOCK-START", "EVOLVE-BLOCK-END")


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
def _yosys_argv(workdir: Path, script: str) -> List[str]:
    if shutil.which("yosys"):
        return ["yosys", "-p", script]
    image = os.environ.get("RTLLM_YOSYS_IMAGE", "hdlc/yosys:latest")
    wd = str(workdir).replace("\\", "/")
    return [
        "docker", "run", "--rm",
        "-v", f"{wd}:/work", "-w", "/work",
        image, "yosys", "-p", script,
    ]


def _run_yosys(workdir: Path, script: str, timeout: int) -> Tuple[int, str]:
    native = bool(shutil.which("yosys"))
    proc = subprocess.run(
        _yosys_argv(workdir, script),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(workdir) if native else None,
    )
    return proc.returncode, proc.stdout + "\n" + proc.stderr


_PPA_FLOW = (
    "read_verilog -sv {f}; hierarchy -top {top} -check; flatten; "
    "proc; opt -fast; techmap; opt -fast; abc -g AND; opt_clean; stat; ltp -noff"
)


def _ppa(workdir: Path, vfile: str, top: str, timeout: int) -> Optional[Tuple[int, int]]:
    """Return (aig_area, aig_depth) or None on synthesis failure."""
    rc, out = _run_yosys(workdir, _PPA_FLOW.format(f=vfile, top=top), timeout)
    if rc != 0:
        return None
    m_area = re.search(r"Number of cells:\s+(\d+)", out)
    m_depth = re.search(r"Longest topological path in \S+ \(length=(\d+)\)", out)
    if not m_area or not m_depth:
        return None
    return int(m_area.group(1)), int(m_depth.group(1))


_EQUIV_FLOW = (
    "read_verilog -sv ref.v; hierarchy -top {ref}; proc; opt_clean; flatten; "
    "rename {ref} gold; design -stash gold; "
    "read_verilog -sv cand.v; hierarchy -top {top}; proc; opt_clean; flatten; "
    "rename {top} gate; design -stash gate; "
    "design -copy-from gold -as gold gold; design -copy-from gate -as gate gate; "
    "equiv_make gold gate equiv; hierarchy -top equiv; "
    "equiv_simple; equiv_induct; equiv_status -assert"
)


def _check_equivalence(workdir: Path, top: str, ref: str, timeout: int) -> Tuple[bool, str]:
    rc, out = _run_yosys(workdir, _EQUIV_FLOW.format(top=top, ref=ref), timeout)
    proven = rc == 0 and "Equivalence successfully proven" in out
    return proven, out


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


def _reference_ppa(problem: Dict, timeout: int) -> Optional[Tuple[int, int]]:
    key = problem["design_name"]
    ref_hash = hashlib.sha1(problem["reference"].encode("utf-8")).hexdigest()[:12]
    cache_file = _ref_cache_path(problem)
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    entry = cache.get(key)
    if entry and entry.get("hash") == ref_hash:
        return entry["area"], entry["depth"]

    with tempfile.TemporaryDirectory(prefix="rtllm_ref_") as tmp:
        wd = Path(tmp)
        (wd / "ref.v").write_text(problem["reference"], encoding="utf-8")
        ppa = _ppa(wd, "ref.v", problem["ref_module"], timeout)
    if ppa is None:
        return None
    cache[key] = {"hash": ref_hash, "area": ppa[0], "depth": ppa[1]}
    try:
        tmp_path = cache_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_file)
    except Exception:
        pass
    return ppa


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

        # --- Gate 2: formal combinational equivalence (the hardened gate) ---
        equivalent, equiv_out = _check_equivalence(wd, top, ref_mod, timeout)
        if not equivalent:
            note = ("passes the finite testbench but is NOT logically equivalent "
                    "(testbench-overfit / edge-case bug)") if tb_pass else \
                   "is not functionally correct"
            return {
                "combined_score": 0.0,
                "public": {"stage": "equivalence", "tb_pass": tb_pass,
                           "tb_failures": failures, "equivalent": False},
                "private": {"equiv_tail": equiv_out[-1500:]},
                "text_feedback": f"Design {note}. It must be logically identical to "
                                 f"the specified function for ALL inputs.",
            }

        # --- PPA: candidate vs reference on the identical AIG flow ---
        cand_ppa = _ppa(wd, "cand.v", top, timeout)
        ref_ppa = _reference_ppa(problem, timeout)
        if cand_ppa is None or ref_ppa is None:
            return {
                "combined_score": 0.0,
                "public": {"stage": "synthesis", "tb_pass": tb_pass,
                           "equivalent": True},
                "private": {},
                "text_feedback": "Correct & equivalent, but synthesis (yosys) failed "
                                 "to map the design. Use synthesizable constructs.",
            }

        a_c, d_c = cand_ppa
        a_r, d_r = ref_ppa
        area_ratio = a_r / a_c if a_c else 0.0
        delay_ratio = d_r / d_c if d_c else 0.0
        speedup = (area_ratio * delay_ratio) ** 0.5
        score = 100.0 * speedup

        verdict = "beats" if score > 100.0 else ("matches" if score == 100.0 else "below")
        feedback = (
            f"Correct (formally equivalent). PPA vs reference: "
            f"area {a_c} vs {a_r} (x{area_ratio:.2f}), "
            f"depth {d_c} vs {d_r} (x{delay_ratio:.2f}); "
            f"speedup={speedup:.3f} -> score {score:.1f} ({verdict} the human reference). "
            f"Reduce AIG area (cell count) and logic depth further while staying equivalent."
        )
        return {
            "combined_score": score,
            "public": {
                "stage": "pass", "tb_pass": tb_pass, "equivalent": True,
                "area": a_c, "depth": d_c,
                "ref_area": a_r, "ref_depth": d_r,
                "area_ratio": area_ratio, "delay_ratio": delay_ratio,
                "speedup": speedup,
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

    correct = metrics["public"].get("equivalent", False)
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
