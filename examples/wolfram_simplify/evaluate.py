from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from shinka.utils.wolfram import (
    build_wolframscript_argv,
    is_wolframscript_available,
    wolframscript_bin,
)

# Per-call subprocess timeout for the whole harness run; the per-call
# TimeConstrained inside Wolfram is much tighter (50ms per task call).
HARNESS_TIMEOUT_S = 60
ABS_TOL = 1e-9
REL_TOL = 1e-9

THIS_DIR = Path(__file__).resolve().parent
SEED_PROGRAM = THIS_DIR / "initial.wl"
HARNESS_SCRIPT = THIS_DIR / "harness.wl"


def _outputs_match(seed, candidate):
    """Per-task per-input numeric comparison with float tolerance. Returns
    (ok, first_mismatch_description). Int/float are compared numerically
    (a simplified candidate may legitimately return integer `1` where the
    seed's Sin/Cos expansion evaluates to a near-1.0 Real). Non-numeric
    values (timeout strings, undefined-symbol fallbacks) must match
    exactly."""
    if isinstance(seed, bool) != isinstance(candidate, bool):
        return False, f"bool/numeric mismatch: seed={seed!r} cand={candidate!r}"
    if isinstance(seed, (int, float)) and isinstance(candidate, (int, float)):
        if math.isclose(float(seed), float(candidate), rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return True, ""
        return False, f"numeric: seed={seed!r} cand={candidate!r}"
    if isinstance(seed, list) and isinstance(candidate, list):
        if len(seed) != len(candidate):
            return False, f"length mismatch: seed={len(seed)} cand={len(candidate)}"
        for i, (a, b) in enumerate(zip(seed, candidate)):
            ok, why = _outputs_match(a, b)
            if not ok:
                return False, f"[{i}] {why}"
        return True, ""
    if type(seed) is not type(candidate):
        return False, f"type mismatch: seed={type(seed).__name__} cand={type(candidate).__name__}"
    if seed != candidate:
        return False, f"value: seed={seed!r} cand={candidate!r}"
    return True, ""


def _run_harness(candidate_path):
    """Invoke harness.wl via wolframscript. Returns the parsed JSON dict
    (with 'seed' and 'candidate' analyses) or raises on subprocess failure."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        argv = build_wolframscript_argv(
            [
                "-file",
                str(HARNESS_SCRIPT),
                str(SEED_PROGRAM),
                candidate_path,
                out_path,
            ]
        )
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=HARNESS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip()[-300:]
            return {"_subprocessError": f"rc={proc.returncode} stderr={stderr_tail}"}
        if not Path(out_path).exists():
            return {"_subprocessError": "harness produced no output file"}
        try:
            return json.loads(Path(out_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"_subprocessError": f"JSON parse: {exc}"}
    finally:
        Path(out_path).unlink(missing_ok=True)


def main(program_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    metrics = {"combined_score": -1.0, "public": {}, "private": {}}
    correct = False
    error = ""

    if not is_wolframscript_available():
        error = (
            f"`{wolframscript_bin()}` not found. Install Wolfram Engine or "
            f"Mathematica, or set WOLFRAMSCRIPT_BIN to the binary's absolute path."
        )
    else:
        data = _run_harness(program_path)

        if "_subprocessError" in data:
            error = data["_subprocessError"]
        elif "seed" not in data or "candidate" not in data:
            error = "harness output missing 'seed' or 'candidate'"
        else:
            seed = data["seed"]
            cand = data["candidate"]

            if cand.get("parseError"):
                error = f"candidate parse error: {cand['parseError']}"
            elif cand.get("blocklisted"):
                error = (
                    f"candidate uses disallowed head(s): "
                    f"{', '.join(cand['blocklisted'])}"
                )
            elif seed.get("leafCount") in (None, "NA") or cand.get("leafCount") in (None, "NA"):
                error = "missing LeafCount for seed or candidate"
            else:
                seed_outputs = seed.get("outputs", {})
                cand_outputs = cand.get("outputs", {})
                mismatch = None
                for task in ("t1", "t2", "t3", "t4", "t5"):
                    if task not in seed_outputs or task not in cand_outputs:
                        mismatch = f"{task}: missing outputs"
                        break
                    s_out = seed_outputs[task].get("outputs", [])
                    c_out = cand_outputs[task].get("outputs", [])
                    ok, why = _outputs_match(s_out, c_out)
                    if not ok:
                        mismatch = f"{task}: {why}"
                        break
                if mismatch:
                    error = f"output mismatch: {mismatch}"
                else:
                    correct = True

            if correct:
                seed_lc = int(seed["leafCount"])
                cand_lc = int(cand["leafCount"])
                # Guard against pathological cand_lc=0 — would never happen
                # for a non-empty EVOLVE-BLOCK, but cheap to defend against.
                ratio = seed_lc / max(cand_lc, 1)
                metrics = {
                    "combined_score": ratio,
                    "public": {
                        "seed_leafcount": seed_lc,
                        "candidate_leafcount": cand_lc,
                        "leafcount_ratio": round(ratio, 4),
                    },
                    "private": {},
                    "text_feedback": (
                        f"All 5 tasks reproduced the seed's outputs.\n"
                        f"Seed LeafCount: {seed_lc}. Candidate LeafCount: {cand_lc}.\n"
                        f"Score: {ratio:.4f}x (higher is better)."
                    ),
                }

    if not correct:
        metrics = {
            "combined_score": -1.0,
            "public": {"error": error[:300]},
            "private": {},
            "text_feedback": (
                f"Candidate failed: {error}\n"
                f"Each function tN[...] must return the same value as the "
                f"deoptimized seed for every test input. `Simplify` and "
                f"`FullSimplify` are disallowed."
            ),
        }

    Path(results_dir, "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=4), encoding="utf-8"
    )
    Path(results_dir, "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )

    print(f"Evaluated program: {program_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Correct: {correct}")
    if error:
        print(f"Error: {error}")
    print(f"Combined score: {metrics.get('combined_score', -1.0):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wolfram simplify program")
    parser.add_argument("--program_path", type=str, default="initial.wl")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
