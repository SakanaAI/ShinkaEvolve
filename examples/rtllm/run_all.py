"""Run ShinkaEvolve on RTLLM designs to optimize PPA under a fixed spec.

Each design starts from its (correct) RTLLM reference as the seed and evolves
toward lower AIG area + logic depth, gated by formal equivalence. Fitness is a
"speedup" vs the reference: reference = 100, beating it > 100.

Usage:
    python run_all.py                                   # all designs in the JSONL
    python run_all.py --designs adder_8bit multi_8bit   # subset
    python run_all.py --generations 30 --workers 3 \
        --models openrouter/anthropic/claude-sonnet-4.5

Requires:
    - problems/rtllm_proto.jsonl + seeds/ (run extract_dataset.py first)
    - iverilog on PATH; yosys native or the hdlc/yosys docker image
    - OPENROUTER_API_KEY (or other provider keys) in the repo-root .env
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

TASK_SYS_MSG = (
    "You are an expert digital design engineer optimizing Verilog RTL for "
    "physical quality (area and timing). The module's FUNCTION is fixed and "
    "checked by FORMAL EQUIVALENCE against a golden reference: your design must "
    "be logically identical for ALL inputs, not merely pass a testbench. Subject "
    "to that, minimize post-synthesis AIG area (cell count) and logic depth. "
    "Explore stronger microarchitectures (carry-lookahead / parallel-prefix "
    "adders such as Kogge-Stone or Brent-Kung; Booth encoding and Wallace/Dadda "
    "trees for multipliers; balanced reduction trees) and let the synthesizer "
    "optimize. Keep the module name and port interface unchanged. Use "
    "synthesizable Verilog only. Preserve the EVOLVE-BLOCK markers."
)


def load_designs(jsonl_path: Path) -> list[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_single(design: str, jsonl_path: str, generations: int,
               models: list[str], results_base: str) -> dict:
    env = {
        **os.environ,
        "RTLLM_DESIGN": design,
        "RTLLM_PROBLEM_FILE": jsonl_path,
        "PYTHONIOENCODING": "utf-8",
    }
    seed = Path(results_base) / design / "initial.sv"
    script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")          # force THIS fork (has Verilog support)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(r"{REPO_ROOT / '.env'}", override=True)
import os
os.chdir(r"{THIS_DIR}")

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_conf = LocalJobConfig(eval_program_path="evaluate.py")
db_conf = DatabaseConfig(num_islands=1, archive_size=12)
evo_conf = EvolutionConfig(
    init_program_path=r"{seed}",
    language="verilog",
    num_generations={generations},
    llm_models={models},
    llm_kwargs=dict(temperatures=[0.3, 0.7], max_tokens=6144),
    embedding_model=None,
    results_dir=r"{results_base}/{design}/evo_results",
    use_text_feedback=True,
    task_sys_msg={TASK_SYS_MSG!r},
)
runner = ShinkaEvolveRunner(
    evo_config=evo_conf, job_config=job_conf, db_config=db_conf,
    max_evaluation_jobs=2, max_proposal_jobs=2,
)
runner.run()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=7200, cwd=str(THIS_DIR),
    )
    return {
        "design": design,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-800:] if result.stdout else "",
        "stderr_tail": result.stderr[-800:] if result.stderr else "",
    }


def main():
    ap = argparse.ArgumentParser(description="Run ShinkaEvolve on RTLLM PPA designs")
    ap.add_argument("--problem-file", default="problems/rtllm_proto.jsonl")
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models", nargs="*",
                    default=["openrouter/anthropic/claude-sonnet-4.5"])
    ap.add_argument("--results-dir", default="results_all")
    args = ap.parse_args()

    jsonl = Path(args.problem_file)
    if not jsonl.is_absolute():
        jsonl = THIS_DIR / jsonl
    if not jsonl.exists():
        print(f"Dataset not found: {jsonl}\nRun: python extract_dataset.py")
        sys.exit(1)

    designs = load_designs(jsonl)
    if args.designs:
        designs = [d for d in designs if d["design_name"] in args.designs]
    names = [d["design_name"] for d in designs]
    if not names:
        print("No matching designs.")
        sys.exit(1)

    results_base = Path(args.results_dir)
    if not results_base.is_absolute():
        results_base = THIS_DIR / results_base

    # stage per-design seed into the results dir
    for d in designs:
        out = results_base / d["design_name"]
        out.mkdir(parents=True, exist_ok=True)
        src = THIS_DIR / "seeds" / d["design_name"] / "initial.sv"
        (out / "initial.sv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"RTLLM x ShinkaEvolve | {len(names)} designs x {args.generations} gens "
          f"| workers={args.workers} | models={args.models}")
    print(f"designs: {names}\nresults: {results_base}\n")

    results, done, failed = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_single, d["design_name"], str(jsonl),
                            args.generations, args.models, str(results_base)): d["design_name"]
                for d in designs}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                ok = r["returncode"] == 0
                done += ok
                failed += (not ok)
                print(f"  [{done+failed}/{len(names)}] {name} - {'OK' if ok else 'FAILED'}")
                if not ok and r["stderr_tail"]:
                    print(f"    {r['stderr_tail'][:300]}")
            except Exception as e:
                failed += 1
                results.append({"design": name, "error": str(e)})
                print(f"  [{done+failed}/{len(names)}] {name} - ERROR: {e}")

    (results_base / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDone: {done} ok, {failed} failed. Summary: {results_base/'summary.json'}")


if __name__ == "__main__":
    main()
