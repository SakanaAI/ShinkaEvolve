"""Run ShinkaEvolve on all CVDP benchmark problems.

Iterates over every problem in the JSONL dataset, generates a per-problem
initial seed with the correct module name and spec, and launches a ShinkaEvolve
evolution run for each.

Usage:
    python run_all.py                                    # All problems, defaults
    python run_all.py --workers 2 --generations 30      # Parallel, more gens
    python run_all.py --problems cvdp_copilot_lfsr_0001  # Specific problems
    python run_all.py --eval-file problems/cvdp_full.jsonl

Requires:
    - problems/cvdp_lite.jsonl or cvdp_full.jsonl (run download_dataset.py for full set)
    - Docker installed and running
    - CVDP simulation image built (nvidia/cvdp-sim:v1.0.0)
    - Azure/OpenAI API keys configured
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def load_problems(jsonl_path: Path) -> list[dict]:
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _extract_module_name(problem: dict) -> str:
    env_content = problem["harness"]["files"].get("src/.env", "")
    for line in env_content.splitlines():
        if line.strip().startswith("TOPLEVEL") and "LANG" not in line:
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "design"


def generate_seed(problem: dict, output_dir: Path) -> Path:
    """Generate initial.sv for a CVDP problem from its spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "initial.sv"

    module_name = _extract_module_name(problem)
    spec = problem["input"]["prompt"]

    spec_lines = "\n".join(f"// {line}" for line in spec.splitlines())

    content = (
        f"// Problem: {problem['id']}\n"
        f"// Module: {module_name}\n"
        f"//\n"
        f"// === SPECIFICATION ===\n"
        f"{spec_lines}\n"
        f"// === END SPECIFICATION ===\n"
        f"\n"
        f"// EVOLVE-BLOCK-START\n"
        f"module {module_name}(\n"
        f"  // TODO: add ports per specification above\n"
        f");\n"
        f"  // TODO: implement per specification\n"
        f"endmodule\n"
        f"// EVOLVE-BLOCK-END\n"
    )
    seed_path.write_text(content, encoding="utf-8")
    return seed_path


def run_single_problem(
    problem_id: str,
    jsonl_path: str,
    generations: int,
    models: list[str],
    results_base: str,
    seed_path: str,
    task_msg_path: str,
) -> dict:
    """Run evolution for a single CVDP problem in a subprocess."""
    env = {
        **os.environ,
        "CVDP_PROBLEM_ID": problem_id,
        "CVDP_PROBLEM_FILE": jsonl_path,
        "PYTHONIOENCODING": "utf-8",
    }

    this_dir = str(Path(__file__).resolve().parent)
    repo_root = str(Path(__file__).resolve().parent.parent.parent)

    script = f"""
import sys, os
sys.path.insert(0, r"{repo_root}")

from pathlib import Path
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

os.chdir(r"{this_dir}")

task_msg = Path(r"{task_msg_path}").read_text(encoding="utf-8")

job_conf = LocalJobConfig(eval_program_path="evaluate.py")
db_conf = DatabaseConfig(num_islands=1, archive_size=10)
evo_conf = EvolutionConfig(
    init_program_path=r"{seed_path}",
    language="verilog",
    num_generations={generations},
    llm_models={models},
    llm_kwargs=dict(temperatures=[0.3, 0.7], max_tokens=4096),
    embedding_model=None,
    results_dir=r"{results_base}/{problem_id}/evo_results",
    task_sys_msg=task_msg,
    use_text_feedback=True,
)
runner = ShinkaEvolveRunner(
    evo_config=evo_conf,
    job_config=job_conf,
    db_config=db_conf,
    max_evaluation_jobs=2,
    max_proposal_jobs=2,
)
runner.run()
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        cwd=str(Path(__file__).resolve().parent),
    )

    return {
        "problem_id": problem_id,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve on all CVDP problems")
    parser.add_argument(
        "--eval-file", type=str, default="problems/cvdp_lite.jsonl",
        help="Path to JSONL problem file (default: problems/cvdp_lite.jsonl)",
    )
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Specific problem IDs to run (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel evolution runs (default: 2)",
    )
    parser.add_argument(
        "--generations", type=int, default=30,
        help="Generations per problem (default: 30)",
    )
    parser.add_argument(
        "--models", nargs="*",
        default=["azure-gpt-4-1-mini", "azure-gpt-5-4-mini", "azure-deepseek-v4-flash"],
        help="LLM models to use (default: gpt-4.1-mini + gpt-5-4-mini + deepseek-v4-flash)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results_all",
        help="Base directory for all results (default: results_all)",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = Path(__file__).resolve().parent / eval_path

    if not eval_path.exists():
        print(f"Dataset not found: {eval_path}")
        print("Run: python download_dataset.py")
        sys.exit(1)

    problems = load_problems(eval_path)
    print(f"Loaded {len(problems)} problems from {eval_path}")

    if args.problems:
        problems = [p for p in problems if p["id"] in args.problems]
        if not problems:
            print(f"No matching problems found for: {args.problems}")
            sys.exit(1)

    results_base = Path(args.results_dir)
    if not results_base.is_absolute():
        results_base = Path(__file__).resolve().parent / results_base

    seeds_dir = results_base / "_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(problems)} problems x {args.generations} generations")
    print(f"Workers: {args.workers}, Models: {args.models}")
    print(f"Results: {results_base}")
    print("Generating per-problem seeds...")

    problem_seeds = {}
    problem_msg_paths = {}
    for problem in problems:
        pid = problem["id"]
        problem_dir = seeds_dir / pid
        seed_path = generate_seed(problem, problem_dir)
        problem_seeds[pid] = str(seed_path)

        spec = problem["input"]["prompt"]
        module_name = _extract_module_name(problem)
        task_msg = (
            "You are an expert digital design engineer specializing in Verilog/SystemVerilog RTL. "
            "You are evolving a hardware module to pass a CocoTB testbench. "
            f"The target module name is '{module_name}'. "
            "You MUST implement the correct module name and port interface as specified below. "
            "Use synthesizable constructs only. Avoid latches. "
            "Use non-blocking assignments (<=) in clocked always blocks. "
            "Preserve EVOLVE-BLOCK markers.\n\n"
            f"=== MODULE SPECIFICATION ===\n{spec}\n=== END SPECIFICATION ==="
        )
        msg_path = problem_dir / "task_msg.txt"
        msg_path.write_text(task_msg, encoding="utf-8")
        problem_msg_paths[pid] = str(msg_path)

    print(f"Generated {len(problem_seeds)} seeds")
    print()

    completed = 0
    failed = 0
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for problem in problems:
            pid = problem["id"]
            future = pool.submit(
                run_single_problem,
                pid,
                str(eval_path),
                args.generations,
                args.models,
                str(results_base),
                problem_seeds[pid],
                problem_msg_paths[pid],
            )
            futures[future] = pid

        for future in as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["returncode"] == 0:
                    completed += 1
                    print(f"  [{completed + failed}/{len(problems)}] {pid} - OK")
                else:
                    failed += 1
                    print(f"  [{completed + failed}/{len(problems)}] {pid} - FAILED")
                    if result["stderr_tail"]:
                        print(f"    {result['stderr_tail'][:200]}")
            except Exception as e:
                failed += 1
                results.append({"problem_id": pid, "returncode": -1, "error": str(e)})
                print(f"  [{completed + failed}/{len(problems)}] {pid} - ERROR: {e}")

    print(f"\nDone: {completed} succeeded, {failed} failed out of {len(problems)}")

    summary_path = results_base / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
