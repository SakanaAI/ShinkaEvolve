"""Evolve one RTLLM design for PPA under a fixed spec.

Mirrors the other ShinkaEvolve examples (circle_packing, etc.): a YAML config holds
db_config + evo_config, and task_sys_msg is set here. Out of the box this evolves the
bundled adder_8bit design (committed initial.sv + example.jsonl, from RTLLM v2.0 under
MIT). To run any other RTLLM design, extract it from a clone (see extract_dataset.py)
and select it via RTLLM_DESIGN / RTLLM_PROBLEM_FILE.

    python run_evo.py                                                  # bundled adder_8bit
    RTLLM_DESIGN=adder_32bit RTLLM_PROBLEM_FILE=problems/rtllm.jsonl python run_evo.py
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

HERE = Path(__file__).resolve().parent

TASK_SYS_MSG = (
    "You are an expert digital design engineer optimizing Verilog RTL for physical "
    "quality (area and timing). The module's FUNCTION is fixed and checked by formal "
    "equivalence / the RTLLM testbench: your design must be correct for the specified "
    "function. Subject to that, minimize post-synthesis area (um^2) and logic depth. "
    "Explore stronger microarchitectures (carry-lookahead / parallel-prefix adders such "
    "as Kogge-Stone or Brent-Kung; Booth encoding and Wallace/Dadda trees for "
    "multipliers; balanced reduction trees). The module header and port declaration are "
    "FIXED and appear OUTSIDE the EVOLVE-BLOCK markers: the interface (port names AND "
    "widths) is part of the spec and must not change — optimize ONLY the implementation "
    "inside the markers. Use synthesizable Verilog only. Preserve the EVOLVE-BLOCK "
    "markers.\n\n"
    "CRITICAL — CYCLE-ACCURATE EQUIVALENCE. Your design must produce identical outputs "
    "to the reference for EVERY possible input SEQUENCE, on EVERY clock cycle — not "
    "merely pass the finite testbench. The following are INVALID even if they pass the "
    "testbench, and will be rejected by formal re-verification:\n"
    "  - removing flip-flops / registers or any sequential state the reference keeps;\n"
    "  - changing the latency or the cycle on which any output becomes valid;\n"
    "  - converting a registered output to combinational (or vice-versa);\n"
    "  - relying on an input being held constant across cycles when the reference does "
    "not require it (e.g. dropping an input latch).\n"
    "Optimize the LOGIC, not the timing contract: better arithmetic structures, smaller "
    "state encodings, right-sized counters, and provably-equivalent transforms (e.g. an "
    "explicit FSM rewritten as an equivalent shift-register + comparator, or a shift-add "
    "loop replaced by a single behavioral operator) are encouraged. Preserve the exact "
    "clocked I/O behavior of the reference."
)


def design_spec(design: str, problem_file: str) -> str:
    """The design's fixed-function spec (the RTLLM description), with a note that the
    suggested implementation is only one option."""
    for line in open(HERE / problem_file, encoding="utf-8"):
        p = json.loads(line)
        if p["design_name"] == design:
            return (
                "\n\n--- TARGET MODULE (fixed function) ---\n"
                + p.get("description", "").strip()
                + "\n\nThe function above is fixed, but the implementation it suggests is "
                "only one option: use any correct, synthesizable microarchitecture."
            )
    return ""


def main(config_path: str):
    with open(HERE / config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    design = os.environ.get("RTLLM_DESIGN", "adder_8bit")
    problem_file = os.environ.get("RTLLM_PROBLEM_FILE", "example.jsonl")
    # Propagate the selection to the evaluate.py subprocess (LocalJobConfig spawns it
    # with this process's environment).
    os.environ["RTLLM_DESIGN"] = design
    os.environ["RTLLM_PROBLEM_FILE"] = problem_file

    # Seed: a design extracted from an RTLLM clone (seeds/<design>/initial.sv) if present,
    # else the bundled committed seed (initial.sv, adder_8bit).
    seed = HERE / "seeds" / design / "initial.sv"
    if not seed.exists():
        seed = HERE / "initial.sv"

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG + design_spec(design, problem_file)
    config["evo_config"]["init_program_path"] = str(seed)
    config["evo_config"]["results_dir"] = str(HERE / "results" / design / "evo_results")

    evo_config = EvolutionConfig(**config["evo_config"])
    db_config = DatabaseConfig(**config["db_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py")

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs", 2),
        max_proposal_jobs=config.get("max_proposal_jobs", 2),
    )
    runner.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config_path", nargs="?", default="shinka.yaml",
                    help="YAML config with db_config + evo_config (default: shinka.yaml)")
    main(ap.parse_args().config_path)
