#!/usr/bin/env python3
"""
Runner for the lean_math ShinkaEvolve task.

Usage:
    # Small test run (30 generations, auto results dir):
    python run_evo.py

    # Continuous run (1000 generations, resumable):
    python run_evo.py --config_path shinka_continuous.yaml

    # Resume a previous run (same results_dir in yaml):
    python run_evo.py --config_path shinka_continuous.yaml
"""

import argparse
import os
import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

LEAN_MATH_SYS_MSG = """\
You are an expert in Lean 4 theorem proving and formal mathematics.
Your task is to improve the `get_proof_tactics()` function so it returns \
correct Lean 4 tactics that compile without errors.

The score equals the number of theorems whose proof compiles successfully \
(max 15). Higher score = better.

Key Lean 4 tactics available in core Lean 4.29 (NO Mathlib):
  - simp           : powerful simplifier; solves most basic arithmetic goals
  - omega          : decision procedure for linear arithmetic over Nat/Int
  - rfl            : proves a = a by reflexivity (definitional equality only)
  - exact <term>   : provide an explicit proof term
  - apply <lemma>  : apply a named lemma and generate subgoals
  - constructor    : split a conjunction (∧) goal into two subgoals
  - intro h        : introduce a hypothesis h

Useful proof terms and lemmas (no import needed):
  Nat.add_comm n m        : n + m = m + n
  Nat.add_assoc n m k     : (n + m) + k = n + (m + k)
  Nat.left_distrib n m k  : n * (m + k) = n * m + n * k
  Nat.zero_le n           : 0 ≤ n
  Nat.lt_succ_self n      : n < n + 1
  Nat.le_add_right n m    : n ≤ n + m
  Or.comm.mp h            : turns p ∨ q into q ∨ p
  Or.inl h                : proves p ∨ q from h : p
  Or.inr h                : proves p ∨ q from h : q

RULES:
  - Do NOT use `sorry` — proofs containing sorry score 0.
  - Do NOT use `ring` — it requires Mathlib, which is not loaded.
  - Multi-step tactics: separate with `<;>` or newlines indented under `by`.
  - For conjunction goals use `exact ⟨h1, h2⟩` or `constructor; exact h1; exact h2`.

Study the failed problems in the text_feedback and fix them one at a time.\
"""


def main(config_path: str) -> None:
    task_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(task_dir, config_path), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evo_cfg_dict = config["evo_config"]
    evo_cfg_dict["task_sys_msg"] = LEAN_MATH_SYS_MSG

    # Resolve init_program_path relative to this file's directory
    if not os.path.isabs(evo_cfg_dict.get("init_program_path", "")):
        evo_cfg_dict["init_program_path"] = os.path.join(
            task_dir, evo_cfg_dict["init_program_path"]
        )

    evo_config = EvolutionConfig(**evo_cfg_dict)
    job_config = LocalJobConfig(
        eval_program_path=os.path.join(task_dir, "evaluate.py"),
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs", 1),
        max_proposal_jobs=config.get("max_proposal_jobs", 1),
        max_db_workers=config.get("max_db_workers", 2),
        verbose=True,
    )

    print(f"[lean_math] Starting evolution — config: {config_path}")
    print(f"[lean_math] Lean binary: {os.path.expanduser('~/.elan/bin/lean')}")
    print(f"[lean_math] Task dir:   {task_dir}")
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lean_math ShinkaEvolve runner")
    parser.add_argument(
        "--config_path",
        default="shinka_small.yaml",
        help="YAML config file relative to this script (default: shinka_small.yaml)",
    )
    args = parser.parse_args()
    main(args.config_path)
