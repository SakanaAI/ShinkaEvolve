import os
import subprocess
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Fixed: problem definitions — NOT evolved
# ---------------------------------------------------------------------------
# 15 problems across 3 tiers. Each entry: (theorem statement, hypothesis bindings)
# The statement ends before `:= by`; the evaluator appends the tactic.
PROBLEMS: Dict[str, str] = {
    # Tier 1 — basic arithmetic (simp / rfl territory)
    "add_zero":     "theorem add_zero     (n : Nat)                         : n + 0 = n",
    "zero_add":     "theorem zero_add     (n : Nat)                         : 0 + n = n",
    "mul_one":      "theorem mul_one      (n : Nat)                         : n * 1 = n",
    "one_mul":      "theorem one_mul      (n : Nat)                         : 1 * n = n",
    "mul_zero":     "theorem mul_zero     (n : Nat)                         : n * 0 = 0",
    # Tier 2 — commutativity, associativity, order
    "add_comm":     "theorem add_comm     (n m : Nat)                       : n + m = m + n",
    "add_assoc":    "theorem add_assoc    (n m k : Nat)                     : (n + m) + k = n + (m + k)",
    "left_distrib": "theorem left_distrib (n m k : Nat)                     : n * (m + k) = n * m + n * k",
    "zero_le":      "theorem zero_le      (n : Nat)                         : 0 ≤ n",
    "lt_succ":      "theorem lt_succ      (n : Nat)                         : n < n + 1",
    "le_add_right": "theorem le_add_right (n m : Nat)                       : n ≤ n + m",
    # Tier 3 — propositional logic
    "and_intro":    "theorem and_intro    (p q : Prop) (h1 : p) (h2 : q)   : p ∧ q",
    "and_flip":     "theorem and_flip     (p q : Prop) (h : p ∧ q)         : q ∧ p",
    "or_comm_thm":  "theorem or_comm_thm  (p q : Prop) (h : p ∨ q)         : q ∨ p",
    "or_inl_thm":   "theorem or_inl_thm   (p : Prop)   (h : p)             : p ∨ False",
}

# Path to the Lean 4 binary installed via elan
_ELAN_BIN = os.path.expanduser("~/.elan/bin")
LEAN_BIN = os.path.join(_ELAN_BIN, "lean")


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------
def get_proof_tactics() -> Dict[str, str]:
    """
    Return a mapping: problem_name -> Lean 4 tactic string.

    The tactic is placed after `by` in:
        theorem <name> ... := by
          <tactic>

    Useful tactics available in core Lean 4.29 (no Mathlib):
      - simp            : simplification; handles most basic arithmetic
      - omega            : linear arithmetic over Nat/Int
      - rfl              : reflexivity (definitional equality)
      - exact <term>    : supply an explicit proof term
      - apply <lemma>   : apply a named lemma
      - constructor; exact h1; exact h2  : split a conjunction goal

    Useful terms / lemmas:
      Nat.add_comm, Nat.add_assoc, Nat.left_distrib,
      Nat.zero_le, Nat.lt_succ_self, Nat.le_add_right,
      Or.inl, Or.inr, Or.comm

    NOTE: `ring` requires Mathlib and is NOT available here.
    NOTE: `sorry` is blocked — a proof using sorry scores 0.
    """
    return {
        # Tier 1
        "add_zero":     "rfl",
        "zero_add":     "rfl",
        "mul_one":      "rfl",
        "one_mul":      "rfl",
        "mul_zero":     "rfl",
        # Tier 2 — rfl won't work here; intentionally wrong to give room to evolve
        "add_comm":     "rfl",
        "add_assoc":    "rfl",
        "left_distrib": "rfl",
        "zero_le":      "rfl",
        "lt_succ":      "rfl",
        "le_add_right": "rfl",
        # Tier 3
        "and_intro":    "rfl",
        "and_flip":     "rfl",
        "or_comm_thm":  "rfl",
        "or_inl_thm":   "rfl",
    }
# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-END
# ---------------------------------------------------------------------------


def _check_proof(problem_name: str, statement: str, tactic: str) -> Tuple[bool, str]:
    """Compile one Lean theorem and return (success, message)."""
    tactic = (tactic or "").strip()
    if not tactic or tactic == "sorry":
        return False, "blocked: sorry or empty tactic"

    lean_src = f"{statement} := by\n  {tactic}\n"
    env = {**os.environ, "PATH": _ELAN_BIN + os.pathsep + os.environ.get("PATH", "")}
    try:
        proc = subprocess.run(
            [LEAN_BIN, "--stdin"],
            input=lean_src,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        output = (stdout + "\n" + stderr).strip()

        if proc.returncode != 0:
            return False, output[:300]
        if "sorry" in output.lower():
            return False, "blocked: proof contains sorry"
        if "error" in stderr.lower():
            return False, stderr[:300]
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout (>20 s)"
    except FileNotFoundError:
        return False, f"lean binary not found at {LEAN_BIN}"
    except Exception as exc:
        return False, str(exc)[:200]


def run_experiment(run_idx: int = 0) -> dict:
    """
    Attempt every proof in PROBLEMS using the tactics from get_proof_tactics().
    Returns a result dict consumed by evaluate.py.
    """
    tactics = get_proof_tactics()
    solved = []
    failed = {}

    for name, statement in PROBLEMS.items():
        tactic = tactics.get(name, "sorry")
        ok, msg = _check_proof(name, statement, tactic)
        if ok:
            solved.append(name)
        else:
            failed[name] = msg

    return {
        "score": len(solved),
        "total": len(PROBLEMS),
        "solved": solved,
        "failed": failed,
    }
