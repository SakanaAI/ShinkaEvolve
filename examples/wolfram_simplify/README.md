# Wolfram Simplify Example

Expression-simplification task in Wolfram Language: rewrite a small suite
of mathematically-equivalent functions into smaller (lower-LeafCount)
forms while preserving behavior on numeric test inputs.

## Files

- `initial.wl`: seed Wolfram program with five tasks (`t1` .. `t5`)
  inside an EVOLVE-BLOCK. Each task is intentionally verbose.
- `harness.wl`: evaluator-invoked Wolfram script that reads the seed and
  the candidate, parses each EVOLVE-BLOCK to a held form, computes
  `LeafCount`, scans for disallowed heads, and runs every tN under a
  per-call `TimeConstrained` budget.
- `evaluate.py`: Python evaluator that invokes `harness.wl`, compares
  per-task outputs with float tolerance, and writes `metrics.json` /
  `correct.json`.
- `run_evo.py`: async Shinka run config (`language="wolfram"`).

## Optimized Metric

The evaluator optimizes `combined_score`:

- `combined_score = seed_total_LeafCount / candidate_total_LeafCount` if
  every `tN` reproduces the seed's output on every test input, else `-1.0`.
- `LeafCount` is taken on the *parsed, held* form of the entire
  EVOLVE-BLOCK contents (`ToExpression[..., InputForm, Hold]`), so
  helper definitions count toward the candidate's total and wrapping
  unevaluated machinery (`Evaluate[...]`, `FullSimplify[...]`) inflates
  the count rather than shrinking it.
- The heads `Simplify` and `FullSimplify` are blocklisted: the prompt
  forbids them and the harness rejects any candidate that contains
  either. Other heads (`Reduce`, `Factor`, `Apart`, etc.) are not
  explicitly blocked because using them would just add leaves and a
  per-call wall-time cap (~50ms) makes any runtime-symbolic-engine
  punt fail anyway.

## State Of The Art

- `LeafCount` of the deoptimized seed is the baseline.
- Hand-written near-optimal rewrites of `t1`..`t5` reach roughly a 3x
  reduction on this seed. Different choices of helper inlining and
  combinator usage shift the score, so there is room for the loop to
  search.

## Requirements

- Wolfram Engine for Developers (free) or Mathematica, with `wolframscript`
  available on `PATH` (or set `WOLFRAMSCRIPT_BIN` to its absolute path).
- Python environment with `shinka` installed.
- Credentials for whichever LLM provider `run_evo.py` is configured to
  use. The default is `claude-cli/sonnet`, which authenticates through
  the local Claude Code OAuth login.

## Run

From repo root:

```bash
cd examples/wolfram_simplify
python evaluate.py --program_path initial.wl --results_dir results/manual_eval
python run_evo.py
```

## Notes

- Each of `t1`..`t5` exercises a different shape: a single algebraic
  expression (`t1`), a trig identity (`t2`), an imperative `Module`
  with a known closed form (`t3`), a top-level sequence of helpers a
  caller can inline (`t4`), and a verbose accumulator with a built-in
  replacement (`t5`). The total score sums LeafCounts across the whole
  block, so the LLM gets a continuous gradient by attacking any subset.
- Test inputs are fixed numeric batches per task, chosen to avoid
  undefined boundaries (e.g. T5's empty-list case is excluded — the
  seed would divide by zero there).
- Float-tolerance comparison is `math.isclose(rel=1e-9, abs=1e-9)`.
  Integer/float crossovers are accepted numerically (a simplified T2
  legitimately returns integer `1` where the seed's Sin/Cos expansion
  evaluates to a near-1.0 Real).
- The EVOLVE-BLOCK markers in Wolfram use `(* ... *)` block comments;
  the patch applier validates them after each mutation.
