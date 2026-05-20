# Wolfram GCD-Sum Example

Program optimization task in Wolfram Language: compute
`S(N) = sum_{i,j in 1..N} GCD(i, j)` for `N = 300`, as fast as possible.
The seed implementation is intentionally deoptimized; the evolutionary
loop is rewarded for speeding it up while keeping the answer identical.

## Files

- `initial.wl`: seed Wolfram implementation with EVOLVE markers.
- `evaluate.py`: Python evaluator that runs the candidate Wolfram program,
  validates its output against the known answer, and scores by runtime.
- `run_evo.py`: async Shinka run config (`language="wolfram"`).

## Optimized Metric

The evaluator optimizes `combined_score`:

- `combined_score = baseline_time_ms / median(candidate_time_ms)` if the
  candidate returns the correct integer (`336784`), else `-1.0`.
- `baseline_time_ms` is calibrated on every evaluator run by timing the
  deoptimized seed (`initial.wl`) through the same harness as the
  candidate — `RepeatedTiming` inside `TimeConstrained` — so the baseline
  and the candidate are measured identically on the host machine.
- The optimization space is open-ended: the canonical idiomatic form
  (`Total[Outer[GCD, Range[n], Range[n]], 2]`), and beyond it the Pillai
  / Euler-totient identity `sum_{i,j} GCD(i,j) = sum_d phi(d) *
  floor(N/d)^2`, which takes the algorithm from `O(N^2)` to `O(N)`.

## State Of The Art

- Pure brute force is `O(N^2)`.
- Vectorized brute force in Wolfram (`Outer[GCD, ...]`) stays
  algorithmically `O(N^2)` but dispatches to the C kernel rather than
  interpreting a loop.
- The Pillai identity drops the asymptotic cost to `O(N)` and dominates
  for any non-trivial `N`. Further micro-optimizations (`Quotient` vs
  `Floor[n/r]`, dot-product fused multiply-add) tighten the constant
  factor.

## Requirements

- Wolfram Engine for Developers (free) or Mathematica, with `wolframscript`
  available on `PATH` (or set `WOLFRAMSCRIPT_BIN` to its absolute path).
- Python environment with `shinka` installed.
- Credentials for whichever LLM provider `run_evo.py` is configured to use.
  The default is `gemini-2.5-flash-lite`, which reads `GEMINI_API_KEY`
  (or `GOOGLE_API_KEY`) from the environment.

## Run

From repo root:

```bash
cd examples/wolfram_gcd_sum
python evaluate.py --program_path initial.wl --results_dir results/manual_eval
python run_evo.py
```

## Notes

- The seed deliberately stacks three orthogonal anti-patterns (nested For
  loops, `Range[N]` recomputed in the inner loop, hand-written Euclidean
  GCD instead of the built-in). The LLM can attack any subset per
  generation, so progress compounds.
- The evaluator runs each candidate `NUM_REPS=3` times and takes the
  median; candidates that exceed a per-run timeout are scored as failures.
  Override the Wolfram-side timeout per run with
  `WOLFRAM_GCD_MAX_SECONDS=<float>` (default 30s) — useful on slow
  hardware where the seed itself uses most of the headroom.
- The EVOLVE-BLOCK markers in Wolfram use `(* ... *)` block comments;
  the patch applier validates them after each mutation to catch the case
  where the LLM accidentally smuggles both markers into a single comment.
