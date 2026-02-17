# Julia Prime Counting Example

Math optimization task in Julia: count how many primes are `<= n` for each query.

## Files

- `initial.jl`: seed Julia implementation with EVOLVE markers.
- `evaluate.py`: Python evaluator that runs Julia program, validates outputs, writes `metrics.json` and `correct.json`.
- `run_evo.py`: async Shinka run config with small resource limits (`language="julia"`).

## Optimized Metric

The evaluator optimizes `combined_score`:

- `combined_score = max(0.0, 100.0 * accuracy - runtime_seconds)`
- `accuracy`: exact-match rate across 10 fixed prime-count queries.
- Correctness dominates: any wrong answers reduce accuracy and score sharply.
- Runtime is a penalty term once correctness is high.

## State Of The Art

- Algorithmic SOTA for repeated `pi(n)` queries is sieve-based precomputation (e.g., Eratosthenes / segmented sieve variants).
- In this repo/example, the current reference `initial.jl` is fully correct and scores about `97.5-98.5` on cold local runs measured on **February 16, 2026**.
- The theoretical score ceiling is `100.0` (perfect accuracy with zero runtime overhead).

## Requirements

- Julia installed and available on `PATH` as `julia`.
- Python environment with `shinka` installed.

## Run

From repo root:

```bash
cd examples/julia_prime_counting
python evaluate.py --program_path initial.jl --results_dir results/manual_eval
python run_evo.py
```

## Notes

- Evaluator sends fixed integer queries to the Julia program and expects one integer answer per line.
- Correctness is strict; score rewards correctness first, then runtime.
