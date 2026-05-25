# Go Collatz Steps Example

Algorithmic optimization task in Go: compute the Collatz stopping time for each
input integer.

Files:

- `initial.go`: seed Go implementation with EVOLVE markers.
- `evaluate.py`: Python evaluator that runs `go run`, validates outputs, writes `metrics.json` and `correct.json`.
- `run_evo.py`: async Shinka run config with `language="go"`.

Prerequisites:

- Go installed and available on `PATH` as `go`.
- ShinkaEvolve installed in the active Python environment.

Run a manual evaluation:

```bash
python evaluate.py --program_path initial.go --results_dir results/manual_eval
```

Run evolution:

```bash
python run_evo.py
```

The evaluator sends fixed positive integers to the Go program and expects one
integer stopping time per line.
