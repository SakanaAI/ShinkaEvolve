# Circle Packing Example

Compact Shinka task: pack `n=26` circles in a unit square, maximize sum of radii.

## Ingredients

- `initial.py`: seed solution; exposes `run_packing()`.
- `evaluate.py`: validator + scorer; runs `run_packing`, checks geometry constraints, writes metrics/artifacts.
- `run_evo_async.py`: async evolution runner (uses top-level job concurrency keys from YAML).
- `run_evo.py`: sync evolution runner.
- `shinka_small.yaml`, `shinka_medium.yaml`, `shinka_long.yaml`: run profiles.
- `load_results.ipynb`: post-run analysis plots (incl. 2x3 dashboard).
- `viz_circles.ipynb`: geometry-focused circle layout visualization.

## Config Profiles

| Config | Intended Use | Core Shape |
|---|---|---|
| `shinka_small.yaml` | default dev run | async `2/2/2` workers, `100` generations, `$0.5` budget, `1` island, prompt evolution enabled |
| `shinka_medium.yaml` | moderate parallel run | async `5/5/4` workers, `100` generations, `$0.1` budget, `2` islands |
| `shinka_long.yaml` | long/high-throughput run | async `20/20/8` workers, `400` generations, `$2.0` budget, `2` islands |

Notes:

- Top-level `max_evaluation_jobs`, `max_proposal_jobs`, `max_db_workers` are consumed by `run_evo_async.py`.
- `run_evo.py` ignores those top-level async worker keys.

## Execution Setups

From repo root:

```bash
cd examples/circle_packing
```

Async evolution (recommended):

```bash
python run_evo_async.py --config_path shinka_small.yaml
# swap config_path to shinka_medium.yaml or shinka_long.yaml as needed
```

Sync evolution:

```bash
python run_evo.py --config_path shinka_small.yaml
```

Single-program evaluation (no evolution loop):

```bash
python evaluate.py --program_path initial.py --results_dir results/manual_eval
```

Result inspection:

- Open `load_results.ipynb` for summary plots.
- Open `viz_circles.ipynb` for layout visuals.
