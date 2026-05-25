# Examples

ShinkaEvolve ships with runnable tasks demonstrating different languages,
evaluation styles, and runtime profiles.

---

## Circle Packing

Recommended first example.

| | |
|-|-|
| **Path** | [`examples/circle_packing`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/circle_packing) |
| **Language** | Python |
| **Focus** | Async evolution, config profiles, result notebooks |

Key files: `initial.py`, `evaluate.py`, `run_evo.py`, `shinka_small.yaml`,
`shinka_medium.yaml`, `shinka_long.yaml`

```bash
cd examples/circle_packing
python run_evo.py --config_path shinka_small.yaml
```

Best reference for budgeted async runs and notebook inspection after evolution.

---

## Game 2048

| | |
|-|-|
| **Path** | [`examples/game_2048`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/game_2048) |
| **Language** | Python |
| **Focus** | Policy optimization in a game environment |

Use this for a nontrivial evaluator with task-specific environment logic and a
control-oriented problem shape.

---

## Julia Prime Counting

| | |
|-|-|
| **Path** | [`examples/julia_prime_counting`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/julia_prime_counting) |
| **Language** | Julia candidate + Python evaluator |
| **Focus** | Cross-language evolution, strict correctness scoring |

```bash
cd examples/julia_prime_counting
python evaluate.py --program_path initial.jl --results_dir results/manual_eval
```

Cleanest example of evolving a non-Python candidate while keeping the evaluation
harness in Python.

---

## Go Collatz Steps

| | |
|-|-|
| **Path** | [`examples/go_collatz_steps`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/go_collatz_steps) |
| **Language** | Go candidate + Python evaluator |
| **Focus** | Cross-language evolution, strict correctness scoring |

```bash
cd examples/go_collatz_steps
python evaluate.py --program_path initial.go --results_dir results/manual_eval
```

Use this for a compact compiled-language task where Go candidates compute
Collatz stopping times and `evaluate.py` owns `go run` validation and scoring.

---

## Fortran Heat Diffusion

| | |
|-|-|
| **Path** | [`examples/fortran_heat_diffusion`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/fortran_heat_diffusion) |
| **Language** | Fortran candidate + Python evaluator |
| **Focus** | Compiled numerical stencil evolution |

```bash
cd examples/fortran_heat_diffusion
python evaluate.py --program_path initial.f90 --results_dir results/manual_eval
```

Use this when candidates should be compiled with `gfortran` before
floating-point correctness and runtime scoring.

---

## Wolfram GCD Sum

| | |
|-|-|
| **Path** | [`examples/wolfram_gcd_sum`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/wolfram_gcd_sum) |
| **Language** | Wolfram Language candidate + Python evaluator |
| **Focus** | Program optimization against an auto-calibrated baseline |

```bash
cd examples/wolfram_gcd_sum
python evaluate.py --program_path initial.wl --results_dir results/manual_eval
```

Use this when candidates run through `wolframscript` and are scored by
speedup over a deoptimized seed.

---

## Novelty Generator

| | |
|-|-|
| **Path** | [`examples/novelty_generator`](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/novelty_generator) |
| **Language** | Python |
| **Focus** | Novelty-oriented generation, nontraditional evaluation |

Use this to inspect prompt design, novelty judgment, and nontraditional
evaluation metrics.

---

## Tutorial Notebook

[`examples/shinka_tutorial.ipynb`](https://github.com/SakanaAI/ShinkaEvolve/blob/main/examples/shinka_tutorial.ipynb) — exploratory, notebook-centered introduction
before moving into full CLI or API workflows.

---

## Choosing an Example

| Goal | Example |
|------|---------|
| Best default choice | Circle Packing |
| Cross-language evolution | Julia Prime Counting |
| Compiled numerical stencil | Fortran Heat Diffusion |
| Program optimization | Wolfram GCD Sum |
| Game / control optimization | Game 2048 |
| Creative / open-ended | Novelty Generator |
