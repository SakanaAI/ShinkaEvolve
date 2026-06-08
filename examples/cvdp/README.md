# CVDP -- Evolving Verilog RTL with CocoTB Testbenches

Evolve Verilog/SystemVerilog modules against NVIDIA's
[CVDP benchmark](https://github.com/NVlabs/cvdp-benchmark) (302 open-source
problems across 5 code-generation categories, best published SOTA 34% pass@1).

## How It Works

Each CVDP problem provides a Docker-based CocoTB testbench that compiles and
simulates the candidate RTL. The evaluator:

1. Loads a problem from a JSONL file (harness files + prompt)
2. Extracts the harness into a temp workspace
3. Writes the candidate module to the expected RTL path
4. Runs `docker compose run --rm direct` (CocoTB + iverilog simulation)
5. Parses pytest output for `N passed, M failed`
6. Returns `combined_score = (passed / total) * 100`

This gives **continuous scoring** (0-100), not binary pass/fail. Evolution can
make partial progress across multiple test cases.

## Requirements

- **Docker** -- installed and running
- **CVDP simulation image** -- Build from the CVDP repository:
  ```bash
  git clone https://github.com/NVlabs/cvdp-benchmark
  cd cvdp-benchmark
  docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
  ```
  The image contains cocotb 2.0.1 + Icarus Verilog v13_0 + pytest.
  
  ⚠️ There is **no public pre-built image**; you must build it locally.
  
  Alternative: Use any Docker image with cocotb≥2.0 and iverilog by setting `CVDP_SIM_IMAGE`.  
  Example: `0x01be/cocotb:latest` (cocotb 1.4, older) or build your own.
- **Python 3.10+** with ShinkaEvolve installed

## Quick Start

```bash
# Run evolution on the included LFSR example problem
python run_evo.py

# Pick a different problem from the lite set
CVDP_PROBLEM_ID=cvdp_copilot_lfsr_0001 python run_evo.py

# Test the evaluator directly
python evaluate.py --program_path initial.sv --results_dir results
```

## Dataset

The `problems/cvdp_lite.jsonl` includes 1 example problem with a known solution
for pipeline validation. To download the full 302-problem dataset:

```bash
# All problems
python download_dataset.py

# Filter by difficulty or category
python download_dataset.py --difficulty medium
python download_dataset.py --category cid003
python download_dataset.py --difficulty medium --category cid003

# Then run evolution on it
CVDP_PROBLEM_FILE=problems/cvdp_full.jsonl CVDP_PROBLEM_ID=<id> python run_evo.py
```

## Problem Categories

| Category | Count | Description | SOTA |
|----------|-------|-------------|------|
| cid002 | 94 | Code completion | 29-40% |
| cid003 | 78 | Spec-to-RTL | 44-49% |
| cid004 | 55 | Code modification | 28-41% |
| cid007 | 40 | Lint / QoR fixes | 23-35% |
| cid016 | 35 | Debugging | 45-53% |

Categories cid005, cid012-014 require commercial EDA tools and are excluded
from the open-source dataset.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CVDP_PROBLEM_FILE` | `problems/cvdp_lite.jsonl` | Path to JSONL problem set |
| `CVDP_PROBLEM_ID` | (first problem) | Specific problem ID to evolve |
| `CVDP_TIMEOUT` | `120` | Docker run timeout in seconds |
| `CVDP_SIM_IMAGE` | `nvidia/cvdp-sim:v1.0.0` | Docker image for simulation. Must be built locally from CVDP repository (cocotb 2.0.1 + iverilog v13_0). ⚠️ No public pre‑built image exists. |

## Architecture

```
ShinkaEvolve
  |-- proposes candidate.sv via LLM
  |-- calls evaluate.py
        |-- loads problem from JSONL
        |-- extracts harness files to temp dir
        |-- writes candidate RTL
        |-- docker compose run (CocoTB + iverilog)
        |-- parses pytest: N passed / M failed
        |-- returns combined_score to evolution loop
```

See `docs/cvdp_integration_plan.md` for compute estimates and Azure deployment.
