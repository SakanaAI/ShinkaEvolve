# Autoformalization Example

Lean 4 example task for ShinkaEvolve. The evolving program is a partial Lean theorem statement in [`initial.lean`](./initial.lean). The evaluator asks an LLM prover to complete that statement, checks the generated proof with Lean, and scores the result from the produced proof.

## What It Does

- evolves Lean 4 theorem/problem formulations rather than Python code
- keeps Lean-specific proof generation and verification inside this example directory
- uses [`evaluate.py`](./evaluate.py) as the Shinka entrypoint
- uses [`utils_lean.py`](./utils_lean.py) for:
  - proof-prompt construction
  - prover calls through an OpenAI-compatible client
  - Lean validation through `lean-interact`

## Prerequisites

- Python environment with ShinkaEvolve installed
- `OPENAI_API_KEY` set for the prover model
- Lean toolchain available locally
- `lean-interact` installed into the same environment

Recommended extra setup:

- install Lean via `elan`
- confirm `lake` / Lean can fetch `mathlib`

## Setup

From repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install lean-interact
```

If you use plain `pip` instead of `uv`:

```bash
pip install -e .
pip install lean-interact
```

Set the prover key:

```bash
export OPENAI_API_KEY=...
```

Optional: choose a different prover model at runtime with `--prover_model`.

## Files

- `initial.lean`: seed Lean formalization with `EVOLVE-BLOCK` markers
- `evaluate.py`: Shinka evaluation entrypoint; runs proof generation + scoring
- `utils_lean.py`: Lean prompt construction, proof post-processing, Lean validation

## Running It

Single evaluation:

```bash
python examples/autoformalization/evaluate.py \
  --program_path examples/autoformalization/initial.lean \
  --results_dir results/autoformalization_manual
```

Hydra/Shinka launch:

```bash
shinka_launch variant=autoformalization_example
```

The task preset lives at [`shinka/configs/task/autoformalization.yaml`](../../shinka/configs/task/autoformalization.yaml).

## Evaluation Flow

1. Shinka mutates the editable block in `initial.lean`.
2. `evaluate.py` passes the current Lean file to `generate_proof`.
3. The prover model completes the proof from the Lean prompt.
4. `utils_lean.py` post-processes the output and validates it with Lean.
5. Metrics are written to `metrics.json` and `correct.json`.

## Notes

- `lean-interact` is example-local on purpose; it is not required for the base Shinka package install.
- This example currently assumes an OpenAI-compatible prover client.
- Lean verification can be slow on first run because toolchain / dependency setup may be downloaded.
