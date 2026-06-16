# RTLLM × ShinkaEvolve — PPA optimization of Verilog under a fixed spec

Evolve a Verilog design to be **smaller, faster, and lower-power** while keeping its
**function fixed**, using a design from the [RTLLM v2.0](https://github.com/hkust-zhiyao/RTLLM)
benchmark as the spec. A pass/fail correctness score is binary and gives evolution nothing
to climb; here the function is frozen and the fitness is a **continuous** PPA speedup over
the human reference — a `p_speedup`-style task (like KernelBench) but for hardware.

```
score = 100 · geomean( area_ref/area_cand , depth_ref/depth_cand , power_ref/power_cand )
```

The RTLLM reference design scores **100**; a correct, smaller/faster/lower-power
implementation scores **> 100**.

## How each metric is measured (all open-source — no commercial EDA licences)

| metric | tool | what it is |
|---|---|---|
| **area** (µm²) | **Yosys** → **Nangate45** standard cells (`stat -liberty`) | post-synthesis cell area |
| **performance** (logic depth) | **Yosys** (`ltp` — longest topological path) | combinational critical-path length |
| **power** (µW) | **OpenSTA** on the gate-level netlist (`report_power`) | switching + leakage power |

**Icarus Verilog** compiles/runs the design against RTLLM's testbench; correctness is then
held by **equivalence to the reference** (Yosys `equiv`/`sat`) so a candidate can't "win" by
overfitting the finite testbench. **Nangate45** is the open 45 nm cell library Yosys and
OpenSTA map to.

## Setup

```bash
# 1. tools
sudo apt-get install iverilog                 # or: brew install icarus-verilog
docker pull hdlc/yosys:latest                 # Yosys (or install natively)
# OpenSTA: build the opensta:local image from https://github.com/parallaxsw/OpenSTA
#   (Nangate45 liberty is vendored under pdk/)

# 2. clone RTLLM and generate the self-contained problem JSONL + seed (not committed)
git clone https://github.com/hkust-zhiyao/RTLLM.git
python extract_dataset.py --rtllm-root /path/to/RTLLM --designs adder_8bit

# 3. provider key in the repo-root .env  (e.g. OPENROUTER_API_KEY=...)
```

## Run one design

```bash
RTLLM_DESIGN=adder_8bit python run_evo.py

# score a single .sv directly
RTLLM_DESIGN=adder_8bit python evaluate.py \
    --program_path seeds/adder_8bit/initial.sv --results_dir /tmp/out
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `RTLLM_DESIGN` | `adder_8bit` | which design to evolve |
| `RTLLM_PROBLEM_FILE` | `problems/rtllm_proto.jsonl` | the problem set |
| `RTLLM_SCORE_AXES` | `3` | `2` = area·depth, `3` = area·depth·power |
| `RTLLM_TIMEOUT` | `60` | per-tool timeout (s) |
| `RTLLM_YOSYS_IMAGE` / `RTLLM_OPENSTA_IMAGE` | `hdlc/yosys:latest` / `opensta:local` | docker images if no native tool |

## Tooling note (honest substitution)

RTLLM's published PPA comes from **Synopsys VCS + Design Compiler** (license-gated). This
example substitutes an all-open flow (Icarus Verilog + Yosys + OpenSTA + Nangate45);
absolute numbers differ from the paper's DC values — the well-known tool-dependence of RTL
PPA. To keep the *relative* claim fair, every candidate is compared against the RTLLM
reference on the **identical Yosys/OpenSTA flow** (evolved vs. human, same measurer).
