# RTLLM × ShinkaEvolve — PPA optimization of Verilog under a fixed spec

Evolve a Verilog design to be **smaller, faster, and lower-power** while keeping its
**function fixed**. A pass/fail correctness score is binary and gives evolution nothing to
climb; here the function is frozen and the fitness is a **continuous** PPA speedup over the
human reference — a `p_speedup`-style task (like KernelBench) but for hardware.

```
score = 100 · geomean( area_ref/area_cand , depth_ref/depth_cand , power_ref/power_cand )
```

The reference design scores **100**; a correct, smaller/faster/lower-power implementation
scores **> 100**.

This example is **self-contained**: it ships one design from the
[RTLLM v2.0](https://github.com/hkust-zhiyao/RTLLM) benchmark — `adder_8bit` (a ripple-carry
adder) — as `initial.sv` + `example.jsonl`, so `python run_evo.py` works out of the box.
Running it, evolution can replace the ripple-carry chain with a parallel-prefix structure
(Kogge-Stone / Brent-Kung) — a shorter carry path, held to formal equivalence with the reference.

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

# 2. standard-cell PDK (proprietary Nangate liberty — NOT redistributed here)
#    fetch the Nangate45 liberty into pdk/nangate45.lib, e.g. from the OpenROAD platforms:
#    https://github.com/The-OpenROAD-Project/OpenROAD/tree/master/test  (Nangate45)
mkdir -p pdk && cp /path/to/Nangate45_typ.lib pdk/nangate45.lib

# 3. provider key in the repo-root .env  (e.g. OPENROUTER_API_KEY=...)
```

## Run the bundled design (out of the box)

```bash
python run_evo.py                       # evolves the bundled adder_8bit (example.jsonl)

# score a single .sv directly
python evaluate.py --program_path initial.sv --results_dir /tmp/out
```

## Run any other RTLLM design

The other RTLLM designs aren't vendored (keep RTLLM's tree as the source of truth). Clone RTLLM
and extract the one(s) you want into a local `problems/*.jsonl` + `seeds/<design>/`:

```bash
git clone https://github.com/hkust-zhiyao/RTLLM.git
python extract_dataset.py --rtllm-root /path/to/RTLLM --designs adder_32bit
RTLLM_DESIGN=adder_32bit RTLLM_PROBLEM_FILE=problems/rtllm.jsonl python run_evo.py
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `RTLLM_DESIGN` | `adder_8bit` | which design to evolve |
| `RTLLM_PROBLEM_FILE` | `example.jsonl` | the problem set (the bundled single design) |
| `RTLLM_TIMEOUT` | `60` | per-tool timeout (s) |
| `RTLLM_POWER` | `1` | set `0` to skip OpenSTA power (score falls back to area·depth) |
| `RTLLM_YOSYS_IMAGE` / `RTLLM_OPENSTA_IMAGE` | `hdlc/yosys:latest` / `opensta:local` | docker images if no native tool |

## Tooling note (honest substitution)

RTLLM's published PPA comes from **Synopsys VCS + Design Compiler** (license-gated). This
example substitutes an all-open flow (Icarus Verilog + Yosys + OpenSTA + Nangate45); absolute
numbers differ from the paper's DC values — the well-known tool-dependence of RTL PPA. To keep
the *relative* claim fair, every candidate is compared against the RTLLM reference on the
**identical Yosys/OpenSTA flow** (evolved vs. human, same measurer).

## Attribution

- The bundled design (`initial.sv`, `example.jsonl`) is derived from
  [RTLLM v2.0](https://github.com/hkust-zhiyao/RTLLM), **MIT License**, © 2024 Nora Lu.
- The **Nangate45** standard-cell liberty is **proprietary to Nangate Inc.** and is **not**
  redistributed here — fetch it yourself (see Setup) into `pdk/nangate45.lib`.
