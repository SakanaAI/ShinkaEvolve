# RTLLM × ShinkaEvolve — PPA optimization under a fixed spec

Evolve Verilog RTL to be **smaller and faster** while staying **provably correct**,
using the [RTLLM v2.0](https://github.com/hkust-zhiyao/RTLLM) benchmark designs as
fixed functional specifications. This is a "p_speedup" task in the spirit of
KernelBench, but for hardware: the function is frozen, and evolution optimizes the
post-synthesis **area** and **logic depth**.

## Fitness

For each candidate the evaluator runs three stages:

1. **Syntax + testbench** — `iverilog` compiles the candidate against RTLLM's own
   `testbench.v` and checks the `Your Design Passed` banner. (Open-source mirror of
   RTLLM's Synopsys VCS flow.)
2. **Formal equivalence (the gate)** — `yosys` proves the candidate is logically
   **identical to the reference for all inputs** via SAT (`equiv_make` + `equiv_simple`
   /`equiv_induct`). This is the open analog of the Synopsys Formality check used by
   RTL-OPT, and it **closes the testbench-overfitting reward-hacking hole**: a design
   that passes the finite testbench but is not equivalent scores 0.
3. **PPA** — `yosys` maps to an AIG (`abc -g AND`); **area = AIG cell count**,
   **delay = AIG logic depth** (`ltp`). License-free, deterministic, technology-
   independent proxy for Design Compiler area/WNS.

```
score = 0                                                  if syntax or equivalence fails
score = 100 · sqrt( area_ref/area_cand · depth_ref/depth_cand )   otherwise
```

The RTLLM human reference design scores **exactly 100**. **Beating it scores > 100.**

## Tooling note (honest substitution)

RTLLM's published PPA comes from **Synopsys VCS + Design Compiler**, which are
license-gated and unavailable on a CPU box. We substitute **Icarus Verilog + Yosys**.
Absolute numbers therefore differ from the paper's DC values — this is the well-known
tool-dependence of RTL PPA. To keep the *relative* claim fair, every candidate is
compared against the RTLLM reference on the **identical Yosys flow** (evolved vs.
human, same measurer), which neutralizes tool bias for the "beat the reference" result.

## Setup

```bash
# 1. clone RTLLM somewhere
git clone https://github.com/hkust-zhiyao/RTLLM.git

# 2. generate the self-contained JSONL + seeds (not committed)
python extract_dataset.py --rtllm-root /path/to/RTLLM

# 3. tools: iverilog on PATH; yosys native OR the docker image
docker pull hdlc/yosys:latest          # if you don't have native yosys

# 4. provider key in the repo-root .env (e.g. OPENROUTER_API_KEY=...)
```

## Run

```bash
# single design
RTLLM_DESIGN=adder_8bit python run_evo.py

# the prototype sweep (3 designs, parallel, model bandit)
python run_all.py --designs adder_8bit adder_32bit multi_8bit \
    --generations 30 --workers 3 \
    --models openrouter/anthropic/claude-sonnet-4.5 \
             openrouter/openai/gpt-5-codex \
             openrouter/deepseek/deepseek-chat-v3.1

# score one file directly
RTLLM_DESIGN=adder_8bit python evaluate.py \
    --program_path seeds/adder_8bit/initial.sv --results_dir /tmp/out
```

## Prototype designs

| Design | Reference is… | Headroom | Baseline (AIG area / depth) |
|---|---|---|---|
| `adder_8bit`  | ripple-carry (8 full adders) | parallel-prefix | 121 / 36 |
| `adder_32bit` | ripple-carry | log-depth prefix | 575 / 74 |
| `multi_8bit`  | behavioral shift-add | Booth / Wallace | 1105 / 117 |

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `RTLLM_PROBLEM_FILE` | `problems/rtllm_proto.jsonl` | the problem set |
| `RTLLM_DESIGN` | first in file | which design to evaluate |
| `RTLLM_TIMEOUT` | `60` | per-tool timeout (s) |
| `RTLLM_YOSYS_IMAGE` | `hdlc/yosys:latest` | docker image if no native yosys |
