# VerilogEval — Evolving Verilog RTL Designs

Evolve Verilog/SystemVerilog module implementations against the
[VerilogEval](https://github.com/NVlabs/verilog-eval) benchmark (156 problems
from HDLBits, NVIDIA MIT License).

## How It Works

Each VerilogEval problem has a testbench that instantiates a golden `RefModule`
and the candidate `TopModule`, drives identical stimuli, and counts output
mismatches cycle-by-cycle. This gives a **continuous correctness score**
(0–100), not binary pass/fail.

The evaluator:
1. Compiles `candidate.sv + testbench.sv + ref.sv` with `iverilog -g2012`
2. Runs simulation with a 60-second timeout
3. Parses `Mismatches: N in M samples` from output
4. Returns `combined_score = (1 - N/M) * 100`

## Requirements

- **iverilog v12** — `apt install iverilog` (Ubuntu) or build from
  [source](https://github.com/steveicarus/iverilog) (checkout `v12-branch`)
- Clone [verilog-eval](https://github.com/NVlabs/verilog-eval) next to
  ShinkaEvolve, or set `VERILOG_EVAL_DIR`

## Quick Start

```bash
# Default problem: 32-bit Galois LFSR
python run_evo.py

# Pick a different problem
VERILOG_PROBLEM=Prob144_conwaylife python run_evo.py
VERILOG_PROBLEM=Prob140_fsm_hdlc python run_evo.py

# Test the evaluator directly
python evaluate.py --program_path initial.sv --results_dir results
```

## Problem Selection

Set `VERILOG_PROBLEM` to any of the 156 problem IDs. Good evolution targets:

| Problem | Difficulty | Description |
|---------|-----------|-------------|
| `Prob082_lfsr32` | Medium | 32-bit Galois LFSR (default) |
| `Prob068_countbcd` | Medium | 4-digit BCD counter |
| `Prob140_fsm_hdlc` | Hard | HDLC framing FSM |
| `Prob144_conwaylife` | Hard | 16×16 Conway's Game of Life |
| `Prob156_review2015_fancytimer` | Hard | FSM-controlled timer |

See `docs/verilog_eval_benchmark.md` for the full problem taxonomy.
