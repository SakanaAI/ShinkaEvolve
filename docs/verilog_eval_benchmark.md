# VerilogEval Benchmark — Complete Test Description

> Source: [NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) (NVIDIA, MIT License)
> Papers: [VerilogEval v1 (ICCAD 2023)](https://arxiv.org/abs/2309.07544), [VerilogEval v2 (2024)](https://arxiv.org/abs/2408.11053)

## Overview

156 self-contained Verilog design problems sourced from HDLBits, ranging from
constant outputs to complex finite state machines. Each problem is a triplet:

| File | Purpose |
|------|---------|
| `*_prompt.txt` | Natural-language spec ("implement a module named TopModule with ...") |
| `*_ref.sv` | Golden reference implementation (`RefModule`) |
| `*_test.sv` | Self-checking testbench that compares `TopModule` vs `RefModule` |

The candidate implements `TopModule`. The testbench instantiates both `RefModule`
(golden) and `TopModule` (candidate), drives identical stimuli into both, and
XOR-compares outputs cycle-by-cycle. Mismatches are counted, not just detected —
giving a continuous numeric score, not binary pass/fail.

## Evaluation Pipeline

```
1. iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb \
     candidate.sv  testbench.sv  ref.sv  -o sim
   → Compile-time errors caught here (syntax, binding, types)

2. timeout 30 ./sim
   → Runs simulation, prints "Mismatches: N in M samples"

3. Parse output:
   - "Mismatches: 0 in M samples" → PASS
   - "Mismatches: N in M samples" → RUNTIME MISMATCH (N errors)
   - "TIMEOUT" → infinite loop / latch
   - iverilog stderr → compile error classification
```

### Mismatch Counting (The Numeric Signal)

Every testbench contains this pattern:

```systemverilog
// XOR trick: X in ref matches anything, but X in dut only matches X
assign tb_match = ( { out_ref } === ( { out_ref } ^ { out_dut } ^ { out_ref } ) );

always @(posedge clk, negedge clk) begin
    stats1.clocks++;
    if (!tb_match) begin
        stats1.errors++;
    end
end

final begin
    $display("Mismatches: %1d in %1d samples", stats1.errors, stats1.clocks);
end
```

This gives us `mismatches / total_samples` as a **continuous correctness ratio**.
For Shinka evolution, `combined_score = (1 - mismatches/total_samples) * 100`.

### Timeout Guard

Every testbench includes a 100K-cycle watchdog:

```systemverilog
initial begin
    #1000000
    $display("TIMEOUT");
    $finish();
end
```

Prevents infinite simulation from combinational loops or missing clock edges.

## Failure Classification

The analysis script (`sv-iv-analyze`) classifies failures into codes:

### Compile-Time Failures

| Code | Failure | Description |
|------|---------|-------------|
| `S` | Syntax Error | General parse failure |
| `c` | Clock Binding | `Unable to bind wire/reg 'clk'` — module uses clk but doesn't declare it |
| `p` | Port Binding | `Unable to bind wire/reg` — other port mismatches |
| `e` | Explicit Cast | `Explicit cast required` — datatype mismatches with enums |
| `w` | Wire/Reg Confusion | `is declared here as wire` — assigning to wire in always block |
| `m` | Missing Module | `Unknown module type` — instantiating non-existent submodule |
| `0` | Zero-Size Constant | `Sized numeric constant must have a size greater than zero` |
| `n` | No Sensitivities | `always_comb process has no sensitivities` — empty sensitivity list |
| `C` | General Compile Error | Any other iverilog error |

### Runtime Failures

| Code | Failure | Description |
|------|---------|-------------|
| `.` | **Pass** | 0 mismatches |
| `R` | Runtime Mismatch | Non-zero mismatches, general logic error |
| `r` | Reset Polarity | Uses `posedge reset` or `negedge reset` (async) when sync expected |
| `T` | Timeout | Simulation exceeded 100K cycles |

## Problem Categories (All 156 Problems)

### Category 1: Basic Gates & Constant Outputs (14 problems)
**Combinational. Trivial difficulty.**

| # | Name | Description |
|---|------|-------------|
| 1 | zero | Output constant LOW |
| 3 | step_one | Output constant HIGH |
| 5 | notgate | NOT gate |
| 7 | wire | Wire passthrough |
| 11 | norgate | NOR gate |
| 12 | xnorgate | XNOR gate |
| 14 | andgate | AND gate |
| 26 | alwaysblock1 | AND/OR/XOR/NAND using always block |
| 65 | 7420 | Dual 4-input NAND (7420 chip) |
| 81 | 7458 | 4-input AND-OR (7458 chip) |
| 87 | gates | AND/OR/XOR/NAND/NOR/XNOR/ANOTB |

**Side effects**: None. Pure combinational. Cannot fail at runtime if syntax is correct.

### Category 2: Vectors, Wiring & Bit Manipulation (18 problems)
**Combinational. Easy difficulty.**

| # | Name | Description |
|---|------|-------------|
| 4 | vector2 | 32-bit vector split into bytes |
| 6 | vectorr | 8-bit vector reversal |
| 15 | vector1 | Vector access and concatenation |
| 23 | vector100r | 100-bit vector reversal |
| 32 | vector0 | 3-bit vector declaration |
| 42 | vector4 | Vector concatenation of 8-bit inputs |
| 43 | vector5 | Vector replication and concatenation |
| 44 | vectorgates | Bitwise vs logical operations on vectors |
| 59 | wire4 | Multi-wire connection with intermediate signals |
| 64 | vector3 | 5x1-bit to 1x5-bit vector packing |
| 77 | wire_decl | Wire declarations for intermediate signals |
| 92 | gatesv100 | 100-bit gate operations |
| 94 | gatesv | Bitwise vector operations |
| 105 | rotate100 | 100-bit rotator |

**Side effects**: Incorrect bit-widths cause simulation X-propagation. Truncation warnings.

### Category 3: Multiplexers (9 problems)
**Combinational. Easy-medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 17 | mux2to1v | 8-bit 2-to-1 mux |
| 18 | mux256to1 | 256-to-1 mux (1-bit output) |
| 21 | mux256to1v | 256-to-1 mux (4-bit output) |
| 22 | mux2to1 | 1-bit 2-to-1 mux |
| 39 | always_if | If-else as 2-to-1 mux |
| 76 | always_case | Case-based 6-to-1 mux |
| 97 | mux9to1v | 16-bit 9-to-1 mux |
| 104 | mt2015_muxdff | Mux-DFF interconnection |

**Side effects**: Incomplete case statements create latches (no default). `case` vs `casez` confusion.

### Category 4: Arithmetic & Reduction (12 problems)
**Combinational. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 2 | m2014_q4i | AND reduction |
| 8 | m2014_q4h | OR/AND combinations |
| 9 | popcount3 | 3-bit population count |
| 10 | mt2015_q4a | Combinational comparator |
| 24 | hadd | Half adder |
| 25 | reduction | Even parity via XOR reduction |
| 27 | fadd | Full adder |
| 30 | popcount255 | 255-bit population count |
| 55 | conditional | Min of 4 8-bit unsigned numbers |
| 33 | ece241_2014_q1c | Two's complement output mapping |
| 51 | gates4 | 4-input AND/OR/XOR gates |
| 52 | gates100 | 100-input AND/OR/XOR reduction |

**Side effects**: Signed vs unsigned confusion. Overflow wrapping. Operator precedence.

### Category 5: Karnaugh Maps & Truth Tables (10 problems)
**Combinational. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 50 | kmap1 | 3-variable K-map (a, b, c → out) |
| 57 | kmap2 | 4-variable K-map (a, b, c, d → out) |
| 69 | truthtable1 | 3-input truth table |
| 70 | ece241_2013_q2 | 4-input truth table with SOP |
| 93 | ece241_2014_q3 | 4-variable K-map |
| 113 | 2012_q1g | K-map with don't-cares |
| 116 | m2014_q3 | 4-variable K-map |
| 122 | kmap4 | 4-variable K-map with SOP minimization |
| 125 | kmap3 | 4-variable K-map |

**Side effects**: Don't-care handling — `casez` vs `case`. SOP vs POS choice changes nothing functionally but LLMs often produce non-minimal expressions.

### Category 6: Flip-Flops & Registers (14 problems)
**Sequential. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 28 | m2014_q4a | D latch |
| 31 | dff | Single D flip-flop |
| 34 | dff8 | 8-bit register (8 D flip-flops) |
| 41 | dff8r | 8-bit register with synchronous reset |
| 46 | dff8p | 8-bit register with active-high reset to 0x34 |
| 47 | dff8ar | 8-bit register with asynchronous reset |
| 48 | m2014_q4c | 8-bit register with enable |
| 49 | m2014_q4b | D latch gated by enable |
| 53 | m2014_q4d | Edge-triggered register from D latches |
| 56 | ece241_2013_q7 | JK flip-flop from truth table |
| 73 | dff16e | 16-bit register with byte-enable |
| 78 | dualedge | Dual-edge-triggered flip-flop |
| 58 | alwaysblock2 | XOR using combinational + clocked always blocks |

**Side effects**: **Reset polarity is the #1 killer.** Async (`posedge reset`) vs sync reset. Latch inference from incomplete if/else. Missing `else` creates implicit state holding. `negedge` vs `posedge` clock edge confusion.

### Category 7: Counters (9 problems)
**Sequential. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 35 | count1to10 | 1-to-10 counter |
| 37 | review2015_count1k | 1000-cycle counter |
| 38 | count15 | 4-bit counter (0-15) |
| 40 | count10 | Decade counter (0-9) |
| 67 | countslow | Slow counter with enable and period |
| 68 | countbcd | 4-digit BCD counter |
| 75 | counter_2bc | 2-bit saturating counter |
| 80 | timer | Programmable down-counter/timer |
| 141 | count_clock | 12-hour clock (hours:minutes:seconds BCD) |

**Side effects**: Off-by-one on wrap-around. BCD carry logic. Enable vs reset priority. `count_clock` is the hardest counter — it requires correct cascading of seconds→minutes→hours with AM/PM.

### Category 8: Shift Registers & LFSRs (9 problems)
**Sequential. Medium-hard difficulty.**

| # | Name | Description |
|---|------|-------------|
| 60 | m2014_q4k | Shift register with enable |
| 61 | 2014_q4a | 4-bit shift register with serial/parallel load |
| 63 | review2015_shiftcount | Shift register + down counter combo |
| 82 | lfsr32 | 32-bit Galois LFSR (taps at 32,22,2,1) |
| 84 | ece241_2013_q12 | 8-bit LFSR with specific taps |
| 85 | shift4 | 4-bit shift register with arith/logical shift |
| 86 | lfsr5 | 5-bit LFSR |
| 105 | rotate100 | 100-bit rotator with load |
| 115 | shift18 | 64-bit arithmetic shift register |
| 118 | history_shift | Branch history shift register |

**Side effects**: XOR tap positions are easy to get off-by-one (1-indexed vs 0-indexed). Direction confusion (shift left vs right). Arithmetic vs logical right shift.

### Category 9: Edge Detection (3 problems)
**Sequential. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 45 | edgedetect2 | Rising and falling edge detector (8-bit) |
| 54 | edgedetect | Rising edge detection (8-bit) |
| 66 | edgecapture | Edge capture register (32-bit, reset clears) |

**Side effects**: Off-by-one-cycle timing. Edge detection requires storing previous state — forgetting to initialize creates X propagation on first cycle.

### Category 10: Finite State Machines (34 problems)
**Sequential. Hard difficulty. This is 22% of the benchmark.**

| # | Name | Description |
|---|------|-------------|
| 74 | ece241_2014_q4 | Water sensor FSM |
| 79 | fsm3onehot | 4-state FSM, one-hot encoding |
| 88 | ece241_2014_q5b | Serial receiver FSM with data output |
| 89 | ece241_2014_q5a | Serial receiver FSM |
| 91 | 2012_q2b | FSM from state transition diagram |
| 95 | review2015_fsmshift | FSM controlling shift register |
| 96 | review2015_fsmseq | Sequence detector FSM |
| 99 | m2014_q6c | FSM Y output mapping |
| 100 | fsm3comb | 4-state FSM combinational logic |
| 107 | fsm1s | Alyssa P. Hacker FSM (Mealy-style) |
| 109 | fsm1 | 2-state Moore FSM (A/B) |
| 110 | fsm2 | 2-state FSM with JK-style transitions |
| 111 | fsm2s | FSM from state diagram |
| 119 | fsm3 | 4-state FSM (A→B→C→D) |
| 120 | fsm3s | 4-state FSM (explicit state table) |
| 121 | 2014_q3bfsm | FSM + datapath for counting |
| 128 | fsm_ps2 | PS/2 keyboard protocol FSM (3-byte packets) |
| 129 | ece241_2013_q8 | Mealy machine pattern detector |
| 133 | 2014_q3fsm | FSM with timer control |
| 134 | 2014_q3c | FSM from state assignment table |
| 135 | m2014_q6b | FSM next-state logic |
| 136 | m2014_q6 | FSM with 4 states (one-hot encoding) |
| 137 | fsm_serial | Serial receiver FSM (start/data/stop bits) |
| 138 | 2012_q2fsm | FSM from state diagram |
| 139 | 2013_q2bfsm | FSM from state diagram |
| 140 | fsm_hdlc | HDLC framing FSM (disc/flag/err detection) |
| 143 | fsm_onehot | FSM with one-hot encoding given |
| 146 | fsm_serialdata | Serial receiver with parity checking |
| 148 | 2013_q2afsm | FSM from state diagram |
| 149 | ece241_2013_q4 | Moore FSM water level controller |
| 150 | review2015_fsmonehot | FSM one-hot encoding |
| 151 | review2015_fsm | Sequence detector FSM |
| 154 | fsm_ps2data | PS/2 receiver + data extraction |
| 156 | review2015_fancytimer | FSM-controlled programmable timer |

**Side effects**: **Most failures occur here.** Common issues:
- Wrong reset type (sync vs async) — the testbench expects one or the other
- State encoding mismatch (binary vs one-hot) — doesn't affect correctness but affects resource usage
- Missing default state → latches
- Mealy vs Moore confusion — Mealy outputs change combinationally, Moore only at clock edge
- Off-by-one in sequence detection
- Missing transitions → stuck states (triggers TIMEOUT)

### Category 11: Cellular Automata & Game-of-Life (4 problems)
**Sequential. Hardest difficulty tier.**

| # | Name | Description |
|---|------|-------------|
| 108 | rule90 | 512-bit Rule 90 cellular automaton |
| 124 | rule110 | 512-bit Rule 110 cellular automaton |
| 144 | conwaylife | 16×16 Conway's Game of Life (toroidal) |
| 153 | gshare | Branch predictor (gshare algorithm) |

**Side effects**: Boundary wrapping (toroidal grid). Neighbor counting errors at edges/corners. Simultaneous update requirement — reading from state being written creates race conditions. Must use double-buffered approach (compute next state, then swap).

### Category 12: Lemmings / Game Simulation (4 problems)
**Sequential. Hard difficulty.**

| # | Name | Description |
|---|------|-------------|
| 127 | lemmings1 | Lemming walker: walk left/right, bump reverses |
| 142 | lemmings2 | Lemmings + ground detection (falling) |
| 152 | lemmings3 | Lemmings + digging |
| 155 | lemmings4 | Lemmings + splat (falling too far kills) |

**Side effects**: State priority (digging > walking, falling > digging). Multi-cycle behavior across states. These chain together — lemmings4 builds on all prior lemming behaviors.

### Category 13: Bug Fixing (4 problems)
**Mixed. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 62 | bugs_mux2 | Fix bug in 8-bit 2:1 mux (output width wrong) |
| 114 | bugs_case | Fix buggy case-statement priority encoder |
| 123 | bugs_addsubz | Fix add/subtract with zero-detect bug |
| 132 | always_if2 | Fix if/else priority to avoid unintended latch |

**Side effects**: The buggy code is given in the prompt. LLMs must identify AND fix the specific bug without introducing new ones.

### Category 14: Mystery / Reverse-Engineering Circuits (9 problems)
**Mixed. Hard difficulty — requires reverse-engineering from behavior.**

| # | Name | Description |
|---|------|-------------|
| 90 | circuit1 | Identify circuit from simulation waveform |
| 98 | circuit7 | Reverse-engineer sequential circuit |
| 101 | circuit4 | Reverse-engineer combinational circuit |
| 102 | circuit3 | Reverse-engineer combinational circuit |
| 103 | circuit2 | Reverse-engineer combinational circuit |
| 117 | circuit9 | Reverse-engineer sequential circuit |
| 126 | circuit6 | Reverse-engineer combinational circuit |
| 130 | circuit5 | Reverse-engineer sequential circuit |
| 145 | circuit8 | Reverse-engineer sequential circuit |
| 147 | circuit10 | Reverse-engineer sequential circuit |

**Side effects**: Under-specification — multiple valid implementations possible but testbench checks specific behavior.

### Category 15: Complex Combinational Logic & Case Statements (7 problems)
**Combinational. Medium difficulty.**

| # | Name | Description |
|---|------|-------------|
| 36 | ringer | Phone ringer/vibrate selection |
| 71 | always_casez | Priority encoder with casez |
| 72 | thermostat | Heating/cooling/fan control logic |
| 83 | mt2015_q4b | 25-bit sign-extension |
| 106 | always_nolatches | Case statement that avoids latches |
| 112 | always_case2 | Scanner code to position mapping |

**Side effects**: **Latch inference** is the main hazard. Missing `default` in `case`, missing `else` in `if`. `casez` vs `casex` semantics (casex matches X, casez treats Z as don't-care).

## Side Effects & Pitfalls Summary

These are the things that cause **silent simulation failures** (code compiles, but produces wrong output):

| Pitfall | Frequency | Impact |
|---------|-----------|--------|
| Reset polarity (sync vs async) | Very High | All sequential outputs wrong from cycle 0 |
| Latch inference (incomplete if/case) | High | Outputs hold stale values unpredictably |
| X-propagation (uninitialized regs) | High | Unknown values spread through logic |
| Bit-width mismatch / truncation | Medium | Silent truncation, wrong upper bits |
| Clock edge (posedge vs negedge) | Medium | All outputs delayed or inverted timing |
| Blocking vs non-blocking assignment | Medium | Race conditions in sequential logic |
| Off-by-one in counters/FSMs | Medium | Wrap-around and boundary errors |
| Operator precedence | Low | Subtle bitwise vs logical confusion |
| Signed vs unsigned | Low | Comparison and shift behavior changes |

## Scoring for ShinkaEvolve Integration

The natural score from each testbench is:

```
correctness_ratio = 1.0 - (mismatches / total_samples)
combined_score = correctness_ratio * 100.0
```

This gives a **continuous signal from 0.0 to 100.0** — not binary pass/fail.
A candidate with 95 mismatches out of 800 samples scores 88.1, which is better
than one with 200 mismatches (75.0) but worse than 0 mismatches (100.0).

For multi-problem benchmarking, aggregate across problems:
```
combined_score = mean([score_prob_i for i in selected_problems])
```

## Eval Requirements

- **iverilog v12** (Icarus Verilog) — `apt install iverilog` or build from source
- No GPU, no special hardware, no commercial tools
- Each problem evaluates in <1 second (30-second timeout guard)
- Fully deterministic — same code always produces same score

---

## Leaderboard — Published Results on VerilogEval

Collected from peer-reviewed papers and preprints. All numbers are **pass@k**
(fraction of problems with 0 mismatches) unless noted otherwise.

### VerilogEval v1 (ICCAD 2023)

> Source: [Liu et al., 2023](https://arxiv.org/abs/2309.07544) — 156 problems,
> "Machine" = description-to-code, "Human" = spec-to-RTL.

| Model | Machine pass@1 | Machine pass@5 | Machine pass@10 | Human pass@1 | Human pass@5 | Human pass@10 |
|-------|---------------|----------------|-----------------|--------------|--------------|---------------|
| GPT-4 | 60.0 | 70.6 | 73.5 | 43.5 | 55.8 | 58.9 |
| GPT-3.5 | 46.7 | 69.1 | 74.1 | 26.7 | 45.8 | 51.7 |
| CodeGen-16B-Verilog-SFT | 46.2 | 67.3 | 73.7 | 28.8 | 45.9 | 52.3 |
| CodeGen-16B-Verilog | 42.1 | 60.3 | 66.3 | 24.2 | 36.7 | 42.9 |
| CodeGen-16B-Multi | 34.3 | 53.7 | 60.7 | 21.1 | 36.7 | 41.8 |
| CodeGen-6B-Verilog | 33.6 | 49.4 | 55.2 | 16.4 | 27.3 | 32.3 |
| CodeGen-2B-Verilog | 24.8 | 38.7 | 45.2 | 12.2 | 21.3 | 26.0 |

### VerilogEval v2 (2024)

> Source: [Pinckney et al., 2024](https://arxiv.org/abs/2408.11053) — 156
> spec-to-RTL problems (natural-language prompts). Best pass@1 at T=0 unless noted.

| Model | Size | pass@1 (%) | Setting |
|-------|------|-----------|---------|
| GPT-4o | — | 65.1 | 1-shot |
| GPT-4 Turbo | — | 59.6 | 0-shot |
| Llama 3.1 405B | 405B | 57.9 | 1-shot |
| Mistral Large 2 | — | 48.7 | 1-shot |
| Llama 3.1 70B | 70B | 48.0 | 1-shot |
| CodeLlama 70B | 70B | 41.5 | 1-shot |
| Llama 3 70B | 70B | 43.9 | 0-shot, T=0.8 |
| DeepSeek Coder 33B | 33B | 40.1 | 1-shot |
| CodeLlama 34B | 34B | 34.7 | 1-shot |
| RTL-Coder 6.7B | 6.7B | 36.8 | 0-shot |
| Llama 3 8B | 8B | 28.3 | 1-shot |
| CodeLlama 7B | 7B | 23.1 | 1-shot |

### Evolutionary & Agentic Approaches

These methods go beyond single-shot generation — they use multi-turn feedback,
evolutionary search, or agentic tool-use loops.

| Method | VerilogEval Version | Metric | Score | Notes |
|--------|-------------------|--------|-------|-------|
| **EvolVE** | v2 | pass@1 | **98.1%** | Evolutionary LLM + self-reflection. Code not yet released. ([arXiv 2601.18067](https://arxiv.org/abs/2601.18067)) |
| **ChipAgents** | v2 | pass@1 | **97.4%** (152/156) | Multi-agent workflow (manager + coder + verifier). ([Paper](https://arxiv.org/abs/2411.10877)) |
| **ChipAgents** | v1 Human | pass@1 | **99.4%** (155/156) | Same framework on v1 Human split |
| **EvoVerilog** | v1 Machine | pass@10 | **89.1%** | Evolutionary Verilog generation. No public code. ([arXiv 2508.13156](https://arxiv.org/abs/2508.13156)) |
| **EvoVerilog** | v1 Human | pass@10 | **80.2%** | Same framework on Human split |

### Key Takeaways

1. **Single-shot LLMs plateau around 60-65%** on spec-to-RTL tasks (GPT-4o best).
2. **Evolutionary/agentic methods push to 97-98%** by using simulation feedback
   to iteratively fix errors — exactly the approach ShinkaEvolve takes.
3. **The gap is largest on FSMs and sequential logic** — categories where
   single-shot models make reset polarity, state encoding, or timing errors that
   are trivially caught by simulation but hard to avoid without feedback.
4. **Small specialized models (RTL-Coder 6.7B) compete with much larger general
   models** (CodeLlama 70B), suggesting domain-specific fine-tuning helps.
5. **No public code exists** for any of the evolutionary approaches (EvoVerilog,
   EvolVE). ShinkaEvolve + VerilogEval fills this gap with an open-source
   evolutionary framework and a reproducible benchmark harness.
