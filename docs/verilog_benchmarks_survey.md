# Verilog/RTL Benchmark Survey — Complete Landscape (May 2025)

A comprehensive comparison of every major benchmark for evaluating LLM-generated
Verilog/SystemVerilog, ordered from most saturated to most open.

---

## 1. VerilogEval (v1 2023, v2 2024)

> Source: [NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) (NVIDIA, MIT)
> Papers: [v1 (ICCAD 2023)](https://arxiv.org/abs/2309.07544), [v2 (2024)](https://arxiv.org/abs/2408.11053)

### What It Is
156 self-contained Verilog design problems sourced from HDLBits. Each problem is
a (prompt, reference, testbench) triplet. The testbench XOR-compares TopModule
vs RefModule outputs cycle-by-cycle, counting mismatches for a continuous score.

### Task Types (15 categories, all spec-to-RTL)
| Category | # Problems | Example |
|----------|-----------|---------|
| Basic Gates & Constants | 14 | NOT, AND, NOR, XNOR |
| Vectors & Bit Manipulation | 18 | 100-bit reversal, byte packing |
| Multiplexers | 9 | 256-to-1 mux, mux-DFF chains |
| Arithmetic & Reduction | 12 | 255-bit popcount, full adder |
| Karnaugh Maps | 10 | 3-4 variable K-maps with don't-cares |
| Flip-Flops & Registers | 14 | DFF, dual-edge, byte-enable |
| Counters | 9 | BCD counter, 12-hour clock |
| Shift Registers & LFSRs | 9 | 32-bit Galois LFSR, arithmetic shift |
| Edge Detection | 3 | Rising/falling edge capture |
| **FSMs** | **34** | PS/2 protocol, HDLC framing, serial receiver |
| Cellular Automata | 4 | Conway's Game of Life, Rule 90/110 |
| Lemmings Simulation | 4 | Walking, falling, digging, splatting FSMs |
| Bug Fixing | 4 | Fix given buggy Verilog |
| Mystery Circuits | 9 | Reverse-engineer from waveforms |
| Complex Combinational | 7 | Priority encoder, thermostat logic |

### Tools Required
- iverilog v12 (open source)
- No GPU, no commercial tools

### Scoring
`combined_score = (1 - mismatches/total_samples) * 100` — continuous 0-100.

### Best Published Results
| Method | Version | Metric | Score | Status |
|--------|---------|--------|-------|--------|
| ChipAgents | v1 Human | pass@1 | **99.4%** (155/156) | **Saturated** |
| EvolVE | v2 | pass@1 | **98.1%** | **Nearly saturated** |
| ChipAgents | v2 | pass@1 | 97.4% (152/156) | Near saturated |
| EvoVerilog | v1 Machine | pass@10 | 89.1% | |
| GPT-4o | v2 | pass@1 | 65.1% | Single-shot baseline |
| Llama 3.1 405B | v2 | pass@1 | 57.9% | |

### Verdict
**Saturated for agentic approaches.** Still useful as a quick sanity check or for
benchmarking new base models, but no longer differentiates evolved approaches.

---

## 2. RTLLM (v1 2023, v2 2024)

> Source: [RTLLM paper (ASP-DAC 2024)](https://arxiv.org/abs/2308.05345)

### What It Is
30 hand-crafted RTL design problems (11 arithmetic, 19 logic circuits) with
natural-language descriptions, testbenches, and reference implementations.

### Task Types
| Category | # Problems | Examples |
|----------|-----------|---------|
| Arithmetic circuits | 11 | Adders, multipliers, dividers |
| Logic circuits | 19 | FSMs, decoders, encoders, arbiters |

### Tools Required
- iverilog / Verilator (open source)

### Best Published Results
| Method | Version | Score |
|--------|---------|-------|
| GPT-4.1 | v2 | **96.1%** |
| Claude Sonnet 4 | v2 | ~90% |
| EvolVE | v2 | 92% |

### Verdict
**Saturated.** Only 30 problems — too small to meaningfully differentiate approaches.

---

## 3. CVDP — Comprehensive Verilog Design Problems (June 2025)

> Source: [NVlabs/cvdp_benchmark](https://github.com/NVlabs/cvdp_benchmark) (NVIDIA, Apache 2.0)
> Paper: [arXiv 2506.14074](https://arxiv.org/abs/2506.14074)
> Dataset: [HuggingFace nvidia/cvdp-benchmark-dataset](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset)

### What It Is
783 expert-authored problems across **13 task categories** spanning RTL generation,
verification, debugging, and comprehension. Problems cover real-world IP blocks
(ALU, FIFO, AES, arbiters, crossbars, CPUs, accelerators). Both non-agentic
(single-turn) and agentic (multi-turn, tool-using) formats.

### Task Categories (13 total)

#### Code Generation (9 categories, 660 problems)

| CID | Category | Non-Agentic | Agentic | Total | Description |
|-----|----------|-------------|---------|-------|-------------|
| **cid02** | RTL Code Completion | 94 | 0 | 94 | Fill in missing logic from partial implementations |
| **cid03** | Spec-to-RTL | 78 | 37 | 115 | Generate complete module from natural language spec |
| **cid04** | Code Modification | 56 | 26 | 82 | Modify existing RTL to change behavior |
| **cid05** | Module Reuse | 0 | 26 | 26 | Instantiate and integrate existing submodules |
| **cid07** | Lint/QoR Improvement | 41 | 0 | 41 | Fix lint warnings, improve synthesis quality |
| **cid12** | Testbench Stimulus Gen | 68 | 18 | 86 | Write stimulus for design verification |
| **cid13** | Testbench Checker Gen | 53 | 18 | 71 | Write output checkers / scoreboards |
| **cid14** | Assertion Generation | 68 | 30 | 98 | Write SVA (SystemVerilog Assertions) |
| **cid16** | RTL Debugging | 36 | 11 | 47 | Find and fix bugs in given RTL |

#### Code Comprehension (4 categories, 123 problems)

| CID | Category | Problems | Description | Scoring |
|-----|----------|----------|-------------|---------|
| **cid06** | RTL/Spec Correspondence | 34 | Does this RTL match this spec? | BLEU |
| **cid08** | TB/Test Plan Correspondence | 29 | Does testbench cover the test plan? | BLEU |
| **cid09** | RTL Q&A | 34 | Answer questions about RTL code | LLM-judged |
| **cid10** | Testbench Q&A | 26 | Answer questions about testbenches | LLM-judged |

### Difficulty Levels
- **Easy** and **Medium** for non-agentic
- **Easy**, **Medium**, and **Hard** for agentic (hard = too complex for single-turn)

### Design Types Covered (Real-World IP Blocks)
| Domain | Examples |
|--------|---------|
| Processing | ALU, CPU cores, accelerators, systolic arrays |
| Memory | FIFOs, CAMs, caches, register files |
| Interconnect | Arbiters (fixed/round-robin), crossbars, routers |
| Communication | UART, SPI, I2C, AXI4Lite bridges, 8b10b codec, 64b/66b codec |
| Crypto | AES encryption/decryption modules |
| Signal Processing | Barrel shifters, multipliers, convolution engines |
| Image Processing | Local binary pattern, distance transform |
| Control | FSMs (Mealy/Moore), programmable timers, LFSRs |
| Algorithms | Brick sort, Hamming distance, Huffman coding |

### Tools Required
| Category | Tool | License |
|----------|------|---------|
| cid02-05, cid07, cid16 | Icarus Verilog, Verilator, Yosys | Open source |
| cid06, cid08-10 | LLM judge / BLEU scoring | N/A |
| cid12-14 | **Cadence Xcelium** (some problems) | **Commercial** |
| All | Docker, CocoTB, Python 3.12 | Open source |

**Important**: ~489 problems (62%) are fully open-source evaluable ("no_commercial").
The remaining 294 problems (38%) require Cadence Xcelium for verification tasks.

### Dataset Accessibility
| Source | Problems | Format | Solutions |
|--------|----------|--------|-----------|
| HuggingFace | 649 (of 783) | JSONL | Withheld (contamination prevention) |
| GitHub repo | Example subset | JSONL | Included in examples |
| Paper | 783 described | Tables | N/A |

**Note**: 20 problems were withheld from initial release (harness issues + licensing).
Reference solutions ("output"/"patch") are withheld to prevent contamination.
Contact NVIDIA for access if needed.

### Per-Category Model Performance (Pass@1, n=5)

#### Non-Agentic Code Generation

| Category | Claude 3.7 | GPT-4.1 | GPT o1 | Llama 405B | Llama 70B |
|----------|-----------|---------|--------|-----------|----------|
| cid02 (Completion) | 34% | 37% | — | 24% | — |
| cid03 (Spec→RTL) | 48% | 44% | — | 31% | — |
| cid04 (Modification) | 45% | 37% | — | 36% | — |
| cid07 (Improvement) | 44% | 32% | — | 20% | — |
| cid12 (Stimulus) | 25% | 16% | — | 21% | — |
| cid13 (Checker) | **6%** | 10% | — | **5%** | — |
| cid14 (Assertions) | 19% | 12% | — | 13% | — |
| cid16 (Debugging) | **53%** | **45%** | — | 32% | — |
| **Overall** | **33.6%** | **28.9%** | **20.1%** | **22.8%** | **17.5%** |

#### Agentic Code Generation

| Category | Claude 3.7 | GPT-4.1 | Llama 405B | GPT o1 |
|----------|-----------|---------|-----------|--------|
| cid03 (Spec→RTL) | 49% | — | — | — |
| cid04 (Modification) | 42% | — | — | — |
| cid05 (Module Reuse) | 24% | — | — | — |
| cid12 (Stimulus) | 7% | — | — | — |
| cid13 (Checker) | **0%** | — | — | — |
| cid14 (Assertions) | 19% | — | — | — |
| cid16 (Debugging) | 53% | — | — | — |
| **Overall** | **29%** | **21%** | **21%** | **14%** |

#### Code Comprehension

| Category | Claude 3.7 | GPT-4.1 | Llama 405B |
|----------|-----------|---------|-----------|
| cid06 (RTL Correspondence) | 63% | — | — |
| cid08 (TB Correspondence) | 42% | — | — |
| cid09 (RTL Q&A) | 78% | — | — |
| cid10 (TB Q&A) | 82% | — | — |
| **Average** | **66%** | — | — |

### Key Observations
1. **Testbench checker generation (cid13) is essentially unsolved** — 0-10% across all models
2. **Assertion generation (cid14) is very hard** — 12-19%
3. **Debugging (cid16) is the easiest code-gen category** — 45-53%
4. **Agentic format is harder** — Claude drops 4%, GPT-4.1 drops 8%
5. **66+ points of headroom** on the overall benchmark

### Verdict
**The clear next target.** 5x more problems than VerilogEval, covers verification
and debugging (not just generation), uses real-world IP blocks, has Docker-based
evaluation, and is massively unsaturated at 34% best score. Same team as VerilogEval.

---

## 4. ICRTL — Industry-Scale IC RTL Benchmark

> Source: [github.com/weiber2002/ICRTL](https://github.com/weiber2002/ICRTL)
> Paper: [EvolVE (arXiv 2601.18067)](https://arxiv.org/abs/2601.18067)

### What It Is
6 industry-scale problems from Taiwan's National IC Design Contest. Focus is on
**PPA optimization** (Power, Performance, Area), not just correctness.

### Task Types
| Problem | Domain | Description |
|---------|--------|-------------|
| Q1 - LBP | Image Processing | Local Binary Pattern accelerator |
| Q2 - GEMM | Matrix Processing | Systolic array for matrix multiplication |
| Q3 - CONV | Signal Processing | 2D convolution accelerator |
| Q4 - HC | Compression | Huffman coding hardware |
| Q5 - JAM | Optimization | Job assignment machine (combinatorial) |
| Q6 - DT | Image Processing | Distance transform on binary images |

### Tools Required
- **Open source**: Icarus Verilog + Yosys
- **Commercial** (optional): Synopsys VCS + Design Compiler + PrimeTime

### Scoring
PPA product: `Area × Delay × Power`. Lower is better.

### Best Published Results
| Method | Metric | Result |
|--------|--------|--------|
| EvolVE | PPA reduction (Huffman) | **66%** improvement |
| EvolVE | PPA reduction (geomean all) | **17%** improvement |

### Verdict
**Interesting but tiny** (6 problems). Unique angle on PPA optimization rather
than correctness. Good complement to correctness-focused benchmarks.

---

## 5. RTL-Repo — Repository-Level RTL Benchmark

> Source: [github.com/AUCOHL/RTL-Repo](https://github.com/AUCOHL/RTL-Repo) (IEEE LAD'24)
> Paper: [arXiv 2405.17378](https://arxiv.org/abs/2405.17378)

### What It Is
4000+ Verilog code samples extracted from real public GitHub repositories. Tests
whether LLMs can generate code that fits within existing multi-file projects,
not isolated snippets.

### Task Type
Repository-level code completion — given the full repo context, complete a
target file or function. Tests understanding of module hierarchies, naming
conventions, and cross-file dependencies.

### Tools Required
- Standard Verilog simulation tools

### Verdict
**Interesting for repo-level evaluation** but focuses on code completion, not
design from scratch. Complements CVDP's agentic tasks well.

---

## 6. CktEvo — Repository-Level RTL Evolution (Feb 2026)

> Source: [github.com/cure-lab/cktevo](https://github.com/cure-lab/cktevo)
> Paper: [arXiv 2603.08718](https://arxiv.org/abs/2603.08718)

### What It Is
A benchmark specifically for **evolving** RTL designs at repository scale. Uses
complete IP cores (not snippets). The task is: "preserve functional correctness
while improving PPA through cross-file edits." Uses island-based population
model inspired by AlphaEvolve.

### IP Cores Included (11 repositories)
Processors, controllers, interfaces, and encoders/decoders spanning real-world
hardware designs.

### Framework
1. Graph-based code analyzer → Control Data Flow Graph
2. Prompt generator with bottleneck identification
3. LLM-driven mutation
4. Formal verification (logic equivalence checking)
5. Evaluation cascade (fast ↔ full)

### Tools Required
- **Open source**: Yosys + SkyWater 130nm PDK
- **Commercial** (optional): Synopsys Formality + Design Compiler

### Best Published Results (Open Source Toolchain)
| Metric | Result |
|--------|--------|
| Area reduction | 2.80% (geomean) |
| Delay reduction | 7.92% |
| ADP reduction | 10.50% |
| Best single design (HSM) | 34.97% timing improvement |

### Models Tested
DeepSeek-v3 (best), GPT-4o, Qwen3-coder

### Verdict
**Most aligned with ShinkaEvolve's paradigm** — literally designed for
evolutionary RTL optimization. But focuses on PPA, not correctness. Small
benchmark (11 repos) but novel angle.

---

## Cross-Benchmark Comparison

### Scope & Saturation

| Benchmark | # Problems | Task Focus | Best Score | Headroom | Open Source Eval? |
|-----------|-----------|------------|-----------|----------|-------------------|
| VerilogEval v1 | 156 | Spec→RTL only | 99.4% | ~0% | Yes (iverilog) |
| VerilogEval v2 | 156 | Spec→RTL only | 98.1% | ~2% | Yes (iverilog) |
| RTLLM v2 | 30 | Spec→RTL only | 96.1% | ~4% | Yes |
| **CVDP** | **783** | **13 categories** | **34%** | **~66%** | **62% yes** |
| ICRTL | 6 | PPA optimization | — | Wide open | Yes + commercial |
| RTL-Repo | 4000+ | Repo completion | — | Wide open | Yes |
| CktEvo | 11 repos | RTL evolution | 10.5% ADP | Wide open | Yes + commercial |

### What Each Tests

| Capability | VerilogEval | RTLLM | CVDP | ICRTL | RTL-Repo | CktEvo |
|------------|-------------|-------|------|-------|----------|--------|
| Spec→RTL generation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Code completion | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| Code modification | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Module reuse/integration | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Lint/QoR improvement | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Testbench writing | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Assertion generation | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Bug finding/fixing | Partial (4) | ❌ | ✅ | ❌ | ❌ | ❌ |
| Code comprehension | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| PPA optimization | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Multi-file/cross-file | ❌ | ❌ | ✅ (agentic) | ✅ | ✅ | ✅ |
| Real-world IP blocks | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Formal verification | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Problem Complexity Scale

```
Simple                                                          Complex
 ├───────┤          ├────────────┤     ├──────────────────────┤
 VerilogEval        RTLLM             CVDP
 (single module,    (single module,   (multi-file IP blocks,
  HDLBits-level)     spec→RTL)        verification, debug,
                                       agentic workflows)

                                       ICRTL    CktEvo
                                       (contest  (repo-level
                                        scale)   evolution)
```

### Alignment with ShinkaEvolve

| Benchmark | Evolution-Friendly? | Why |
|-----------|-------------------|-----|
| VerilogEval | ✅ Great | Continuous scoring, fast eval, simple programs |
| RTLLM | ⚠️ OK | Small (30 problems), binary pass/fail |
| **CVDP** | **✅ Excellent** | Continuous scoring possible, huge headroom, Docker eval |
| ICRTL | ⚠️ OK | PPA is continuous signal, but only 6 problems |
| RTL-Repo | ❌ Poor | Code completion, no evolution signal |
| CktEvo | ✅ Great | Literally built for evolution, PPA signal |

---

## Recommendation for ShinkaEvolve

### Tier 1 (Implement now)
- **VerilogEval** ← Done ✅
- **CVDP no-commercial subset** ← 489 problems, open-source eval, enormous headroom

### Tier 2 (Implement next)
- **CktEvo** ← Most philosophically aligned (evolutionary RTL optimization)
- **ICRTL** ← PPA optimization is novel evolution signal

### Tier 3 (Consider later)
- **CVDP commercial subset** ← Needs Cadence Xcelium license
- **RTL-Repo** ← Repo-level completion, different paradigm

---

## Sources

- [VerilogEval v1](https://arxiv.org/abs/2309.07544) — Liu et al., ICCAD 2023
- [VerilogEval v2](https://arxiv.org/abs/2408.11053) — Pinckney et al., 2024
- [RTLLM](https://arxiv.org/abs/2308.05345) — Lu et al., ASP-DAC 2024
- [CVDP](https://arxiv.org/abs/2506.14074) — Pinckney et al., June 2025
- [ICRTL / EvolVE](https://arxiv.org/abs/2601.18067) — Wei et al., Jan 2026
- [RTL-Repo](https://arxiv.org/abs/2405.17378) — Allam et al., IEEE LAD 2024
- [CktEvo](https://arxiv.org/abs/2603.08718) — Fang et al., Feb 2026
- [EvoVerilog](https://arxiv.org/abs/2508.13156) — 2025
- [ChipAgents](https://arxiv.org/abs/2411.10877) — 2024
