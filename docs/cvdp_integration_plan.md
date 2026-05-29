# CVDP × ShinkaEvolve — Integration Plan & Compute Estimate

## Scope

489 open-source CVDP problems (no Cadence license needed), covering 9 code
generation categories. Goal: establish evolutionary baselines and push past the
current 34% SOTA (Claude 3.7 Sonnet single-shot).

---

## Phase Breakdown

### Phase 0: Pipeline Integration (1-2 days, no compute cost)

Wire CVDP evaluation into ShinkaEvolve:
- [ ] Write `examples/cvdp/evaluate.py` — adapter that maps ShinkaEvolve's
  `(program_path, results_dir)` interface to CVDP's Docker-based CocoTB harness
- [ ] Handle CVDP's JSONL dataset format (download from HuggingFace, parse
  context/harness/prompt fields)
- [ ] Build `nvidia/cvdp-sim:v1.0.0` Docker image (one-time, ~5 min)
- [ ] Write `examples/cvdp/run_evo.py` with CVDP-specific system prompt
- [ ] Validate on 3-5 example problems from `example_dataset/`

### Phase 1: Single-Shot Baseline (establish what to evolve)

Run every open-source problem once (no evolution) to find which problems
current LLMs fail on. These are the evolution targets.

| Item | Count |
|------|-------|
| Problems | 489 |
| LLM calls per problem | 5 (pass@5) |
| Total LLM calls | 2,445 |

**Compute:**
- Eval (CPU): 2,445 × 15 sec avg = ~10 CPU-hours
- LLM (API): 2,445 calls × ~4K input + ~1.5K output tokens

**API Cost:**
| Model | Input Cost | Output Cost | Total |
|-------|-----------|-------------|-------|
| Claude Haiku 4.5 | $2.44 | $4.58 | **~$7** |
| GPT-4o-mini | $1.47 | $2.20 | **~$4** |
| Claude Sonnet 4 | $29 | $55 | **~$84** |
| GPT-4.1 | $29 | $37 | **~$66** |

**Expected outcome:** ~320-340 problems fail (based on 34% SOTA pass@1).
These become evolution targets.

### Phase 2: Focused Evolution on Failures (~300 problems)

Run ShinkaEvolve evolution on every problem that failed Phase 1.

| Parameter | Value |
|-----------|-------|
| Problems to evolve | ~300 |
| Generations | 30 |
| Proposals per generation | 4 |
| Islands | 2 |
| LLM calls per problem | 120 |
| **Total LLM calls** | **36,000** |

**Compute:**

Evaluation (CPU only — Docker + iverilog/Verilator):
```
300 problems × 30 gens × 4 candidates × 15 sec = 540,000 sec
= 150 CPU-hours
= ~$28 on Azure D4s_v3 ($0.19/hr)
```

LLM Proposal (the expensive part):
```
Input:  36,000 calls × 4,000 tokens = 144M tokens
Output: 36,000 calls × 1,500 tokens = 54M tokens
```

| Model | Input Cost | Output Cost | Total |
|-------|-----------|-------------|-------|
| Claude Haiku 4.5 | $36 | $68 | **~$104** |
| GPT-4o-mini | $22 | $32 | **~$54** |
| Claude Sonnet 4 | $432 | $810 | **~$1,242** |
| GPT-4.1 | $432 | $540 | **~$972** |
| Llama 3.1 70B (Together) | $129 | $129 | **~$258** |
| Llama 3.1 405B (Together) | $432 | $432 | **~$864** |

### Phase 3: Deep Evolution on Hardest Problems

Take the ~50 hardest problems (FSMs, assertions, testbench checkers) and run
longer evolution with larger populations.

| Parameter | Value |
|-----------|-------|
| Problems | 50 |
| Generations | 100 |
| Proposals per generation | 8 |
| Islands | 4 |
| LLM calls per problem | 800 |
| **Total LLM calls** | **40,000** |

Cost is similar to Phase 2 (slightly more due to longer context from
evolution history).

---

## Total Compute Summary

### API Path (Recommended)

| Phase | LLM Calls | CPU Hours | Cost (Haiku) | Cost (Sonnet) | Cost (4o-mini) |
|-------|-----------|-----------|-------------|---------------|----------------|
| 0 - Setup | ~20 | <1 | ~$0 | ~$0 | ~$0 |
| 1 - Baseline | 2,445 | 10 | $7 | $84 | $4 |
| 2 - Evolution | 36,000 | 150 | $104 | $1,242 | $54 |
| 3 - Deep Evo | 40,000 | 200 | $120 | $1,400 | $60 |
| **Total** | **78,465** | **~360** | **~$231** | **~$2,726** | **~$118** |

Azure VM for eval: 360 CPU-hours × $0.19/hr = **~$68**

**Total all-in:**
- Budget option (GPT-4o-mini): **~$186**
- Mid option (Haiku 4.5): **~$299**
- Premium option (Claude Sonnet 4): **~$2,794**

### Self-Hosted GPU Path

If running Llama 3.1 locally instead of API:

| Model | GPUs Needed | Output Speed | Wall Time (Phase 2) | GPU-Hours | Azure Cost |
|-------|------------|-------------|--------------------|-----------|-----------| 
| Llama 70B | 2× A100 80GB | ~15 tok/s | ~1,000 hrs | 2,000 | ~$6,800 |
| Llama 405B | 8× A100 80GB | ~5 tok/s | ~3,000 hrs | 24,000 | ~$81,600 |

**Verdict: API is 10-50x cheaper than self-hosted for this workload.**
Self-hosting only makes sense if you have idle GPUs or need data privacy.

---

## GPU Hours — Direct Answer

**You don't need GPU hours for evaluation.** Evaluation is CPU-only Docker
containers running iverilog/Verilator.

**For LLM inference (proposing candidates):**

| Scenario | "GPU-Hours" Equivalent |
|----------|----------------------|
| Using API (recommended) | **0 GPU-hours** — it's API cost instead |
| Self-hosting Llama 70B | ~2,000 A100-hours for full benchmark |
| Self-hosting Llama 405B | ~24,000 A100-hours for full benchmark |

The 78K LLM calls generate ~120M output tokens. On API, that's $50-$2,800
depending on model. On your own GPUs, that's weeks of A100 time.

---

## Wall Clock Time (Parallelism)

| Phase | Sequential | 4-way parallel | 8-way parallel |
|-------|-----------|----------------|----------------|
| Phase 1 (baseline) | ~10 hrs | ~3 hrs | ~1.5 hrs |
| Phase 2 (evolution) | ~5 days | ~30 hrs | ~15 hrs |
| Phase 3 (deep evo) | ~6 days | ~36 hrs | ~18 hrs |
| **Total** | **~12 days** | **~3 days** | **~1.5 days** |

Parallelism is limited by: API rate limits, Docker container overhead,
and `max_evaluation_jobs` / `max_proposal_jobs` in ShinkaEvolve config.

---

## Azure Architecture (If Cloud-Deployed)

```
┌─────────────────────────────────────┐
│  Azure VM (D8s_v3, 8 cores, $0.38/hr)  │
│                                     │
│  ShinkaEvolve Runner                │
│   ├── LLM Client (API calls out)   │
│   └── Evaluation Jobs (local Docker)│
│        ├── Container 1 (cvdp-sim)   │
│        ├── Container 2 (cvdp-sim)   │
│        ├── Container 3 (cvdp-sim)   │
│        └── Container 4 (cvdp-sim)   │
└─────────────────────────────────────┘
         │
         ▼ API calls
┌─────────────────┐
│ Claude/GPT API  │
│ (no GPU needed  │
│  on your side)  │
└─────────────────┘
```

No GPU VM needed. A single D8s_v3 ($0.38/hr) handles 4-8 parallel Docker
evals while making API calls for proposals.

---

## Task Count Summary

| Work Item | Estimated Effort |
|-----------|-----------------|
| Pipeline adapter (evaluate.py) | 4-6 hours |
| JSONL dataset parser | 2-3 hours |
| Docker image build + test | 1 hour |
| run_evo.py + system prompt | 2 hours |
| Phase 1 baseline run | 3-10 hours (compute) |
| Phase 2 evolution run | 15-30 hours (compute) |
| Phase 3 deep evolution | 18-36 hours (compute) |
| Results analysis + writeup | 4-6 hours |
| **Total human effort** | **~2 days coding + 2-3 days compute** |

---

## What Success Looks Like

| Metric | Current SOTA | Target | Significance |
|--------|-------------|--------|-------------|
| Overall pass@1 | 34% (Claude 3.7) | 50%+ | First evolutionary result on CVDP |
| cid13 (checker gen) | 6-10% | 20%+ | Currently almost unsolved |
| cid14 (assertions) | 12-19% | 35%+ | Huge practical value |
| cid16 (debugging) | 45-53% | 70%+ | Low-hanging fruit |
| cid03 (spec→RTL) | 44-49% | 65%+ | Direct comparison to VerilogEval |
