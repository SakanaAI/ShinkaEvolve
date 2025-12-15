# Multi-Turn Agentic Architecture PR Validation

> **⚠️ HARD REQUIREMENTS - NON-NEGOTIABLE**
>
> The validation criteria in this ExecPlan are NOT suggestions. They are hard requirements that MUST ALL PASS before the PR can be submitted. Do not adjust, skip, or weaken any criterion. If a validation fails, fix the code - do not modify the requirement.
>
> This PR is for Sakana AI's ShinkaEvolve. Robert Tjarko Lange has specific expectations. We deliver what he asked for, fully validated, or we don't submit.

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Maintained in accordance with `/Users/juno/workspace/shrinkaevolve-codexevolve/PLANS.md`.

## Purpose / Big Picture

This ExecPlan validates that the `feat/multi-turn-architecture-clean` branch is ready for PR to Sakana AI's ShinkaEvolve. After this work, users can:
1. Run agentic multi-turn editing with ShinkaAgent (native) or Codex CLI backends
2. Use multi-file workspaces (e.g., boids_flocking with 5 files)
3. Have bandit sampling select models dynamically in agentic mode
4. Continue using legacy single-file mode with zero regressions

The PR addresses Robert Tjarko Lange's specific requests: native control (not black-box CLI wrapper), multi-file support, and full backward compatibility.

## Progress

- [x] (2025-12-14 18:18Z) Fixed Hydra config override syntax (`override /evolution@_global_: agentic`)
- [x] (2025-12-14 18:19Z) Temporarily used gpt-4.1 due to missing gpt-5.2 in pricing.py
- [x] (2025-12-14 23:40Z) Added gpt-5.2 to pricing.py and REASONING_OAI_MODELS, restored gpt-5.2 as default
- [x] (2025-12-14 18:19Z) Fixed display.py NoneType subscript bug in patch_name
- [x] (2025-12-14 18:21Z) Restructured boids task config to nest evo_config for proper Hydra merging
- [x] (2025-12-14 18:22Z) Created boids_flocking_agentic variant with correct overrides
- [x] (2025-12-14 18:25Z) Committed all changes, working tree clean (13 commits ahead)
- [x] (2025-12-15 13:31Z) V8.1: pytest tests/ passes - 39 passed
- [x] (2025-12-15 13:31Z) V8.2: ruff check passes (changed files only)
- [x] (2025-12-15 13:31Z) V8.3: black --check passes (changed files only)
- [x] (2025-12-15 13:31Z) V8.4: isort --check passes (changed files only)
- [x] (2025-12-15 13:51Z) V7: Legacy regression - 15 gens, score 0.96→2.02 correct (2.35 raw), all legacy features working
- [x] (2025-12-15 14:44Z) V1.1: ShinkaAgent E2E - agent explores with shell commands, files in gen_1/, patch_type=agentic
- [~] (2025-12-15 15:50Z) V1.2: Codex backend E2E - PARTIAL: Integration launches Codex correctly, CLI works directly; default model (gpt-4.1-mini) is slow; ShinkaAgent (same arch) passed V1.1
- [ ] V2: Bandit sampling - GPT-5.2 + Claude 4.5 + Gemini 3 Pro rotation
- [ ] V2.5: Circle Packing baseline - MUST hit ≥2.635983 with agentic backend
- [ ] V2.6: Agent Design baseline - MUST hit ≥80% AIME accuracy with agentic backend
- [ ] V2.7: ALE-Bench Lite baseline - MUST hit Mean 1932.1 with agentic backend
- [ ] V2.8: Boids Flocking baseline - Establish and record reference score
- [ ] V3: Multi-file embedding - verify embedding includes all workspace files
- [ ] V4: Novelty detection - verify embedding-based novelty checks work
- [ ] V5: LLM novelty judge - verify LLM-based novelty assessment works
- [ ] V6: LLM scratchpad/meta memory - verify meta summaries generated
- [ ] V9.1: Core evolution logic unchanged (agentic isolated)
- [ ] V9.2: All 13 commits audited for necessity
- [ ] V9.3: No debug/experimental code
- [ ] V9.4: No unnecessary file touches
- [ ] V9.5: Bandit sampling tested with multiple models
- [ ] V9.6: PR description checklist complete

## Surprises & Discoveries

- Observation: Hydra config merging requires `override` keyword when replacing existing defaults at `@_global_` package
  Evidence: Error "Multiple values for evolution@_global_" without override keyword

- Observation: Task config's evo_config block doesn't merge automatically with global evo_config unless using package syntax
  Evidence: boids task_sys_msg was being overwritten by agentic evolution config loaded second

- **CRITICAL BUG (2025-12-15 14:30Z):** PromptSampler doesn't support agentic mode - always sends DIFF prompts
  Evidence: Agent outputs `<DIFF>` format XML instead of bash commands; session logs show LLM trying to use legacy diff format
  Root cause: `sample()` method has no `agentic_mode` parameter; always returns `patch_type` from legacy set
  Impact: Agentic mode completes but "no files changed" because agent never executes shell commands

- **ARCHITECTURE INSIGHT:** In agentic mode, CLI harness owns the system prompt
  Evidence: codexevolve has `AGENTIC_SYS_FORMAT = ""` (empty string)
  Rationale: Codex/Claude/Gemini CLI harnesses inject their own system prompts with tool instructions
  Task context should go in user prompt as "# Task" section, not in system prompt

- **FIX IMPLEMENTED (2025-12-15 14:35Z):** Agentic-aware PromptSampler
  Files modified:
  1. `shinka/prompts/prompts_agentic.py` - Changed AGENTIC_SYS_FORMAT to empty string
  2. `shinka/core/sampler.py` - Added agentic_mode param, implemented _sample_agentic()
  3. `shinka/core/runner.py` - Passed agentic_mode to PromptSampler

## Decision Log

- Decision: Add gpt-5.2 to pricing.py and use it as default model
  Rationale: gpt-5.2 was missing from shinka/llm/models/pricing.py (present in codexevolve). Added pricing entry and REASONING_OAI_MODELS entry.
  Date/Author: 2025-12-14 / Claude

- Decision: Put boids-specific evo_config overrides in variant file rather than task file
  Rationale: Hydra loads variant last, ensuring overrides aren't clobbered by evolution config
  Date/Author: 2025-12-14 / Claude

- Decision: Quality bar (black/isort) only on files changed in this branch
  Rationale: Running formatters on entire codebase would introduce unrelated diffs - bad practice for open source PRs. Only lint/format files we substantively modified.
  Date/Author: 2025-12-15 / User feedback

- Decision: E2E tests must include full auth flows
  Rationale: True end-to-end validation requires testing from logged-out state (Codex headless auth) and UI API key upload (ShinkaAgent). Can't assume pre-existing auth.
  Date/Author: 2025-12-15 / User feedback

- Decision: Empty AGENTIC_SYS_FORMAT with task context in user prompt
  Rationale: CLI harnesses (Codex, Claude CLI, Gemini CLI) inject their own system prompts with tool instructions. Shinka's system prompt would conflict. Task context goes in user prompt as "# Task" section per codexevolve pattern.
  Date/Author: 2025-12-15 / Claude (based on codexevolve research)

## Outcomes & Retrospective

(To be filled after validation completes)

## Context and Orientation

**Branch:** `feat/multi-turn-architecture-clean` (13 commits ahead of origin/main)

**Key Files:**
- `shinka/core/runner.py` - Evolution runner with agentic mode and bandit sampling
- `shinka/edit/shinka_agent.py` - Native ShinkaAgent backend (Protocol-based)
- `shinka/edit/codex_cli.py` - Codex CLI wrapper
- `shinka/edit/agentic.py` - AgenticEditor orchestration
- `configs/evolution/agentic.yaml` - Agentic mode config with llm_models
- `configs/variant/boids_flocking_agentic.yaml` - Multi-file agentic variant

**Terms:**
- **Agentic mode**: Multi-turn editing where an LLM agent can read files, run commands, and make iterative changes
- **ShinkaAgent**: Native agent implementation using LLMClient (not CLI wrapper)
- **Bandit sampling**: UCB algorithm that dynamically selects models based on performance
- **Multi-file workspace**: Task with multiple editable files (e.g., boids with initial.py, boid.py, simulation.py)

## Plan of Work

### Phase 1: Quality Bar (V8)
Run all automated checks to ensure code health before E2E validation.

### Phase 2: Legacy Regression (V7)
Verify legacy single-file mode works without any agentic CLI references.

### Phase 3: Backend Integration (V1)
Validate ShinkaAgent and Codex backends produce actual changes:
- Files must appear in gen_1/ directory
- Score must improve toward baseline targets
- Database must contain new program entries

### Baseline Targets (from codexevolve EXECPLAN) - ALL REQUIRED

| Task | Target Score | Notes |
|------|-------------|-------|
| **Circle Packing (26 circles)** | ≥2.635983 sum of radii | Primary benchmark, strict verifier 2.635977 |
| **Boids Flocking** | Establish baseline | Record best score as reference |
| **Agent Design (AIME)** | ≥80% accuracy | Within ≤10 calls/problem |
| **ALE-Bench Lite** | Mean 1932.1 | ahc039: 3140 (rank 2) |

**ALL baselines must be hit with agentic backend before PR submission. No exceptions.**

### Phase 4: Bandit Sampling (V2)
Verify bandit posteriors are recorded and change over generations.

## Concrete Steps

### V8 - Quality Bar

**IMPORTANT**: Only check files we actually modified in this branch. Running black/isort on the entire codebase would reformat untouched files, which is bad practice for an open source PR. First run `git diff --name-only origin/main` to get the list of changed files, then only lint/format those.

**V8.1 - Pytest**
    uv run pytest tests/ -q

    Expected: All tests pass (39+ passed)

**V8.2 - Ruff (changed files only)**
    # Get list of changed .py files
    git diff --name-only origin/main -- '*.py' | xargs uv run ruff check

    Expected: All checks passed on changed files

**V8.3 - Black (changed files only)**
    # VERIFY FIRST: Run --diff to see what would change
    git diff --name-only origin/main -- '*.py' | xargs uv run black --check --diff

    # If any files would be reformatted that we didn't touch substantively,
    # DO NOT run black on them - that's scope creep for the PR

    Expected: 0 files would be reformatted (or only files we substantively edited)

**V8.4 - Isort (changed files only)**
    # VERIFY FIRST: Run --diff to see what would change
    git diff --name-only origin/main -- '*.py' | xargs uv run isort --check --diff

    # Same rule: don't reformat imports in files we only touched incidentally

    Expected: No import reordering needed (or only in files we substantively edited)

### V7 - Legacy Regression

    rm -rf results/
    uv run shinka_launch variant=circle_packing_example evo_config.num_generations=2

    Validation:
    1. Check logs for NO references to Codex/Gemini/Claude/ShinkaAgent CLI
    2. Verify gen_1 directory exists: ls results/shinka_circle_packing/*/gen_1/
    3. Verify score changes from ~0.96:
       sqlite3 results/shinka_circle_packing/*/evolution_db.sqlite \
         "SELECT generation, combined_score FROM programs ORDER BY generation"
    4. Verify patch type is 'diff' or 'full' (not 'agentic'):
       sqlite3 results/shinka_circle_packing/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.patch_type') FROM programs"

### V1.1 - ShinkaAgent Backend E2E

**Pre-requisite: API key in environment or credential store**
    # Option 1: Environment variable (recommended)
    export OPENAI_API_KEY=sk-...

    # Option 2: Credential file at ~/.shinka/credentials.json
    # {"OPENAI_API_KEY": "sk-..."}

**Run evolution:**
    rm -rf results/
    uv run shinka_launch variant=boids_flocking_agentic evo_config.num_generations=3

    Validation:
    1. Logs show "ShinkaAgent completed task" (not Codex/Gemini/Claude)
    2. Files appear in gen directories:
       ls results/shinka_boids_flocking/*/gen_1/
       ls results/shinka_boids_flocking/*/gen_2/
    3. Multiple files loaded (5 for boids):
       Look for "Checked 5 files" in logs
    4. Score in database:
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, combined_score FROM programs ORDER BY generation"
    5. Patch type is 'agentic':
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.patch_type') FROM programs WHERE generation > 0"
    6. Session logs written:
       ls results/shinka_boids_flocking/*/agent_sessions/*/session_log.jsonl

### V1.2 - Codex Backend E2E (with headless auth from logged-out state)

**Pre-requisite: Test headless auth flow from scratch**
    1. Log out of Codex CLI:
       codex logout
    2. Verify logged out:
       codex auth status  # Should show not authenticated
    3. Run evolution - headless auth should trigger automatically:
       rm -rf results/
       uv run shinka_launch variant=boids_flocking_agentic \
         evo_config.agentic.backend=codex evo_config.num_generations=2
    4. Auth flow should:
       - First try subscription auth (device flow or existing session)
       - Fall back to API key if subscription unavailable
       - Log which auth method was used

    Validation:
    1. Logs show Codex CLI launched AND auth method used
    2. Logs show Codex session completed (not error about auth)
    3. Files appear in gen_1/:
       ls results/shinka_boids_flocking/*/gen_1/
    4. Score in database
    5. Session logs written:
       ls results/shinka_boids_flocking/*/agent_sessions/*/session_log.jsonl

### V2.5-V2.8 - Baseline E2E Tests WITH Bandit Sampling

**These baselines demonstrate that agentic mode + bandit sampling works end-to-end.**

All baseline runs use the 3-provider bandit (GPT-5.2, Claude 4.5 Opus, Gemini 3 Pro) so the system can dynamically select the best-performing model. This proves the bandit improves evolution.

**Pre-requisite:** User must log in and provide API keys for all 3 providers.

#### V2.5 - Circle Packing Baseline (MANDATORY)

Target: ≥2.635983 sum of radii on Circle Packing (26 circles)

    rm -rf results/
    uv run shinka_launch variant=circle_packing_example \
      +evo_config.agentic_mode=true \
      +evo_config.agentic.backend=shinka \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb \
      evo_config.num_generations=50

    # Monitor progress:
    sqlite3 results/shinka_circle_packing/*/evolution_db.sqlite \
      "SELECT MAX(combined_score) FROM programs"

    Validation:
    1. Best score ≥2.635983 (or 2.635977 strict)
    2. Bandit rotates between all 3 providers (check model_name in metadata)
    3. Record run directory, generation count, and which model achieved best score

#### V2.6 - Agent Design Baseline (MANDATORY)

Target: ≥80% accuracy on AIME 2024 within ≤10 calls/problem

    rm -rf results/
    uv run shinka_launch variant=agent_design_example \
      +evo_config.agentic_mode=true \
      +evo_config.agentic.backend=shinka \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb \
      evo_config.num_generations=50

    Validation:
    1. AIME accuracy ≥80%
    2. Within ≤10 calls per problem
    3. Bandit used all 3 providers

#### V2.7 - ALE-Bench Lite Baseline (MANDATORY)

Target: Mean score 1932.1 (ahc039: 3140 rank 2)

    rm -rf results/
    uv run shinka_launch variant=ale_bench_example \
      +evo_config.agentic_mode=true \
      +evo_config.agentic.backend=shinka \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb \
      evo_config.num_generations=50

    Validation:
    1. Mean score ≥1932.1
    2. ahc039 task: ≥3140
    3. Bandit used all 3 providers

#### V2.8 - Boids Flocking Baseline (ESTABLISH)

Establish reference baseline for Boids Flocking task.

    rm -rf results/
    uv run shinka_launch variant=boids_flocking_agentic \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb \
      evo_config.num_generations=50

    Validation:
    1. Record best combined_score achieved
    2. Document as reference baseline for future runs
    3. Score must show improvement from initial (0.96)
    4. Bandit used all 3 providers

**If any baseline not achieved, continue running or investigate model performance.**

### V3 - Multi-File Embedding (Legacy Parity)

The embedding system must consider ALL files in the workspace, not just a single main file.

    # After running V1.1 or V2, check embedding metadata:
    sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
      "SELECT json_extract(metadata, '$.embedding_corpus_meta') FROM programs WHERE generation > 0 LIMIT 1"

    Validation:
    1. `included_files` lists multiple files (initial.py, boid.py, simulation.py, etc.)
    2. `total_bytes` reflects combined size of all workspace files
    3. Embedding changes when ANY file changes (not just primary file)

### V4 - Novelty Detection (Legacy Parity)

Embedding-based novelty checks must work to prevent duplicate programs.

    # Check novelty logs during run - look for similarity scores:
    # "[shinka.core.novelty_judge][INFO] - Top-5 similarity scores: ..."
    # "[shinka.core.novelty_judge][INFO] - NOVELTY CHECK: ..."

    Validation:
    1. Novelty checks run for each new program
    2. Similarity scores computed against existing programs
    3. High-similarity programs rejected (if threshold exceeded)

### V5 - LLM Novelty Judge (Legacy Parity)

When embedding similarity is borderline, LLM judge must assess true novelty.

    # Enable LLM novelty judge and check logs:
    # Look for "LLM novelty check" or similar in logs

    Validation:
    1. LLM judge triggered for borderline similarity cases
    2. Judge uses configured model (not hardcoded)
    3. Decision logged with reasoning

### V6 - LLM Scratchpad / Meta Memory (Legacy Parity)

Meta summaries must be generated to track evolution progress.

    # After run completes, check meta memory:
    cat results/shinka_boids_flocking/*/meta_memory.json

    # Check for meta summary output:
    ls results/shinka_boids_flocking/*/meta_*.txt

    Validation:
    1. `meta_memory.json` exists with program summaries
    2. Meta summary text files generated
    3. Recommendations/insights extracted from evolution history

### V2 - Bandit Sampling (Multi-Provider Frontier Models)

**Must test with all 3 frontier models from different providers:**
- GPT-5.2 (OpenAI)
- Claude Opus 4.5 (Anthropic) - `claude-opus-4-5-20251101`
- Gemini 3 Pro (Google) - `gemini-3-pro-preview`

**Pre-requisite:** User provides API keys for all 3 providers

    rm -rf results/
    uv run shinka_launch variant=boids_flocking_agentic evo_config.num_generations=10 \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb

    Validation:
    1. Logs show bandit selecting from all 3 providers
    2. Each provider hit at least once across 10 generations
    3. Model name varies in database:
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.model_name') FROM programs"
    4. Bandit posteriors update:
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.bandit_posteriors') FROM programs WHERE generation > 0"

## Success Criteria & Validation

| Criterion | Command | Expected | Status |
|-----------|---------|----------|--------|
| V1.1 ShinkaAgent | UI API key upload → `variant=boids_flocking_agentic` | Files in gen_1/, session logs, key upload | [ ] |
| V1.2 Codex | `codex logout` → headless auth → evolution | Auth succeeds, files in gen_1/, session logs | [ ] |
| V2 bandit | `num_generations=10` with GPT-5.2, Claude 4.5, Gemini 3 Pro | All 3 providers hit, posteriors update | [ ] |
| **V2.5 circle packing** | `circle_packing_example +agentic_mode=true` | **≥2.635983 sum of radii** | [ ] |
| **V2.6 agent design** | `agent_design_example +agentic_mode=true` | **≥80% AIME accuracy** | [ ] |
| **V2.7 ALE-Bench** | `ale_bench_example +agentic_mode=true` | **Mean ≥1932.1** | [ ] |
| **V2.8 boids flocking** | `boids_flocking_agentic` | **Establish baseline** | [ ] |
| V3 multi-file embed | Check `embedding_corpus_meta` in DB | `included_files` has multiple files | [ ] |
| V4 novelty detection | Check logs for similarity scores | Novelty checks run, duplicates rejected | [ ] |
| V5 LLM novelty judge | Check logs for LLM novelty assessment | LLM judge triggered for borderline cases | [ ] |
| V6 meta memory | Check `meta_memory.json` and `meta_*.txt` | Summaries and recommendations generated | [ ] |
| V7 legacy | `variant=circle_packing_example` | Score changes, no agentic CLI | [ ] |
| V8.1 pytest | `uv run pytest tests/ -q` | 39+ passed | [ ] |
| V8.2 ruff | `git diff --name-only origin/main -- '*.py' \| xargs ruff check` | Pass on changed files only | [ ] |
| V8.3 black | `git diff ... \| xargs black --check --diff` | No unexpected reformats | [ ] |
| V8.4 isort | `git diff ... \| xargs isort --check --diff` | No unexpected import changes | [ ] |
| V9.1 core unchanged | `git diff origin/main -- runner.py` | Agentic code isolated in conditionals | [ ] |
| V9.2 commits audited | Review 13 commits | All necessary, no scope creep | [ ] |
| V9.3 no debug code | `grep -E "print\(\|TODO\|DEBUG"` | No debug artifacts | [ ] |
| V9.4 minimal changes | `git diff --name-only` | All file changes substantive | [ ] |
| V9.5 bandit multi-provider | GPT-5.2 + Claude 4.5 + Gemini 3 Pro | All 3 providers rotate, posteriors update | [ ] |
| V9.6 PR description | Manual checklist | Robert's 3 requirements mapped | [ ] |

## Idempotence and Recovery

- Each validation run uses `rm -rf results/` to start clean
- Failed runs leave artifacts for debugging; create new timestamped run rather than modifying
- Tests and linters are safe to re-run; clean caches with `rm -rf .pytest_cache .ruff_cache` if needed
- If Hydra launch fails, kill process and check `/tmp/shinka_launch.log` for diagnostics

## Artifacts and Notes

### Commits in Branch

    fdee648 feat: add boids_flocking_agentic variant and fix config merging
    6639b62 feat: integrate bandit sampling with agentic mode
    1fda8e3 fix: hydrate workspace for legacy multi-file patches
    810e318 feat: propagate multi-file workspace between generations
    ec6307e fix: correct embedding corpus args for agentic files
    a860e08 fix: prefer subscription auth for codex
    23915e0 feat: codex headless auth (device + api key)
    ea6e91e fix: harden agentic backends and config
    15d579f fix: Align TerminalRenderer signature with MatplotlibRenderer
    e7faefe fix: Remove embedded script tag breaking HTML parser
    729ac1a feat: Add Boids Flocking multi-file example
    bd46743 feat: Add multi-file diff viewer and agentic node indicator
    e12fe6b feat: Agentic backend core and routing logic

(Evidence logs to be added as validations complete)

## Interfaces and Dependencies

- `shinka/edit/shinka_agent.py`: Native agent implementing `AgentRunner` protocol
- `shinka/edit/agentic.py`: `AgenticEditor.run_agentic_session()` orchestrates workspace setup and agent execution
- `shinka/core/runner.py`: `_run_agentic_edit()` integrates bandit model selection with agentic sessions
- `configs/evolution/agentic.yaml`: Defines `llm_models`, `llm_dynamic_selection: ucb`, `agentic.backend`

---

## V9 - PR Minimalism & Reviewability (Robert's Requirements)

**Goal:** Deliver the smallest, most reviewable PR that meets Robert's 3 requirements:
1. Native control (ShinkaAgent, not black-box CLI wrapper)
2. Multi-file support
3. Backward compatibility

### V9.1 - Verify Core Evolution Logic Unchanged

The legacy (non-agentic) code path must remain IDENTICAL except for the conditional branching into agentic mode.

    # Diff the core runner to ensure agentic additions are isolated
    git diff origin/main -- shinka/core/runner.py | head -200

    # Look for:
    # - All agentic code guarded by `if self.evo_config.agentic_mode:`
    # - No changes to legacy LLM query path
    # - No changes to database schema
    # - No changes to evaluation logic (except agentic evaluator addition)

### V9.2 - Audit Commits for Necessity

Review all 13 commits and verify each is required for the PR:

    git log --oneline origin/main..HEAD

    For each commit, ask:
    1. Is this directly required for native control, multi-file, or backward compat?
    2. Could this be split into a separate PR?
    3. Does this introduce unnecessary scope creep?

    Commits to scrutinize:
    - Any "fix" commits - are they fixing things broken by this PR, or unrelated?
    - Any config changes - are they all necessary?
    - Any visualization/UI changes - strictly required or nice-to-have?

### V9.3 - Remove Debug/Experimental Code

    # Search for debug prints, TODO comments, or experimental flags
    git diff origin/main -- '*.py' | grep -E "(print\(|# TODO|# DEBUG|# HACK|# FIXME)"

### V9.4 - Verify No Unnecessary File Touches

    # List all changed files
    git diff --name-only origin/main

    # For each file, verify the changes are substantive and required
    # Remove any files that only have formatting/import changes

### V9.5 - Bandit Sampling with Frontier Models (Multi-Provider)

**This is not just a config test - we must test bandit rotation across 3 different API providers with their latest frontier models:**

1. **GPT-5.2** (OpenAI)
2. **Claude Opus 4.5** (Anthropic) - model slug: `claude-opus-4-5-20251101`
3. **Gemini 3 Pro** (Google) - model slug: `gemini-3-pro-preview`

**Pre-requisite: User must provide API keys for all 3 providers**

    # Verify API keys are configured:
    # - OPENAI_API_KEY (for gpt-5.2)
    # - ANTHROPIC_API_KEY (for claude-opus-4-5-20251101)
    # - GOOGLE_API_KEY or GEMINI_API_KEY (for gemini-3-pro-preview)

**Run bandit with all 3 frontier models:**

    rm -rf results/
    uv run shinka_launch variant=boids_flocking_agentic evo_config.num_generations=10 \
      'evo_config.llm_models=[gpt-5.2,claude-opus-4-5-20251101,gemini-3-pro-preview]' \
      evo_config.llm_dynamic_selection=ucb

**Validation:**
    1. Logs show bandit selecting from all 3 models across generations
    2. Each provider is hit at least once (verify different API calls)
    3. Database shows model_name varying:
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.model_name') as model FROM programs ORDER BY generation"
    4. Bandit posteriors update based on performance:
       sqlite3 results/shinka_boids_flocking/*/evolution_db.sqlite \
         "SELECT generation, json_extract(metadata, '$.bandit_posteriors') FROM programs WHERE generation > 0"

**This validates:**
- Multi-provider support works
- Bandit UCB algorithm rotates between providers
- Cost tracking works across providers
- No provider-specific bugs in the agentic path

### V9.6 - PR Description Checklist

Before submitting, ensure PR description includes:
- [ ] Summary of what's added (native ShinkaAgent, multi-file, agentic mode)
- [ ] What's NOT changed (legacy mode, database schema, existing examples)
- [ ] How to test (exact commands from this ExecPlan)
- [ ] Robert's 3 requirements explicitly mapped to implementation
- [ ] Known limitations or follow-up work

---

## Change Log

- (2025-12-15 00:20Z) Added legacy parity requirements: V3 multi-file embedding, V4 novelty detection, V5 LLM novelty judge, V6 meta memory/scratchpad. Added session log verification to V1.1/V1.2.
- (2025-12-15 00:10Z) Added V9 PR minimalism section. Updated V2/V9.5 to require 3 frontier models (GPT-5.2, Claude 4.5 Opus, Gemini 3 Pro). Added hard requirements warning at top.
- (2025-12-14 23:35Z) Rewrote ExecPlan following PLANS.md format from codexevolve worktree. Added proper validation criteria based on EXECPLAN_VALIDATION.md baselines. Previous version was too weak - didn't verify files in gen directories, score changes, or database entries.
