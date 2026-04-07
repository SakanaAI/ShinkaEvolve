# Core Concepts

ShinkaEvolve works best when the task boundary is explicit: one initial program,
one evaluator, one results directory, and a search loop that mutates code while
preserving correctness.

---

## Task Contract

Every runnable task centers on two files:

| File | Purpose |
|------|---------|
| `initial.<ext>` | Seed program with mutable code regions |
| `evaluate.py` | Evaluation harness that executes candidates, validates outputs, returns metrics |

At minimum, the evaluator surfaces a `combined_score` and whether the candidate
is functionally correct. This creates a stable interface for both the Hydra
launcher and the task-directory CLI.

---

## Evolution Loop

Each generation follows the same loop:

1. Sample a parent or inspiration context from the database and archive
2. Ask one or more LLMs to propose a patch or full candidate program
3. Validate and materialize the candidate
4. Run the evaluator and collect metrics
5. Persist the result into the program database
6. Update archive state, island state, and optional meta-memory

The async runner overlaps proposal generation and evaluation to keep worker
slots busy when LLM sampling is slower than the evaluator.

---

## Runtime Layers

The runtime splits into three major configuration domains:

| Domain | Controls |
|--------|----------|
| `EvolutionConfig` | Mutation behavior, model selection, budgets, prompt evolution, async proposal controls |
| `DatabaseConfig` | Archive sizing, island behavior, migration, parent selection, archive ranking |
| `JobConfig` variants | Where and how candidate programs are executed |

See [Configuration](configuration.md) for field-level reference.

---

## Archives and Islands

The database stores both the current population and derived state used to guide
future sampling:

- **Global archive** of high-value programs
- **Island-local populations** to preserve diversity
- **Migration settings** for cross-island transfer
- **Parent and inspiration selection** strategies

This mechanism lets Shinka balance local exploitation with broader search over
qualitatively different candidate programs.

---

## Prompt Co-Evolution

Shinka can evolve not only programs but also the system prompts that generate
them:

- System-prompt archive
- Prompt mutation operators
- Prompt sampling controls
- Prompt fitness tracking over time

Use prompt co-evolution when task quality depends heavily on mutation style or
when different prompts unlock distinct improvement modes.

---

## Execution Modes

| Mode | Description |
|------|-------------|
| Local current interpreter | Default; runs in active Python env |
| Local sourced environment | Via `activate_script` |
| SLURM + Conda | Conda-backed cluster environments |
| SLURM + Docker | Container-backed cluster jobs |

This keeps the search loop decoupled from the execution environment of the
candidate code.

---

## Results and Inspection

Each run writes artifacts into a results directory:

- Persisted program records and metrics
- Candidate code snapshots and diffs
- Timing metadata
- Prompt-evolution artifacts when enabled

Inspect results through the built-in [WebUI](webui.md), notebooks in the
examples directory, or custom post-processing scripts.
