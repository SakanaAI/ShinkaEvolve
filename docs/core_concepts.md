# Core Concepts

ShinkaEvolve works on repository candidates. The task boundary is one seed git
repository, one evaluator, one results directory, and a search loop that mutates
allowed paths while preserving correctness.

---

## Task Contract

Every runnable task centers on a seed repository and evaluator:

| Artifact | Purpose |
|----------|---------|
| `seed_repo/` | Git repository used as the parent for generated worktrees |
| `evaluate.py` | Evaluation harness that accepts `--repo_path`, validates outputs, returns metrics |

At minimum, the evaluator surfaces a `combined_score` and whether the candidate
is functionally correct. This creates a stable interface for both the Hydra
launcher and the task-directory CLI.

Repo-backed evolution uses the same evaluator result contract, but the candidate
is a worktree rather than a single source file. Evaluators receive `--repo_path`
and may run with the candidate repo as their working directory. In repo mode,
the coding agent decides which cheap validation or smoke checks are worth
running before Shinka hands the candidate to the authoritative evaluator.

---

## Evolution Loop

Each generation follows the same loop:

1. Sample a parent or inspiration context from the database and archive
2. Ask one or more LLMs to propose a patch or, in repo mode, invoke a coding
   agent inside an isolated worktree
3. Validate and materialize the candidate
4. Run the evaluator and collect metrics
5. Persist the result into the program database
6. Update archive state, island state, and optional meta-memory

The async runner overlaps proposal generation and evaluation to keep worker
slots busy when LLM sampling is slower than the evaluator.

In repo mode, each candidate also writes `.shinka/individual.md`. This strict
summary is the dense representation of the individual: it captures parent
lineage, the essence of the diff, changed files, validation performed, risks,
and minimal snippets. Shinka stores and embeds this summary instead of embedding
entire repositories.

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

## Model Selection

When `EvolutionConfig.llm_models` contains multiple mutation models, Shinka can
shift sampling probability across them with dynamic bandit selection.

- Reward-side utility from observed candidate improvements
- Exploration bonus for under-sampled models
- Optional cost-aware blending to prefer cheaper models when quality is close

Use the interactive [UCB1 Bandit LLM Selection](bandit_selection.md)
to see how `cost_aware_coef` changes the selection posterior over time.

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
