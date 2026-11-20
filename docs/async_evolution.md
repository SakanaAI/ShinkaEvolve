# Async Evolution Pipeline

Shinka runs evolution through `ShinkaEvolveRunner`.
Use proposal concurrency to control throughput and emulate prior sync behavior.

---

## Interactive Throughput Demo

Use the controls below to change the conceptual worker-pool sizes for:

- proposal sampling (`max_proposal_jobs`)
- evaluation (`max_evaluation_jobs`)
- database/finalization (`max_db_workers`)

The demo uses proposal jobs directly:

```text
proposal_capacity = sampling_workers
```

This is a teaching model, not a replay of exact runtime data. It is meant to
show how sampling hands candidates off to evaluation while database workers
finalize completed generations.

<div id="async-throughput-demo"></div>

---

## Quick Start

```python
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.launch import LocalJobConfig
from shinka.database import DatabaseConfig


evo_config = EvolutionConfig(
    num_generations=50,
    llm_models=["gpt-5-mini"],
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(),
    max_evaluation_jobs=2,
    max_proposal_jobs=3,  # ceiling; going above max_evaluation_jobs needs controlled oversubscription
    max_db_workers=4,
)

runner.run()
```

In async contexts (for example notebooks/async apps), use:

```python
await runner.run_async()
```

---

## Concurrency Knobs

- `max_evaluation_jobs`: max concurrent evaluation jobs.
- `max_proposal_jobs`: max concurrent proposal generation jobs.
- `max_db_workers`: max async database worker threads.
- `enable_controlled_oversubscription`: adaptive controller for bounded proposal oversubscription.

`max_proposal_jobs=1` gives sequential proposal generation behavior.
All concurrency knobs live on `ShinkaEvolveRunner`.

### What the Runner Grants

At startup the runner checks the requested worker counts against the machine:

| Knob | Limit | Reason |
|------|-------|--------|
| `max_evaluation_jobs` with `LocalJobConfig` | the smaller of the logical CPU count (`os.cpu_count()`) and two per GB of installed RAM | Each local evaluation is a CPU-bound subprocess. |
| `max_evaluation_jobs` with a Slurm job config | none | The jobs run on the cluster, not on the submitting machine. |
| `max_proposal_jobs` | none | A proposal job mostly waits on the LLM. |
| `max_db_workers` | 8 | Threads on one SQLite database. |

A request above a limit is reduced, and the runner logs a warning with the
requested and the granted number:

```text
Concurrency clamp: max_evaluation_jobs requested 16, granted 10 (one CPU-bound evaluation per core, 10 CPU cores)
```

`os.cpu_count()` is the CPU count of the whole machine. In a container or a
Slurm allocation that was given fewer CPUs it is too high, so set
`max_evaluation_jobs` to the number you were given.

With a Slurm job config the runner does not know the cluster's limits. Keep
`max_evaluation_jobs` at or below the number of jobs your account may have
submitted at one time (`MaxSubmitJobs` of your Slurm association or QOS). A
submission that `sbatch` rejects is recorded as a failed generation, and the
LLM calls for its proposal are already paid for. Jobs above a limit on running
jobs wait in the queue, and each holds an evaluation slot until Slurm starts it.

`max_proposal_jobs` is a ceiling, not a target. Without controlled
oversubscription the runner keeps proposals plus running evaluations at
`max_evaluation_jobs`, so a reduced evaluation count also reduces the proposals
in flight, and `max_proposal_jobs` above `max_evaluation_jobs` has no effect.

With `max_api_costs` set, the cap also limits the proposals in flight. A run
whose results directory has no proposal cost on record starts one proposal and
holds the others back until that one has been handed to evaluation or has
failed; then the other proposal jobs start. From then on a proposal starts only
while the cap covers a reservation for it and for every proposal in flight, so
near the cap fewer proposals run than the job counts allow. The total can still
pass the cap by the requests that are in flight when it is reached, and more
jobs mean more of those. See [API Cost Cap](configuration.md#api-cost-cap).

`headless/*` models run one CLI call at a time per runner process, so proposal
jobs beyond the first wait for their turn. Keep `max_proposal_jobs` small with
them.

When sampling/proposal generation is slower than evaluation, set
`max_proposal_jobs > max_evaluation_jobs` and enable controlled oversubscription.
This allows a small backlog of proposals to keep evaluation workers fed without
creating an unbounded queue.

---

## ShinkaEvolveRunner Parameters

```python
ShinkaEvolveRunner(
    evo_config=EvolutionConfig(...),
    job_config=JobConfig(...),
    db_config=DatabaseConfig(...),
    verbose=True,
    max_evaluation_jobs=2,
    max_proposal_jobs=3,
    max_db_workers=4,
)
```

---

## Recommended Settings

| Scale | max_evaluation_jobs | max_proposal_jobs | Notes |
|-------|-------------------|-------------------|-------|
| Sequential-like | 1-4 | 1 | sync-like proposal behavior |
| Small | 2-6 | eval + 1 | good default if eval waits on proposals |
| Medium | 5-20 | eval + 1 to eval + 2 | use adaptive oversubscription |
| Large | 20+ | eval + 2 to eval + 6 | keep bounded with caps |

Local evaluation jobs above the logical CPU count are reduced (see
[What the Runner Grants](#what-the-runner-grants)), so the larger rows need a
Slurm job config or a machine with that many CPUs.

---

## Measured Scaling

Throughput relative to one job, measured on a 10-core laptop (8 performance and
2 efficiency cores, 64 GB RAM) that was also running other programs (1-minute
load average 5 to 18 before the runs). Setup: `examples/circle_packing` with
`shinka_small.yaml`, a CPU-bound evaluation of about 2 seconds, equal evaluation
and proposal jobs, `max_db_workers=4`, and a stub LLM that answers after a fixed
delay with a full rewrite. Embeddings, novelty checks and meta-recommendations
were off. Each cell is the mean of one to three runs of 24 or 40 generations.

| Jobs (evaluation = proposal) | LLM answers in 3 s | LLM answers in 15 s |
|------------------------------|--------------------|---------------------|
| 1 | 11.4 generations/min | 3.5 generations/min |
| 2 | 1.9x | not measured |
| 4 | 3.5x | not measured |
| 8 | 5.9x | 7.0x |
| 10 | 6.4x | not measured |

The stub numbers need a machine with at least as many CPUs as jobs. With 8 or
10 jobs they have not been checked against a real LLM API. Against the OpenAI
API (`gpt-5.4-mini` and `gpt-5.4-nano`, with embeddings and novelty checks on)
the same example gave about 2x with 2 jobs and 3.9x with 4, as the mean of two
runs of 16 generations whose speedups ranged from 1.6x to 2.3x and from 3.6x to
4.1x.

In the stub runs one evaluation took longer the more jobs ran: 2.1 s with 1
job, 2.2 s with 2, 2.5 s with 4, 3.1 s with 8 and 3.8 s with 10. The OpenAI runs
with 2 and 4 jobs did not show this (1.8 to 1.9 s throughout). If your
evaluation scores run time, check its timing at the job count you plan to use.
The runner process itself used 0.12 to 0.19 CPU-seconds per generation in every
run.

---

## Controlled Oversubscription

Adaptive oversubscription uses observed proposal and evaluation timings to
compute a bounded proposal target.

Key settings on `EvolutionConfig`:

- `enable_controlled_oversubscription`
- `proposal_target_mode`
- `proposal_target_min_samples`
- `proposal_target_ratio_cap`
- `proposal_buffer_max`
- `proposal_target_hard_cap`
- `proposal_target_ewma_alpha`

### What Each Oversubscription Setting Does

Oversubscription never increases evaluation concurrency.
`max_evaluation_jobs` still caps concurrent evals.
These settings only control how many proposal/sampling jobs Shinka is willing
to keep in flight ahead of those eval workers.

| Key | What it controls | When to raise it | When to lower it |
|-----|------------------|------------------|------------------|
| `enable_controlled_oversubscription` | Master on/off switch. If `false`, proposal target stays at `max_evaluation_jobs`. Default is `false`. | Turn it on if proposals are slower than evals and workers go idle waiting for new candidates. | Leave it off for predictable sync-like behavior or easier debugging. |
| `proposal_target_mode` | How Shinka chooses the proposal target. `adaptive` uses observed timings. `fixed` uses `max_evaluation_jobs + proposal_buffer_max`. | Use `adaptive` for most runs. Use `fixed` if workload timing is stable and you want deterministic behavior. | Switch away from `fixed` if it overfills the queue; switch away from `adaptive` if you need simpler tuning. |
| `proposal_target_min_samples` | Warmup count before adaptive mode trusts observed timing ratios. Before this, Shinka only adds a small buffer. | Raise if early timings are noisy or unrepresentative. | Lower if you want the controller to react sooner. |
| `proposal_target_ratio_cap` | Upper bound on the observed `sampling_seconds / evaluation_seconds` ratio used by adaptive mode. Prevents extreme spikes from asking for too many proposals. | Raise if proposal generation is consistently much slower than eval and backlog is still too small. | Lower if one slow sample causes too much queued proposal work. |
| `proposal_buffer_max` | Max number of extra proposal jobs allowed above `max_evaluation_jobs`. Primary backlog-size knob. | Raise if eval workers go idle waiting for proposals. | Lower if memory/API pressure grows or proposal backlog gets too large. |
| `proposal_target_hard_cap` | Absolute cap on adaptive/fixed proposal target before applying `max_proposal_jobs`. Useful when `max_proposal_jobs` is high but you want a lower oversub ceiling. | Raise if the controller is hitting the cap too early. | Lower if you want a strict safety stop regardless of timing estimates. |
| `proposal_target_ewma_alpha` | Smoothing factor for timing EWMAs. Higher values react faster; lower values react more slowly but more stably. | Raise if workload phase changes quickly and the controller lags behind. | Lower if proposal target oscillates too much from noisy timings. |

### How the Limits Combine

Think of the final proposal target as:

```text
base target = max_evaluation_jobs
adaptive/fixed target = mode-specific estimate
final target = clamp(
    adaptive/fixed target,
    lower=max_evaluation_jobs,
    upper=min(
        max_evaluation_jobs + proposal_buffer_max,
        proposal_target_hard_cap or max_proposal_jobs,
        max_proposal_jobs,
    ),
)
```

Practical read:

- `max_evaluation_jobs`: eval capacity.
- `max_proposal_jobs`: hard ceiling for proposal workers.
- `proposal_buffer_max`: how far above eval capacity you can go.
- `proposal_target_hard_cap`: extra absolute stop, even if other limits are higher.
- `proposal_target_ratio_cap`: only affects adaptive mode's estimate before clamping.

### Tuning Heuristics

- Eval workers idle often: raise `proposal_buffer_max` first, then maybe `max_proposal_jobs`.
- Backlog too deep: lower `proposal_buffer_max` or `proposal_target_hard_cap`.
- Controller too jumpy: lower `proposal_target_ewma_alpha`.
- Controller too sluggish: raise `proposal_target_ewma_alpha`.
- Startup phase too conservative: lower `proposal_target_min_samples`.
- Startup phase too noisy: raise `proposal_target_min_samples`.

Example:

```python
evo_config = EvolutionConfig(
    num_generations=100,
    llm_models=["gpt-5.4-nano", "gpt-5.4-mini"],
    enable_controlled_oversubscription=True,
    proposal_target_mode="adaptive",
    proposal_target_min_samples=5,
    proposal_target_ratio_cap=2.0,
    proposal_buffer_max=2,
    proposal_target_hard_cap=7,
    proposal_target_ewma_alpha=0.3,
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(),
    max_evaluation_jobs=5,
    max_proposal_jobs=7,
    max_db_workers=4,
)
```

---

## Troubleshooting

- Too many requests: reduce `max_proposal_jobs`.
- Proposal backlog grows too much: lower `proposal_buffer_max` or `proposal_target_ratio_cap`.
- Evaluation workers idle: raise `max_proposal_jobs` modestly and keep controlled oversubscription enabled.
- Memory pressure: lower `max_evaluation_jobs` first (each local evaluation is its own process), then `max_proposal_jobs`.
- DB contention: lower `max_db_workers`.
- File I/O errors: ensure `aiofiles` installed.
