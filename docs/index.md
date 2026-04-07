---
title: ShinkaEvolve
hide:
  - toc
---

<div class="hero">
  <div class="hero-copy">
    <p class="hero-kicker">Open-ended program evolution</p>
    <div class="hero-title">
      <img src="media/favicon.png" alt="ShinkaEvolve logo" class="hero-logo">
      <h1>ShinkaEvolve</h1>
    </div>
    <p class="hero-lead">
      Evolve scientific code with LLM-guided mutation, archive-based search,
      async proposal pipelines, and local or cluster-backed evaluation.
    </p>
    <div class="hero-actions">
      <a class="md-button md-button--primary" href="getting_started/">GET STARTED</a>
      <a class="md-button" href="reference/">API REFERENCE</a>
      <a class="md-button" href="https://sakana.ai/shinka-evolve/" target="_blank" rel="noopener">BLOG</a>
      <a class="md-button" href="https://arxiv.org/abs/2509.19349" target="_blank" rel="noopener">PAPER</a>
    </div>
  </div>
</div>

---

## Why ShinkaEvolve

ShinkaEvolve combines LLM mutation operators with an evolutionary archive,
parallel evaluation, and a reproducible task contract. The repository gives you
both the framework primitives and runnable examples — start from a simple
`evaluate.py` + `initial.py` task and scale up to async runs or cluster
workflows.

<div class="feature-grid">
  <div class="feature-card">
    <h3>Runtime</h3>
    <p>
      <code>ShinkaEvolveRunner</code> handles async evolution, proposal/eval
      concurrency, prompt co-evolution, and resumable runs.
    </p>
  </div>
  <div class="feature-card">
    <h3>Config</h3>
    <p>
      Runtime dataclasses, Hydra presets, and agent-oriented
      <code>shinka_run</code> overrides. Compose at any level.
    </p>
  </div>
  <div class="feature-card">
    <h3>Execution</h3>
    <p>
      Run locally, source a project env per job, or launch on SLURM with
      Conda or Docker-backed workers.
    </p>
  </div>
  <div class="feature-card">
    <h3>Inspection</h3>
    <p>
      Lineages, metrics, diffs, throughput, and prompt evolution artifacts
      in the built-in WebUI.
    </p>
  </div>
</div>

---

## Quickstart

```bash
pip install shinka-evolve
shinka_launch variant=circle_packing_example
```

The distribution name is `shinka-evolve`; the import path is `import shinka`.
For source installs and a full first-run walkthrough, see
[Getting Started](getting_started.md).

---

## Two Entry Paths

### Hydra Launcher

Shared Hydra presets, compact override syntax, config-composed workflow:

```bash
shinka_launch \
  task=circle_packing \
  database=island_large \
  evolution=small_budget \
  cluster=local \
  evo_config.num_generations=20
```

### Agent-Friendly CLI

Task directory as the interface — explicit flags, no Hydra group files:

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_agent_run \
  --num_generations 20 \
  --set db.num_islands=2
```

The CLI split and precedence rules are documented in [CLI Usage](cli_usage.md).

---

## What To Read Next

| Page | Content |
|------|---------|
| [Getting Started](getting_started.md) | Install, first run, API usage, troubleshooting |
| [Core Concepts](core_concepts.md) | Task contract, async loop, archives, islands |
| [Configuration](configuration.md) | Runtime dataclasses, presets, override grammar |
| [API Reference](reference/index.md) | Curated reference for runtime modules |
| [Examples](examples.md) | Runnable tasks and recommended entry points |

