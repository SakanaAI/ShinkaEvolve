# Clean checkpoints and deterministic resume

Shinka can preserve its local pseudo-random streams at a clean async boundary.
This is useful for long campaigns, planned shutdowns, and provider quota limits.

## Create and resume

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_checkpointed \
  --num_generations 100 \
  --random-seed 1234 \
  --checkpoint-resume-mode strict
```

Press **Ctrl-C once** to pause new proposals, drain active work, commit SQLite,
publish `checkpoint.pkl`, and exit successfully. A second Ctrl-C keeps the usual
immediate interrupt. Python callers can use `runner.request_checkpoint_and_exit()`.

`num_generations` is the total target and may be increased when resuming. Use
the same results directory and selection-relevant configuration.

| Mode | Behavior |
|------|----------|
| `strict` | Require a clean checkpoint whose source, configuration, runtime, RNG implementations, and database watermark match. Fail before restoring RNG on mismatch. |
| `if_available` | Restore a matching checkpoint; otherwise log and continue with best-effort legacy resume. Default for compatibility. |
| `reseed` | Ignore checkpoint RNG position and start Shinka-owned streams from the configured seed. Recorded as non-deterministic. |

Publication is atomic: the prior generation is kept as `checkpoint.previous.pkl`,
and a SHA-256 checksum is verified on load. `checkpoint_resume.json` reports
whether resume was deterministic. The selection-state checkpoint is published
before bounded W&B shutdown.

## Guarantee boundary

A clean checkpoint stores Python's global `random` state, NumPy's legacy global
RNG, every Shinka-owned NumPy `Generator` (including dynamic model selection),
runner counters, bandit/meta state, and program/prompt database watermarks.
Strict mode also rejects a database that is ahead of or behind the checkpoint.

The guarantee covers Shinka-owned randomness at a drained boundary. It does not
serialize live coroutines, provider requests, or evaluator processes, and it
cannot make external LLM responses deterministic. Exact replay tests therefore
need deterministic fake providers. Python, NumPy, Shinka source, evaluator, or
selection-configuration changes are rejected by strict mode.

!!! warning "Trusted results directories only"

    Checkpoints use Python pickle. Loading a pickle can execute code, so only
    resume from a results directory you trust.
