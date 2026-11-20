# Launch API

Defines how candidate programs are executed once generated.

---

## `LocalJobConfig`

Local execution, optionally with a sourced environment or explicit Conda
environment.

::: shinka.launch.job_config.LocalJobConfig
    handler: python
    options:
      show_source: false

---

## `SlurmCondaJobConfig`

SLURM-backed execution with a Conda environment or sourced activation script.

::: shinka.launch.job_config.SlurmCondaJobConfig
    handler: python
    options:
      show_source: false

---

## `SlurmDockerJobConfig`

SLURM-backed execution where the evaluator runs in a container.

::: shinka.launch.job_config.SlurmDockerJobConfig
    handler: python
    options:
      show_source: false

---

## `JobScheduler`

Lower-level scheduler abstraction for submitting and monitoring evaluation jobs
across local and SLURM modes.

::: shinka.launch.scheduler.JobScheduler
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - run
