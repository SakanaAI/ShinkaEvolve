# Core Runtime API

Primary runtime objects for constructing and running an evolution loop from Python.

---

## `EvolutionConfig`

Controls mutation behavior, model selection, budgets, prompt evolution, and
async proposal targeting.

::: shinka.core.config.EvolutionConfig
    handler: python
    options:
      show_source: false

---

## `ShinkaEvolveRunner`

Main async runtime. Coordinates proposal generation, evaluation submission,
persistence, and side-effect handling.

::: shinka.core.async_runner.ShinkaEvolveRunner
    handler: python
    options:
      members:
        - __init__
        - run
      show_source: false

---

## `run_shinka_eval`

Helper for evaluators: standard way to execute candidate programs and aggregate
metrics.

::: shinka.core.wrap_eval.run_shinka_eval
    handler: python
    options:
      show_source: false

---

## Supporting Runtime Types

Lower-level components for customization:

- `PromptSampler`
- `MetaSummarizer`
- `NoveltyJudge` / `AsyncNoveltyJudge`
- `SystemPromptEvolver`
- `SystemPromptSampler`

Available from `shinka.core`. Most integrations should start with the runner +
config objects above.
