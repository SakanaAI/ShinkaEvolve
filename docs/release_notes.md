# Release Notes (Unreleased)

These notes summarize local changes since the last commit.

## Highlights

- Added a full async pipeline via `AsyncEvolutionRunner` for concurrent proposal generation and evaluation.
- Introduced prompt co-evolution (system prompt archive, prompt mutation, prompt fitness tracking).
- Expanded island and parent sampling logic, including dynamic island spawning on stagnation.
- Refactored LLM + embedding stack into provider-based modules.
- Major WebUI refresh with dashboard + compare views and new API endpoints.

![Image #1](async_speed.png)

## Cost Budgeting (`max_api_costs`)

- `max_api_costs` is now a first-class runtime budget guard in evolution runners.
- Budget checks use a **committed cost** model:
  - realized DB costs (`api_costs`, `embed_cost`, `novelty_cost`, `meta_cost`)
  - plus estimated cost of in-flight work
- Once the budget is reached, new proposals stop and the runner drains ongoing jobs.
- If `num_generations` is omitted, `max_api_costs` is required to bound the run.

## Added

- Async runtime and helpers:
  - `shinka/core/async_runner.py`
  - `shinka/core/async_summarizer.py`
  - `shinka/core/async_novelty_judge.py`
  - `shinka/database/async_dbase.py`
- Prompt evolution system:
  - `shinka/core/prompt_evolver.py`
  - `shinka/database/prompt_dbase.py`
  - `shinka/prompts/prompts_prompt_evo.py`
- Fix-mode prompt path for incorrect-only populations:
  - `shinka/prompts/prompts_fix.py`
- Island sampling strategies:
  - `shinka/database/island_sampler.py` (`uniform`, `equal`, `proportional`, `weighted`)
- New plotting modules:
  - `shinka/plots/plot_costs.py`
  - `shinka/plots/plot_evals.py`
  - `shinka/plots/plot_time.py`
  - `shinka/plots/plot_llm.py`
- New docs:
  - `docs/async_evolution.md`
  - `docs/design/dynamic_evolve_markers.md`
  - `docs/design/evaluation_cascades.md`
- New example:
  - `examples/game_2048/*`

## Changed

- `EvolutionRunner` and async runner now include:
  - stronger resume behavior (meta memory + bandit state persistence)
  - fix-mode sampling fallback when no correct program exists
  - richer metadata and cost accounting
- Database model expanded with:
  - dynamic island spawning controls
  - island selection strategy config
  - `system_prompt_id` lineage field
- LLM/embedding refactor:
  - `shinka/llm/models/*` replaced by `shinka/llm/providers/*`
  - embeddings moved to `shinka/embed/*`
- Evaluation wrapper (`shinka/core/wrap_eval.py`) now supports:
  - early stopping (`bayesian`, `ci`, `hybrid`)
  - optional plot artifact generation
  - NaN/Inf score guards
- WebUI backend and frontend expanded:
  - new endpoints for summary/count/details/prompts/stats/plots
  - improved WAL retry behavior under active writes
  - added `shinka/webui/index.html` and `shinka/webui/compare.html`

## Packaging and Dependencies

- `pyproject.toml` updates:
  - `google-generativeai` -> `google-genai`
  - added `psutil`
  - pinned `httpx==0.27`
  - setuptools packaging config updated

## Tests Added

- `tests/test_async_complexity_1000.py`
- `tests/test_bandit_persistence.py`
- `tests/test_dynamic_islands.py`
- `tests/test_island_sampler.py`
- `tests/test_prompt_evolution.py`
