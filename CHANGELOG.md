# Changelog

All notable changes to `shinka-evolve` are documented in this file.

## 0.0.3 - TBD

### Changed

- Reworked async completion handling so completed scheduler jobs are detected immediately, evaluation slots are released before persistence finishes, and shutdown now waits for queued completed-job batches plus post-persistence side effects to drain.
- Moved database archive / best-program / island maintenance off the insert hot path via deferred replay hooks, while letting async writers use fresh worker-local connections and merge runtime metadata back into the shared DB state.
- Expanded pipeline timing metadata with post-evaluation queue wait, post-persistence side-effect timing, and summary statistics for end-to-end async throughput analysis.
- Tuned `examples/circle_packing/shinka_long.yaml` for a smaller long-run preset and ignored generated `results*` / `shinka_scale*` artifacts in the repo root.

### Fixed

- Fixed completion-time accounting so retried or duplicate-persisted jobs keep the original scheduler completion timestamp instead of inflating evaluation duration.
- Fixed the async job monitor to finalize cleanly once the generation target is reached, even when no jobs remain active at the polling boundary.
- Fixed high-concurrency SQLite persistence regressions by covering deferred maintenance replay, multi-writer overlap, and shutdown drain behavior with new recovery and database tests.

## 0.0.2 - 2026-03-22

### Added

- Added adaptive proposal oversubscription controls and documentation for bounded async proposal backlogs.
- Added a new `Throughput` tab in the WebUI with runtime timeline, worker occupancy, normalized occupancy, occupancy distribution, completion-rate, and utilization summaries.
- Added regression coverage for WebUI runtime timeline and Throughput-tab structure.

### Changed

- Improved async runtime accounting so evaluation timing starts at evaluation-slot acquisition instead of scheduler submission.
- Improved runtime timeline rendering in the WebUI, including better legend placement and spawned-island copy deduplication for resource-usage plots.
- Improved the embedding similarity matrix layout so large runs preserve cell size and scroll cleanly instead of collapsing visually.
- Expanded `docs/async_evolution.md` with detailed explanations of controlled oversubscription settings and tuning heuristics.

### Fixed

- Prevented duplicate async program persistence for the same completed scheduler job by treating `source_job_id` writes as idempotent.
- Fixed inflated runtime timeline peaks caused by counting spawned-island copies as separate runtime jobs.
- Fixed runtime timeline legend overlap and related layout issues in the WebUI.

## 0.0.1 - 2026-03-12

First PyPI release for `shinka-evolve`.

### Added

- Initial PyPI release for `shinka-evolve`.
- Trusted publishing workflow via GitHub Actions.
- Packaged Hydra presets and release artifact checks for PyPI builds.
- Added a full async pipeline via `ShinkaEvolveRunner` for concurrent proposal generation and evaluation.
- Added prompt co-evolution support, including a system-prompt archive, prompt mutation, and prompt fitness tracking.
- Added island sampling strategies via `shinka/database/island_sampler.py` (`uniform`, `equal`, `proportional`, `weighted`).
- Added fix-mode prompts for incorrect-only populations.
- Added new plotting modules:
  - `shinka/plots/plot_costs.py`
  - `shinka/plots/plot_evals.py`
  - `shinka/plots/plot_time.py`
  - `shinka/plots/plot_llm.py`
- Added new documentation:
  - `docs/async_evolution.md`
  - `docs/design/dynamic_evolve_markers.md`
  - `docs/design/evaluation_cascades.md`
- Added the `examples/game_2048` example.

### Changed

- Refactored the API around `ShinkaEvolveRunner` and the async evolution pipeline.
- Added prompt co-evolution, expanded island logic, provider-based LLM and embedding modules, and a major WebUI refresh.
- Preserved original shorthand launch syntax such as `variant=...`, `task=...`, `database=...`, and `cluster=...`.
- Expanded island and parent sampling logic, including dynamic island spawning on stagnation.
- Refactored the LLM and embedding stack into provider-based modules.
- `ShinkaEvolveRunner` gained stronger resume behavior, fix-mode sampling fallback, and richer metadata/cost accounting.
- Database model expanded with dynamic island spawning controls, island selection strategy config, and `system_prompt_id` lineage.
- `shinka/core/wrap_eval.py` gained per-run process parallelism, deterministic ordering, clearer worker error surfacing, early stopping, optional plot artifacts, and NaN/Inf score guards.
- WebUI backend/frontend expanded with summary/count/details/prompts/stats/plots endpoints plus dashboard and compare views.
- README and install docs were updated to prefer PyPI install and document `--config-dir` for user-defined presets.
- `pyproject.toml` packaging/dependency updates:
  - `google-generativeai` -> `google-genai`
  - added `psutil`
  - pinned `httpx==0.27`
  - updated setuptools packaging config

### Cost Budgeting

- `max_api_costs` became a first-class runtime budget guard in evolution runners.
- Budget checks use committed cost:
  - realized DB costs (`api_costs`, `embed_cost`, `novelty_cost`, `meta_cost`)
  - plus estimated cost of in-flight work
- Once the budget is reached, new proposals stop and the runner drains ongoing jobs.
- If `num_generations` is omitted, `max_api_costs` is required to bound the run.
