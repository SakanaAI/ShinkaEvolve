# Changelog

All notable changes to `shinka-evolve` are documented in this file.

## Unreleased

### Added

- Added a new `shinka_models` CLI that inspects the current environment plus discovered `.env` files and reports which priced LLM and embedding models are available to use.

### Changed

- Changed `shinka_models` default output to a compact JSON object with separate `embedding` and `llm` model lists, while `--verbose` now emits provider-level availability details with the same top-level lists.
- Updated the `shinka-run` skill so run planning must validate mutation, meta-recommendation, prompt-evolution, and embedding models against `shinka_models` before launching evolution.

## 0.0.3 - 2026-04-04

### Added

- Added local OpenAI-compatible embedding endpoint support via `evo.embedding_model=local/<model>@http(s)://host[:port]/v1`.
- Added `CONTRIBUTING.md` plus GitHub issue and pull-request templates to document the contribution flow.
- Added Python throughput plotting utilities in `shinka.plots` for generation runtime timelines and normalized occupancy-over-time views.
- Added a durable SQLite `generation_event_log` journal for async generation lifecycle debugging, including stop and persistence-failure events.
- Added regression coverage for the new Python throughput plotting helpers, including pool-slot prep, occupancy math, and legend/layout behavior.
- Added regression coverage for concurrent async completed-job persistence so multi-worker postprocessing throughput stays exercised.
- Added regression coverage for lightweight program summaries and WebUI embed-tab hydration so similarity matrices only render from fully loaded embedding data.

### Changed

- Reworked async completion handling so completed scheduler jobs are detected immediately, evaluation slots are released before persistence finishes, and shutdown now waits for queued completed-job batches plus post-persistence side effects to drain.
- Moved database archive / best-program / island maintenance off the insert hot path via deferred replay hooks, while letting async writers use fresh worker-local connections and merge runtime metadata back into the shared DB state.
- Expanded pipeline timing metadata with post-evaluation queue wait, post-persistence side-effect timing, and summary statistics for end-to-end async throughput analysis.
- Tuned `examples/circle_packing/shinka_long.yaml` for a smaller long-run preset and ignored generated `results*` / `shinka_scale*` artifacts in the repo root.
- Renamed the local backend guide from `docs/support_local_llm.md` to `docs/support_local_models.md` and expanded it to cover local embedding backends alongside local LLMs.
- Refactored async code validation to use a shared subprocess helper across Python, Rust, Swift, JSON, and C++ validators without changing validation behavior.
- Updated `examples/circle_packing/load_results.ipynb` to include the new throughput plots at the bottom of the notebook.
- Updated `examples/circle_packing/load_results.ipynb` and `examples/circle_packing/shinka_long.yaml` for the latest large async circle-packing run analysis setup.
- Refined Python throughput plot legends to use compact centered panels below each subplot for cleaner notebook rendering.
- Reduced the async generation journal to high-signal failure/stop events only so persistence debugging does not add heavy hot-path overhead.

### Fixed

- Fixed completion-time accounting so retried or duplicate-persisted jobs keep the original scheduler completion timestamp instead of inflating evaluation duration.
- Fixed the async job monitor to finalize cleanly once the generation target is reached, even when no jobs remain active at the polling boundary.
- Fixed high-concurrency SQLite persistence regressions by covering deferred maintenance replay, multi-writer overlap, and shutdown drain behavior with new recovery and database tests.
- Fixed async proposal scheduling so `num_generations` is now a hard cap on assigned proposal generations instead of launching extra `gen_*` attempts to compensate for failed or discarded work.
- Fixed async evaluation slot lifecycle bugs so local evaluation concurrency no longer exceeds `max_evaluation_jobs` through stale double-release of reassigned worker slots.
- Fixed retry-time completion accounting so successful DB retries refresh async progress before the hard generation-budget stop condition is evaluated.
- Fixed async database retry races by treating in-flight `source_job_id` inserts as already claimed, preventing duplicate persisted programs while timed-out writes are still finishing in worker threads.
- Fixed async resume/recovery bookkeeping so restarted runs continue from the number of persisted completed programs instead of stopping early when failed proposals or hung local evals left gaps in generation IDs.
- Fixed WebUI meta-analysis labeling so `meta_*.txt` snapshots are presented as meta updates / processed-count checkpoints instead of misleading generation numbers.
- Fixed duplicate-retry recovery so already-persisted jobs replay post-persistence side effects exactly once, restoring missing meta-memory updates after DB timeouts.
- Fixed SQLite persistence stability by increasing busy timeouts and the outer async DB-add timeout for long high-concurrency runs.
- Fixed Python throughput plot preparation so frames without optional metadata columns like `is_island_copy`, `patch_name`, or `model_name` still render correctly.
- Fixed legacy throughput accounting in both the Python plotter and WebUI Throughput tab so reused worker lanes no longer show impossible peaks like `31/20`.
- Fixed the WebUI embed tab so summary-only loads or single lazily hydrated programs no longer produce a misleading 1x1 similarity matrix instead of the full run.

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
