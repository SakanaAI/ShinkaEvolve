# Changelog

All notable changes to `shinka-evolve` are documented in this file.

## TBD

### Added

- Added configurable tournament parent selection in PR #217, with
  `tournament_top_k=8` and `tournament_num_draws=3` defaults. Draw distinct
  contestants uniformly from the selected island's highest-scoring correct,
  finite-scored archive entries and mutate a highest scorer, breaking winner
  ties uniformly. Empty eligible archives fall back to the island's eligible
  population; pool and draw sizes shrink to available candidates. The default
  parent selection strategy remains `weighted`.
- Expanded the bundled model catalog with OpenAI GPT-6, Claude 5.5/Fable 5.1,
  Gemini 4 Argon, Gemini 3.5 Flash-Lite and 3.6–3.8 Flash, and DeepSeek V4 Flash
  and V4 Pro. Added provider-specific reasoning and thinking controls, refreshed
  pricing, and historical-price overlays for upstream-removed IDs in PR #214.
  Gemini 3.7 Flash and Gemini 4 Argon use introductory 2026 pricing. Existing
  default model pools remain unchanged. Thanks @anantgar for Gemini 3.6 support.
- Added optional Weights & Biases logging with resumable run IDs, per-individual
  scores, timing and cost metrics, and final run summaries in PR #149. PR #177
  adds monotonic population/island snapshots and a secured metadata boundary.
  Thanks @anantgar.
- Added Verilog/SystemVerilog as a first-class evolution target in PR #137,
  including `.sv` task detection, syntax validation, EVOLVE-BLOCK marker support,
  failure-artifact rendering, and a self-contained RTLLM example. Thanks @Tyronita.
- Added startup pricing refresh from models.dev with conditional HTTP caching,
  validated offline fallback, provider-qualified model discovery, and per-run
  pricing snapshots in PR #141. The tutorial now includes a no-cost catalog
  preflight. Thanks @rodmarkun.
- Added evaluation output controls: `JobConfig.eval_verbose` suppresses framework
  progress output, and `DatabaseConfig.max_stdout_log_chars` limits persisted
  `stdout_log` metadata to its tail while keeping full logs on disk. Quiet mode
  (`run_shinka_eval(..., verbose=False)`) also suppresses result-save framework
  output. Thanks @marcopirazzini.
- Added SLURM support for `numeric_threads_per_job`, applying numeric-library thread caps consistently across local and SLURM evaluation jobs. Thanks @marcopirazzini.

### Changed

- Reused provider SDK clients and their HTTP connection pools across compatible
  requests, with explicit credential/configuration isolation, process-safe sync
  caching, event-loop-scoped async caching, and cache shutdown helpers.
- Reduced Python complexity-analysis overhead by parsing each candidate AST once
  while preserving the existing metrics contract. Thanks @dexhunter.
- Relaxed the exact HTTPX dependency pin to a `>=0.27` compatibility lower
  bound in issue #180. Thanks @htmai-880.
- Removed the automatic Claude Code Review pull-request workflow.

### Fixed

- Fixed prompt evolution resume inserting another seed when only generation-zero
  prompts exist, retaining stored prompt IDs, fitness history, and costs.
- Fixed custom prompt-model sampler settings being forwarded as raw provider
  arguments, allowing configured token limits and reasoning settings to work
  for both diff and full prompt mutations.
- Fixed billed empty OpenAI replies being omitted from budgets across single,
  batch, proposal, novelty, prompt, and meta-analysis calls in PR #211
  (issue #210). Output exhaustion stops proposal retries; failed charges survive
  resume, and recovered replies retain complete billed usage. Thanks @prath47.

- Fixed sync and async meta-summary metadata alignment and generation ordering
  after failed batch responses in PR #208. Thanks @lll-hhh.
- Fixed duplicated API, embedding, novelty, and meta-analysis costs on initial
  and spawned island copies in PR #215, preserving original spend and cumulative
  cost plots. Thanks @guudewie.
- Fixed population selection: archive inspirations stay unique across fallback
  selection (PR #202), multi-criteria ranking ties cannot let worse newcomers
  displace better programs (PR #199), and island migration selects all migrants
  before moving programs to preserve quotas and elites (PR #200).
  Thanks @flyingdoubleG.
- Fixed chart-to-program selection and cumulative-max step timing in PR #201.
  Thanks @guudewie.
- Fixed proposal metadata and LLM request settings diverging by reusing each
  proposal's sampled kwargs in PR #213. Thanks @yuiandyuki.
- Fixed evaluated-program imports for dataclasses with postponed annotations
  and pickling program-defined result types in PR #212. Thanks @yuiandyuki.
- Fixed circle-packing validation and scoring: invalid packings retain raw data
  and diagnostics while receiving the correct score in PR #205 (issue #197),
  and the notebook validator rejects negative radii and non-finite values in
  PR #209 (issue #203). Thanks @Yushu-Li, @iamgahr, @lll-hhh, and @maximusrufus.
- Local OpenAI-compatible endpoints fail without proposal retries on HTTP
  401/403/404, with a typed endpoint failure preserved in `failure.json`, in
  PR #204 (issue #196). Thanks @ToolongTD and @iamgahr.
- Fixed run completion reporting: `shinka_run` exits 1 when evolution produces
  no evolved program; seed-only runs and partial runs with evaluated candidates
  retain exit 0. The final database summary separates generation progress, seeds,
  island copies, and evolved programs, and labels row correctness as including
  copies.
- Fixed evaluation timeouts to start when evaluation begins, so slow LLM proposals
  no longer consume `LocalJobConfig.time` and cause immediate candidate termination
  in PR #198. Thanks @fortunto2.
- Fixed patch application and code extraction: shifted SEARCH/REPLACE patches
  preserve relative indentation and dedents (PR #190; thanks @yurekami),
  `extract_between` matches fence tags literally to avoid corrupting C++ rewrites
  or raising `re.error` on Python 3.10 (PR #193; thanks @MaxFreedomPollard), and
  empty SEARCH insertions preserve line boundaries and EVOLVE-BLOCK-END markers
  for consecutive patches and block-comment languages such as Wolfram
  (issue #183; thanks @Atharva-Kanherkar).
- Fixed `ThompsonSampler.decay()` to avoid invalid-value warnings while arms
  remain unsampled in PR #192. Thanks @yurekami.
- Fixed provider response parsing: Anthropic dispatches by content-block type to
  avoid crashes and truncation for redacted, thinking-only, and multi-block
  responses (PR #188; thanks @Atharva-Kanherkar). OpenAI Responses finds assistant
  text by content type without mistaking reasoning or tool text for an answer.
- Fixed pricing and usage accounting: Anthropic and DeepSeek sync/async queries
  agree on token costs, separate thinking tokens, and safely fall back to zero
  cost for models missing from the catalog. Runtime models.dev refreshes preserve
  pinned embedding-price overrides, and CSV reasoning flags handle whitespace
  and mixed value types.
- Fixed Gemini retry handling so unsupported structured-output requests fail
  immediately, synchronous retries are paced, and sync/async calls share the
  same default thinking budget.
- Fixed Rust candidate validation on stable toolchains by replacing the
  nightly-only parser flag with isolated, non-mutating `rustfmt` syntax checks
  in PR #179. Thanks @Atharva-Kanherkar.
- Fixed Azure OpenAI LLM client construction to use the v1 base URL contract,
  avoiding doubled `/openai/v1/openai` request paths and 404 responses reported
  in issue #147.
- Fixed OpenAI retry handling so ShinkaEvolve respects provider `retry_after` hints from transient Cloudflare/API 5xx responses.
- Fixed provider request compatibility: omitted unsupported `temperature` for
  DeepSeek V4 thinking calls, Claude Opus 4.8 on Anthropic and Bedrock, and GPT-5
  reasoning models on the Responses API. DeepSeek V4 receives thinking-mode
  controls and `reasoning_effort`, and its retry logs include model kwargs.
- Fixed Google GenAI client setup to detect broken IPv6 connectivity to Google API hosts and prefer IPv4 when needed.

## 0.0.7 - 2026-06-02

### Added

- Added Gemini 3.5 Flash pricing to the Google LLM pricing catalog.
- Added Claude Opus 4.8 pricing entries for the Anthropic API and Amazon Bedrock (`us.anthropic.claude-opus-4-8`) in the LLM pricing catalog.
- Added Wolfram Language as a first-class evolution target: registry entries (`wolfram`, `wl`, `wls`, `mathematica`), code-fence and EVOLVE-BLOCK marker support, and end-to-end coverage in `apply_diff` / `apply_full`.
- Added EVOLVE-BLOCK marker validation in `shinka.edit.marker_validation`, catching the LLM failure mode where a block-comment-language marker is emitted without its closing delimiter (Wolfram, Markdown), which would silently trap the candidate body inside a comment.
- Added the `examples/wolfram_gcd_sum` task: deoptimized seed for `S(N) = sum_{i,j in 1..N} GCD(i,j)`. The evaluator calibrates the baseline on every run by timing the seed through the same `RepeatedTiming` harness as the candidate, and the Wolfram-side timeout is configurable via `WOLFRAM_GCD_MAX_SECONDS`.
- Added Go as a first-class evolution target with `go`/`golang` language registration, `.go` task detection, WebUI failure-artifact support, regression coverage, and the `examples/go_collatz_steps` task.

### Fixed

- Fixed the WebUI dashboard so runs that have completed only the initial generation remain visible instead of being filtered out as empty results.
- Fixed the WebUI Ensembling tab so single-model Headless runs render visible data points and keep the top y-axis tick labels in view.

## 0.0.6 - 2026-05-03

### Added

- Added Headless CLI-backed LLM provider support via `headless/<agent>` model strings, defaulting to `npx -y @roberttlange/headless` for subscription-backed Codex, Claude, and other local agent calls.
- Added Headless startup validation, prompt artifacts, usage/cost parsing, and a `examples/sine_approx_headless` task showing API-free mutation calls with embeddings disabled.
- Added Vertex AI authentication support for Gemini LLM and embedding clients in PR #125. Thanks @wu375.
- Added async-runner validation for configured LLM and embedding model environment access before run artifacts are created in PR #127. Thanks @RobertTLange.
- Added GPT-5.5 and GPT-5.5 Pro entries to the OpenAI LLM pricing catalog.
- Added Fortran evolution support, including language detection, patch application, validation, visualization metadata, and a compiled heat-diffusion example in PR #131.

### Fixed

- Fixed bandit sampler resume from legacy or changed `llm_models` state so saved per-arm arrays are resized, name-aligned when possible, and cost-aware UCB range state remains active after loading `bandit_state.pkl` in PR #130, addressing issue #129.
- Fixed the WebUI embedding similarity heatmap so hydrated programs with empty embeddings no longer leave the tab stuck on `Loading full embedding data...`, and stale cached full-program data is refetched after summary updates.

## 0.0.5 - 2026-04-22

### Added

- Added the GitHub Pages documentation website for `shinka-evolve`.
- Added interactive async-throughput and UCB bandit-selection demos to the documentation website.
- Added dashboard sorting controls to the local WebUI so result cards can be reordered by the active setting.
- Added a `Hide Plot` / `Show Plot` toggle for the WebUI Throughput tab runtime timeline while keeping the plot visible by default.
- Added Claude Opus 4.7 pricing entries for the Anthropic API and Amazon Bedrock (`anthropic.claude-opus-4-7`) in the LLM pricing catalog.

### Changed

- Changed `shinka_run` startup output to use a minimal `Shinka CLI` banner while other launch paths keep the full gradient banner.
- Changed `enable_controlled_oversubscription` to default to `false` across the shared `EvolutionConfig` baseline and packaged Hydra evolution presets.
- Changed the WebUI meta header to show the active results directory directly in the info panel.
- Changed the WebUI Throughput tab runtime timeline to scale its height with worker-lane count so each worker keeps a dedicated visible row.

### Fixed

- Fixed async run-time regressions from proposal-failure persistence by keeping terminal failed proposals in `attempt_log` plus `failure.json` artifacts instead of inserting them into the main `programs` table during evolution runs.
- Fixed the local WebUI to render failed proposal nodes from `attempt_log` plus `failure.json` so failure lineage remains visible without storing synthetic failed programs in the main results database.
- Fixed failed terminal proposal and prompt-evolution cost accounting so the runtime API budget counter now reflects those spend buckets, and added runtime-timeline metadata for failed proposal nodes rendered from `attempt_log`.
- Fixed bandit summary tables to preserve readable `local/<model>` and `openrouter/<model>` labels while stripping endpoint and API-key query details from local OpenAI-compatible model labels.
- Fixed the LLM bandit sampling summary to use the same 120-column table width as the program, patch, and other Rich summaries.
- Fixed the default `AsymmetricUCB` bandit summary to omit the `div` and `log mean` columns for a more compact Rich table.
- Fixed the compare view so negative best scores remain visible instead of being clamped away by the WebUI summary logic.
- Fixed documentation links and metadata URLs to use the deployed GitHub Pages path capitalization (`/ShinkaEvolve/`).
- Fixed oversubscription regression coverage and docs so adaptive proposal backlog behavior is now clearly opt-in instead of implied by defaults.
- Fixed Gemini client timeout handling so the shared second-based timeout is converted to the millisecond unit expected by `google-genai`, avoiding accidental `1.2s` read timeouts on long-running requests.
- Fixed async runtime regressions from sampling/evaluation worker-lane persistence by falling back to timestamp-based throughput lane inference instead of storing those lane IDs in program metadata.
- Fixed local WebUI cache staleness by disabling browser caching for the main HTML shells (`index.html`, `viz_tree.html`, `compare.html`) served by the local visualization server.
- Fixed WebUI Throughput tab hydration so `right_tab=throughput` restores reliably after data loads and the generation runtime timeline renders pool stages using timestamp-inferred sampling/evaluation lanes on the current Plotly build.

## 0.0.4 - 2026-04-06

### Added

- Added a new `shinka_models` CLI that inspects the current environment plus discovered `.env` files and reports which priced LLM and embedding models are available to use.
- Added per-model `api_key_env` support for local OpenAI-compatible LLM and embedding backends, allowing credentials to be selected inline via `local/<model>@http(s)://...?...api_key_env=ENV_VAR`.

### Changed

- Changed `shinka_models` default output to a compact JSON object with separate `embedding` and `llm` model lists, while `--verbose` now emits provider-level availability details with the same top-level lists.
- Updated the `shinka-run` skill so run planning must validate mutation, meta-recommendation, prompt-evolution, and embedding models against `shinka_models` before launching evolution.
- Improved local async runtime scaling by launching evaluation subprocesses with the active project interpreter, capping per-process numeric-library thread fan-out, and reducing local monitor polling latency.
- Moved prompt-fitness percentile recomputation off the prompt side-effect hot path and onto a debounced background task using fresh read-only database connections.
- Updated the circle-packing scaling presets to run for `100` generations and reduced `max_patch_resamples` to `1` for the small / medium / large benchmark configs.

### Fixed

- Fixed adaptive proposal targeting so invalid `proposal_target_hard_cap` values below evaluation capacity no longer silently disable oversubscription.
- Fixed prompt-percentile background refresh to avoid SQLite thread-affinity failures when recomputing prompt fitness during async runs.
- Fixed Windows Unicode handling for async candidate-file I/O, diff summaries, and best-path exports by forcing UTF-8 reads and writes on LLM-generated artifacts.

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
