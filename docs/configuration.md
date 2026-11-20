# Configuration Guide

This document is synced to the current code + config files in this repo.

---

## Default Layers (Source of Truth)

Configuration values are resolved in this order (later wins):

1. Dataclass defaults in code:
   - `shinka/core/config.py` (`EvolutionConfig`)
   - `shinka/database/dbase.py` (`DatabaseConfig`)
   - `shinka/launch/job_config.py` (`LocalJobConfig`, `SlurmDockerJobConfig`, `SlurmCondaJobConfig`)
2. Hydra preset YAMLs in `shinka/configs/`
3. Task/cluster/variant overrides from Hydra composition
4. CLI overrides (`shinka_launch ... key=value`, or `shinka_run --set ...`)
5. Authoritative `shinka_run` flags (`--results_dir`, `--num_generations`)

---

## Runtime Config Objects

### EvolutionConfig (`shinka.core.EvolutionConfig`)

Concurrency is configured on `ShinkaEvolveRunner`, not on `EvolutionConfig`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_sys_msg` | `Optional[str]` | `"You are an expert optimization and algorithm design assistant. Improve the program while preserving correctness and immutable regions."` | Task-specific system prompt. |
| `patch_types` | `List[str]` | `['diff', 'full', 'cross']` | Patch formats; supports `diff`, `full`, `cross`. |
| `patch_type_probs` | `List[float]` | `[0.6, 0.3, 0.1]` | Sampling probabilities for `patch_types` (must sum to 1). |
| `num_generations` | `int` | `50` | Target number of generations. |
| `max_patch_resamples` | `int` | `3` | Max patch resample loops per novelty attempt. |
| `max_patch_attempts` | `int` | `1` | Max attempts to produce a syntactically valid patch. |
| `job_type` | `str` | `'local'` | Job backend: `local`, `slurm_docker`, `slurm_conda`. |
| `language` | `str` | `'python'` | Language tag for prompts + file handling. |
| `llm_models` | `List[str]` | `['gpt-5-mini', 'gemini-3-flash-preview', 'gemini-3.1-pro-preview', 'gpt-5.4']` | Mutation model pool. |
| `llm_dynamic_selection` | `Optional[Union[str, BanditBase]]` | `'ucb'` | Dynamic model selection (`fixed`, `ucb`, `ucb1`, `thompson`, or bandit object). |
| `llm_dynamic_selection_kwargs` | `dict` | `{'cost_aware_coef': 0.5}` | kwargs forwarded to selected bandit. |
| `llm_kwargs` | `dict` | `{'temperatures': [0.0, 0.5, 1.0], 'max_tokens': 16384}` | Sampling settings for mutation LLM calls. See [LLM sampling settings](#llm-sampling-settings-llm_kwargs). |
| `meta_rec_interval` | `Optional[int]` | `10` | Generation interval for meta recommendations. |
| `meta_llm_models` | `Optional[List[str]]` | `None` | Model pool for meta-recommendations. |
| `meta_llm_kwargs` | `dict` | `{}` | kwargs for meta-recommendation LLM calls. |
| `meta_max_recommendations` | `int` | `5` | Max recommendations produced per meta step. |
| `sample_single_meta_rec` | `bool` | `True` | Whether to sample one recommendation when multiple exist. |
| `embedding_model` | `Optional[str]` | `'text-embedding-3-small'` | Embedding model for code similarity. Also supports `local/<model>@http(s)://host[:port]/v1` for local OpenAI-compatible embedding endpoints, with optional `?api_key_env=ENV_VAR` for per-model credentials. |
| `init_program_path` | `Optional[str]` | `'initial.py'` | Initial program path. |
| `results_dir` | `Optional[str]` | `None` | Results directory; auto-assigned when `None`. |
| `enable_wandb_logging` | `bool` | `False` | Mirror evolution metrics to W&B. Existing SQLite and WebUI logging remains enabled. Install the `wandb` extra first. |
| `wandb_project` | `Optional[str]` | `'shinka-evolve'` | W&B project used when wandb logging is enabled. |
| `wandb_entity` | `Optional[str]` | `None` | Optional W&B entity/team. |
| `wandb_group` | `Optional[str]` | `None` | Optional W&B run group. |
| `wandb_name` | `Optional[str]` | `None` | Optional W&B run name; defaults to the results directory name. |
| `wandb_mode` | `Optional[str]` | `None` | Optional W&B mode, e.g. `offline` or `disabled`. |
| `wandb_tags` | `List[str]` | `[]` | Optional W&B tags. |
| `wandb_notes` | `Optional[str]` | `None` | Optional W&B run notes. |
| `wandb_dir` | `Optional[str]` | `None` | Optional local W&B directory; defaults to `results_dir`. |
| `wandb_run_id` | `Optional[str]` | `None` | Optional W&B run ID; otherwise generated and persisted in the results directory. |
| `wandb_resume` | `str` | `'allow'` | W&B resume policy used with the persisted run ID. |
| `wandb_config` | `Dict[str, Any]` | `{}` | Extra W&B config values merged into the run config. |
| `max_novelty_attempts` | `int` | `3` | Max novelty loops per generation. |
| `code_embed_sim_threshold` | `float` | `0.99` | Similarity threshold used by novelty checks. |
| `novelty_llm_models` | `Optional[List[str]]` | `None` | Optional novelty-judge model pool. |
| `novelty_llm_kwargs` | `dict` | `{}` | kwargs for novelty-judge LLM calls. |
| `use_text_feedback` | `bool` | `False` | Include text feedback in mutation prompts. |
| `max_api_costs` | `Optional[float]` | `None` | Stops new paid API requests once known charges reach this amount in USD. |
| `enable_controlled_oversubscription` | `bool` | `False` | Enable bounded proposal oversubscription when proposal generation is slower than evaluation. |
| `proposal_target_mode` | `str` | `'adaptive'` | Proposal target controller mode: `adaptive` or `fixed`. |
| `proposal_target_min_samples` | `int` | `5` | Minimum completed timing samples required before adaptive targeting activates. |
| `proposal_target_ratio_cap` | `float` | `2.0` | Maximum sampling/evaluation ratio used by the adaptive controller. |
| `proposal_buffer_max` | `int` | `2` | Maximum extra proposal jobs above `max_evaluation_jobs`. |
| `proposal_target_hard_cap` | `Optional[int]` | `None` | Absolute cap for the adaptive proposal target. |
| `proposal_target_ewma_alpha` | `float` | `0.3` | EWMA smoothing factor for proposal/evaluation timing estimates. |
| `inspiration_sort_order` | `str` | `'ascending'` | Inspiration ordering (`ascending`, `chronological`, `none`). |
| `evolve_prompts` | `bool` | `False` | Enable system-prompt evolution. |
| `prompt_patch_types` | `List[str]` | `['diff', 'full']` | Patch formats for prompt evolution. |
| `prompt_patch_type_probs` | `List[float]` | `[0.7, 0.3]` | Sampling probabilities for prompt patch formats. |
| `prompt_evolution_interval` | `Optional[int]` | `None` | Prompt-evolution interval in generations. |
| `prompt_archive_size` | `int` | `10` | Prompt archive size. |
| `prompt_llm_models` | `Optional[List[str]]` | `None` | Prompt-evolution model pool (falls back to `llm_models`). |
| `prompt_llm_kwargs` | `dict` | `{}` | Prompt-client sampler settings, converted to provider arguments for each mutation. |
| `prompt_ucb_exploration_constant` | `float` | `1.0` | UCB exploration constant for prompt sampler. |
| `prompt_epsilon` | `float` | `0.1` | Epsilon-greedy exploration for prompt sampler. |
| `prompt_evo_top_k_programs` | `int` | `3` | Number of top programs used during prompt evolution. |
| `prompt_percentile_recompute_interval` | `int` | `20` | Generations between prompt percentile recomputations. |

Resuming prompt evolution reuses stored prompts, including when only the
initial generation-zero prompt exists. Existing IDs and fitness data are retained.

Billed OpenAI replies count toward `max_api_costs` even when they contain no
usable text. A reply exhausted by `max_output_tokens` stops proposal retries;
increase `llm_kwargs.max_tokens` or lower `reasoning_efforts` before retrying.
The async runner checks known charges before paid requests and retries,
including embeddings, novelty checks, prompt evolution, and final meta-analysis.
Requests already running can complete and add costs above the limit. Zero-price
requests can finish their stages after the limit, including unpriced local
routes. The runner restores persisted charges before starting new work and
pauses paid requests while retrying a failed charge write. If it still cannot
save the charge, it stops the run and retries the pending write during shutdown.

Cancellation of initialization, proposal generation, prompt evolution, or
meta-analysis retains known charges on resume. Small accounting writes finish before
cancellation propagates, including when cancellation interrupts an insert.

When calling the LLM clients directly, a billed failure can return a
`QueryResult` with `content=None`, its cost and usage, and an
`incomplete_reason`. Check `content` before using the answer. Batch clients
also retain these billed results so callers can include their costs.

#### LLM sampling settings (`llm_kwargs`)

`llm_kwargs`, `meta_llm_kwargs`, `novelty_llm_kwargs` and `prompt_llm_kwargs`
take the same keys. Each key holds one value or a list; a list is sampled from
for each query.

| Key | Meaning | When the key is missing |
|-----|---------|-------------------------|
| `temperatures` | Sampling temperature. Not sent to every model; see below. | `0.75` |
| `max_tokens` | Output token limit. Reasoning tokens count toward it where the provider bills them as output. Capped at 128000 for Claude models with adaptive thinking and at 64000 for older Claude reasoning models. | `4096` |
| `reasoning_efforts` | Reasoning or thinking effort, one of the names below. | `disabled` |

The missing-key values are the client's own defaults, not the `llm_kwargs`
default of the table above. They apply to `meta_llm_kwargs`,
`novelty_llm_kwargs` and `prompt_llm_kwargs`, which default to `{}`, and to an
`llm_kwargs` you replace as a whole: `{"reasoning_efforts": "high"}` alone
runs with `max_tokens` 4096.

`temperatures` does not apply to Claude models, on Anthropic or Bedrock: the
API no longer accepts a temperature, so they sample at its default. It is also
not sent to OpenAI reasoning models, DeepSeek V4 models, and the Gemini models
that take a thinking level. OpenRouter models listed as reasoning models always
get temperature 1.0. Azure deployments differ from the OpenAI models they
serve: see the note below the effort table.

`reasoning_efforts` accepts `disabled`, `none`, `min`, `minimal`, `low`,
`medium`, `high`, `xhigh` and `max`, in any letter case. An unset value, `""`
and an unquoted `off` in a YAML file mean `disabled`. `auto` is deprecated and
treated as `disabled` with a warning. Any other name raises an error when the
LLM client is built, before the first request. Every name is accepted for
every model and mapped onto what the model offers:

| Models | Mapping |
|--------|---------|
| Models the pricing catalog does not list as reasoning models (for example `gpt-4.1`, `claude-opus-4-7`, `claude-opus-4-8` and `local/...` endpoints) | Ignored. |
| OpenAI and OpenRouter reasoning models before GPT-6 | Sent as named, except `min` → `low` and `max` → `high`. `none`, `minimal` and `xhigh` exist only on some of these models: where the model lacks the level, the name is still sent and the provider decides. `disabled` sends no effort, so the provider's default applies, or `low` to a model that cannot run without reasoning. |
| GPT-6 family, on OpenAI or as an Azure deployment named after the model | `disabled` and `none` → `none`, or `low` on GPT-6 Astra and GPT-6.1 Sol, which require reasoning. `min` and `minimal` → `low`. The rest as named. |
| Claude with adaptive thinking (Fable 5.1 and 5.2, Opus 5.5, Sonnet 5.5, Haiku 5.5) | `disabled`, `none`, `min` and `minimal` → `low`. The rest as named. `disabled` and `none` also set thinking to `disabled` on Haiku 5.5 and to `between_tools` on Sonnet 5.5; on Fable and Opus it stays adaptive. |
| Models with a thinking-token budget (Claude models before adaptive thinking that the catalog lists as reasoning models, such as Opus 4.6 and Sonnet 4.6, but not Opus 4.7 and 4.8; Gemini models without thinking levels) | `min` and `minimal` 1024 tokens, `low` 2048, `medium` 4096, `high` 8192, `xhigh` 12288, `max` 16384. A budget that is not below `max_tokens` falls back to 1024: with `max_tokens: 16384`, `max` gives 1024, so set `max_tokens` above the budget you want. `disabled` and `none` turn thinking off, or set a 128-token budget on Gemini 2.5 Pro and Gemini 3 Pro preview, which cannot turn it off. |
| Gemini with thinking levels (3.5 Flash-Lite, 3.6 to 3.8 Flash, Gemini 4) | `min`, `minimal` and `low` → `LOW`, except `min` and `minimal` → `MINIMAL` on 3.5 Flash-Lite; `medium` → `MEDIUM`; `high`, `xhigh` and `max` → `HIGH`. `disabled` and `none` send no level. |
| DeepSeek reasoning models | `disabled` and `none` turn thinking off. `max` → `max`. Every other name → `high`. |

Azure deployments (`azure-<deployment name>`) are looked up in the catalog
under the prefixed name, and the catalog lists no Azure models. A deployment
named after a GPT-6 model (`azure-gpt-6-luna`) follows the GPT-6 row. Every
other deployment falls under the first row, also when it serves a reasoning
model: for `azure-gpt-5-mini` or `azure-o4-mini`, `reasoning_efforts` is
ignored without a warning and a `temperatures` value is sent, where
`gpt-5-mini` and `o4-mini` get the effort and no temperature.

#### LLM endpoint failures

Provider errors that cannot succeed on a retry are not retried:

- HTTP 401, 402, 403 and 404 mean that the endpoint, key, model or account is
  wrong for every request. So do the account-wide failures that arrive under
  other statuses: `API_KEY_INVALID`, `invalid_api_key`, `insufficient_quota`
  and Anthropic's "credit balance is too low".
- Any other 4xx except 408, 409, 425, 429 and 499 rejects that one request: it
  is not sent again as it is. The proposal is resampled like any other failed
  patch, and if no attempt succeeds the generation is recorded as a failed
  proposal and the run goes on. An OpenRouter 403 counts as a rejected request,
  because OpenRouter answers 403 when its moderation flags the input.
- A provider's `x-should-retry` response header takes precedence for every
  error outside the first bullet: `false` makes it a rejected request, `true`
  keeps it retried.

When the endpoint of a mutation model fails, the proposal is tried again with
another model of the pool, and a warning names the model that failed. Each
switch uses one of the proposal's resamples (`max_patch_resamples`), and the
failed model is left out for that proposal only. A proposal that ends without
an answer from any endpoint counts as an endpoint failure. A single-model pool
stops the run on the first one. A larger pool records the first two in a row as
failed proposals and stops the run on the third; a proposal that gets an answer
restarts the count.

A stopped run lets the evaluations already submitted finish, logs
`ASYNC EVOLUTION STOPPED: no LLM endpoint was reachable`, and leaves the last
generation unconsumed. Fix the credentials, the model access or the endpoint,
then start the run again with the same `results_dir` to resume it.
`runner.evolution_failed` is `True` after such a run and `shinka_run` exits
with status 1, as both do when a run of more than one generation produces no
evolved program. `shinka_launch` does not report the stop in its exit status.

W&B logging examples:

```bash
# Install the optional integration.
pip install 'shinka-evolve[wandb]'

# Authenticate online runs. In CI, provide this through a secret manager.
export WANDB_API_KEY=<your-api-key>

# Add W&B metrics and a compact final table alongside the WebUI database.
shinka_run --task-dir examples/circle_packing --results_dir results/circle_wandb --num_generations 20 \
  --set evo.enable_wandb_logging=true \
  --set evo.wandb_project=shinka-evolve
```

Population and island snapshots use the monotonic
`population/evaluated_count` axis. They include counts, correctness, scores,
cumulative `cost/*`, and cumulative `timing/*_total`. Raw evaluated candidates
use `score/individual` and `individual/*` fields without binding those events to
generation. Administrative island copies remain in population/table counts but
are excluded from evaluated count, candidate history, and costs. The compact
`population/final_table` omits program code, embeddings, and unrestricted
metadata. When a results directory is resumed, its `.wandb_run_id` is reused
with `wandb_resume='allow'` by default. Online mode uses credentials from
`wandb login` or `WANDB_API_KEY`; set `wandb_mode=offline` to record locally
without uploading. W&B failures are non-fatal and do not alter the existing
database or WebUI path.

The integration disables W&B console, Git, code, requirements, machine, and
system-stat capture. Its run config contains only an explicit allowlist of safe
scalar experiment settings; `wandb_config` remains an explicit extension and is
recursively redacted for secret-like keys. Arbitrary `wandb_config` values cross
the upload boundary when their keys are not recognized as sensitive, so do not
put secrets or private data there. Candidate events omit private metrics and
include at most 64 public numeric metrics whose full keys are at most 128
characters; secret-like keys are discarded. Population snapshots use a compact
database aggregate on a dedicated serialized telemetry worker, run at most once
every five seconds, and overlapping refresh requests are coalesced. If the
bounded raw-candidate queue saturates, the newest candidate event is dropped
with a rate-limited warning; population and final snapshots remain authoritative.

### DatabaseConfig (`shinka.database.DatabaseConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `Optional[str]` | `None` | File name of the run database inside `results_dir`; `programs.sqlite` when unset. See the note below. |
| `num_islands` | `int` | `2` | Number of islands. |
| `archive_size` | `int` | `40` | Global archive size cap. |
| `max_stdout_log_chars` | `Optional[int]` | `None` | Keep only the last N characters of an evaluation's stdout in the program metadata. `None` stores it in full; the complete log always stays on disk. |
| `elite_selection_ratio` | `float` | `0.3` | Fraction of elite inspirations. |
| `num_archive_inspirations` | `int` | `1` | Number of archive inspirations sampled. |
| `num_top_k_inspirations` | `int` | `1` | Number of top-k inspirations sampled. |
| `migration_interval` | `int` | `10` | Generations between migration events. |
| `migration_rate` | `float` | `0.0` | Fraction of programs migrated at migration events. |
| `island_elitism` | `bool` | `True` | Preserve best programs on islands. |
| `enforce_island_separation` | `bool` | `True` | Restrict inspiration sampling to source island. |
| `island_selection_strategy` | `str` | `'uniform'` | Island sampler: `uniform`, `equal`, `proportional`, `weighted`. |
| `enable_dynamic_islands` | `bool` | `False` | Enable stagnation-triggered island spawning. |
| `stagnation_threshold` | `int` | `100` | No-improvement generations before spawn. |
| `island_spawn_strategy` | `str` | `'initial'` | Spawn seed: `initial`, `best`, `archive_random`. |
| `island_spawn_subtree_size` | `int` | `1` | Number of copied programs when spawning. |
| `parent_selection_strategy` | `str` | `'weighted'` | Parent selector: `weighted`, `power_law`, `beam_search`, `tournament`. |
| `tournament_top_k` | `int` | `8` | Highest-scoring eligible archive entries considered by tournament selection. |
| `tournament_num_draws` | `int` | `3` | Distinct uniform contestants; mutate one with the highest score. Must not exceed `tournament_top_k`. |
| `exploitation_alpha` | `float` | `1.0` | Power-law strength for parent selection. |
| `exploitation_ratio` | `float` | `0.2` | Probability of selecting from archive. |
| `parent_selection_lambda` | `float` | `10.0` | Sigmoid sharpness for weighted parent selection. |
| `num_beams` | `int` | `5` | Beam count for beam-search parent selection. |
| `archive_selection_strategy` | `str` | `'fitness'` | Archive replacement strategy: `fitness` or `crowding`. |
| `archive_criteria` | `Dict[str, float]` | `{'combined_score': 1.0}` | Weighted criteria for fitness archive scoring. |

A run stores its programs in `<results_dir>/<db_path>`, or in
`<results_dir>/programs.sqlite` when `db_path` is unset, as in the shipped
presets and examples. Rules for `db_path` in a run:

- Only the file name counts. A new database always sits directly in
  `results_dir`, because the prompt database, the meta files, the web UI and
  the analysis helpers look next to it. An absolute path or a path with
  directories is reduced to its file name and the runner logs a warning: set
  `results_dir` to move a run.
- Give the name a lower-case `.sqlite` or `.db` suffix. The web UI, `load_programs_to_df`
  and the resume rule below recognise database files by it; the runner warns
  about any other name. `prompts.sqlite` and `prompts.db`, in any letter case,
  belong to the prompt database and are refused, as is a value that names no
  file (`..`).
- A run never starts at generation 0 next to programs it did not ask for. If
  the configured file has no programs but `programs.sqlite` or a database of the
  first releases does, the run resumes in that one and logs a warning. The
  first-release places are `evolution_db.sqlite`, a file named `None`, and the
  nested path `<results_dir>/<db_path>` for a `db_path` with directories. Use a
  new `results_dir` to start over.
- Programs under any other name are not adopted. If the configured file has no
  programs and another database in `results_dir` does (the run was started with
  a `db_path` the configuration no longer has, or somebody left a copy there),
  the runner stops with an error that lists the files: set `db_path` to the one
  to continue.
- A configured database that exists but cannot be read (damaged, locked, not
  permitted, or with a `programs` table of another layout) is not swapped for
  another file in the directory. The run fails when it opens it. A file without
  a `programs` table counts as having no programs.

The database presets carry `db_path: null`, so `db_config.db_path=<name>` works
as a Hydra override. With `shinka_run`, use `--set db.db_path=<name>`.

`load_programs_to_df(results_dir)`, `python -m shinka.plots.code_path_anim` and
the `shinka-inspect` skill look in the same places. Because reading changes
nothing, they also load a database under an unknown name when it is the only
one in the directory. When `load_programs_to_df` loads another file than
`programs.sqlite`, it prints `Loading <name>: programs.sqlite holds no programs.`
With several such databases none of them guesses: `load_programs_to_df` prints
the error and returns `None`, and the skill stops with it. Both take the path
of the database file instead of the directory. A `ProgramDatabase` you open
yourself uses `db_path` as the full file path.

`island_selection_strategy` picks the island a parent is sampled from. `uniform`
treats all islands alike and `equal` picks the island with the fewest correct
programs. `proportional` uses a Boltzmann distribution over each island's best
score, which depends only on score differences. `weighted` divides the best
score by the number of correct programs on the island; if any island's best
score is negative, the scores are first shifted to
`score - worst + (best - worst)`, so the best island weighs twice the worst.

Tournament selection chooses the mutation parent, leaving archive retention and
inspiration selection unchanged. Set `parent_selection_strategy: tournament`,
`tournament_top_k: 8`, and `tournament_num_draws: 3` for Tournament(8,3).
Only correct, finite-scored programs are eligible. The pool is the selected
island's top K archive entries by `combined_score`; an empty eligible archive
falls back to correct programs in that island. Both pool and draw sizes shrink
to available candidates during startup. Ties at the pool boundary use program
ID order; tied tournament winners are selected uniformly.

With distinct scores, rank r wins with probability
`C(N-r, D-1) / C(N, D)`, where N and D are the effective pool and draw sizes.
Tournament(8,3) gives ranks 1–8 shares of 37.5%, 26.8%, 17.9%, 10.7%, 5.4%,
1.8%, 0%, and 0%. See `examples/circle_packing/shinka_tournament.yaml` for a
complete CLI configuration.

### Job Configs (`shinka.launch.*JobConfig`)

`JobConfig` base fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_program_path` | `Optional[str]` | `'evaluate.py'` | Evaluation script path. |
| `extra_cmd_args` | `Dict[str, Any]` | `{}` | Extra CLI args forwarded to the eval script as `--key value`. Values reach it literally: `$VAR` and `~` are not expanded. |
| `eval_verbose` | `bool` | `True` | Print the per-run evaluation banners to stdout. Passed to the evaluation process as `SHINKA_EVAL_VERBOSE`. |
| `numeric_threads_per_job` | `Optional[int]` | `None` | Thread cap for numeric libraries (OMP, BLAS, MKL, ...) in each evaluation process. `None` leaves the libraries' defaults on Slurm; for local jobs the runner sets it to `cpu_count // max_evaluation_jobs`. |

`LocalJobConfig` adds:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | `Optional[str]` | `None` | Wall-clock limit for one evaluation, in any format listed under [Time limits](#time-limits-time). Unset: the [hang limit](#hang-limit-for-local-jobs-without-time) applies. |
| `conda_env` | `Optional[str]` | `None` | Optional conda env for local execution. |
| `activate_script` | `Optional[str]` | `None` | Optional sourceable env script, e.g. `.venv/bin/activate`. |
| `python_executable` | `Optional[str]` | `None` | Interpreter that runs the evaluation script. `None` uses the interpreter running Shinka. Ignored when `conda_env` or `activate_script` is set. |

`SlurmDockerJobConfig` adds:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str` | `'ubuntu:latest'` | Docker image. |
| `image_tar_path` | `Optional[str]` | `None` | Optional image tar for upload/load. |
| `docker_flags` | `str` | `''` | Extra docker flags. |
| `partition` | `str` | `'gpu'` | SLURM partition. |
| `time` | `str` | `'01:00:00'` | SLURM time limit, passed to `sbatch --time`. See [Time limits](#time-limits-time). |
| `cpus` | `int` | `1` | CPU request. |
| `gpus` | `int` | `1` | GPU request. |
| `mem` | `Optional[str]` | `'8G'` | Memory request. |

`SlurmCondaJobConfig` / `SlurmEnvJobConfig` add:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conda_env` | `str` | `''` | Conda environment name. |
| `activate_script` | `Optional[str]` | `None` | Sourceable env script path, e.g. `.venv/bin/activate`. |
| `modules` | `Optional[List[str]]` | `None` | Modules to load (normalized to `[]` at runtime). |
| `partition` | `str` | `'gpu'` | SLURM partition. |
| `time` | `str` | `'01:00:00'` | SLURM time limit, passed to `sbatch --time`. See [Time limits](#time-limits-time). |
| `cpus` | `int` | `1` | CPU request. |
| `gpus` | `int` | `1` | GPU request. |
| `mem` | `Optional[str]` | `'8G'` | Memory request. |

`conda_env` and `activate_script` are mutually exclusive.

#### Time limits (`time`)

`time` takes the numeric formats of `sbatch --time`, for local and Slurm jobs:

| Format | Example | Meaning |
|--------|---------|---------|
| `minutes` | `90` | 90 minutes |
| `minutes:seconds` | `10:30` | 10 minutes 30 seconds |
| `hours:minutes:seconds` | `01:00:00` | 1 hour |
| `days-hours` | `1-12` | 1 day 12 hours |
| `days-hours:minutes` | `1-12:30` | 1 day 12 hours 30 minutes |
| `days-hours:minutes:seconds` | `1-12:30:15` | 1 day 12 hours 30 minutes 15 seconds |

A bare number is minutes, not seconds. As in Slurm, fields are not
range-checked: `00:90:00` is 90 minutes. Slurm job configs also accept
`unlimited`, `infinite` and `-1`. `LocalJobConfig` rejects those: leave `time`
unset for a local job without a fixed limit. A local `time` of `"0"`, given as
a string, is a zero-second limit: an evaluation is killed the first time the
runner checks on it. The number 0 (an unquoted `time: 0` in YAML,
`+job_config.time=0` with Hydra) counts as unset. The value is checked when the
job config is created, so an invalid `time` is rejected before any evaluation
starts: the config class raises `ValueError`, and `shinka_launch` and
`shinka_run` report it as a startup error.

Quote the value in YAML files: `time: "10:00"`. YAML reads an unquoted `10:00`
or `1:30:00` as the base-60 number 600 or 5400, which is then taken as minutes
(10 hours and 90 hours), and an unquoted `030` as the octal number 24. Clock
values that start with a zero (`00:10:00`), `--set job.time=...` and Hydra
command-line overrides keep their text.

With `shinka_launch` and the local preset (`cluster=local`, the default),
`job_config` has no `time` key. Add one with `+job_config.time=00:05:00`;
without the `+`, Hydra stops with `Could not override 'job_config.time'`. The
`time` of a task preset sits under `distributed_job_config` and reaches Slurm
jobs only, so a local `shinka_launch` run has just the hang limit until you
add a `time`.

Who enforces the limit depends on the job type:

- **Local jobs.** The clock starts when the evaluator is launched. The runner
  kills an evaluation that runs longer than `time`, together with its process
  group, and stores the generation as an incorrect program.
- **Slurm jobs.** `time` goes to `sbatch`, and Slurm enforces it from the moment
  the job starts running. The runner puts no clock of its own on a Slurm job,
  so time spent waiting in the queue does not count.

#### Hang limit for local jobs without `time`

When `time` is unset, the runner still stops local evaluations that never
return. It learns the limit from the run:

- Once an evaluation has finished with a correct result, the limit is
  `5 x longest successful evaluation + 60 s`. Crashes and incorrect results are
  left out, so a candidate that fails after one second cannot shrink the limit.
  This limit has no cap, and only a longer success raises it: a candidate that
  needs more than five times the longest success so far is killed every time.
  Set `time` if your runtimes vary that much.
- Before the first success the limit is provisional:
  `5 x longest evaluation that has ended + 60 s`, whatever its outcome, and at
  most one hour. Evaluations the limit killed count too, so a limit learned
  from an early crash grows about fivefold with each kill instead of killing
  every slower candidate. When it reaches one hour the runner logs a warning:
  set `time` if your evaluations need longer. The first success replaces the
  provisional limit, which can lower it.
- The seed evaluation runs before anything is known. Only `time` bounds it.
- A resumed run evaluates no seed. It starts from the evaluation time stored
  with its best program when that program is correct. Otherwise it has no limit
  until one of its own evaluations has ended.

An evaluation over the limit is logged as `Hung job detected`, killed with its
process group, and stored as an incorrect program. The limit in use is
logged as `Hang limit for local evaluations: ...` each time it changes. Slurm
jobs get no hang limit.

#### Evaluation results

Every evaluation must leave two files in its results directory, for the seed
and for later generations, local or Slurm. `run_shinka_eval` writes both, with
`metrics.json` first. An evaluator that writes them itself has to follow the
same contract:

- `correct.json` is a JSON object whose `correct` is a JSON boolean (`true` or
  `false`, not `"true"` or `1`). An optional `error` string says what failed.
- `metrics.json` is a JSON object. `combined_score` is a finite number, not
  NaN, an infinity, `null`, a string or a boolean; a missing `combined_score`
  counts as 0.0. `public` and `private` are JSON objects when present. `text_feedback`
  is optional.

A program is stored as correct only when both files can be read and are valid,
`correct` is `true` and, for local jobs and the Slurm seed, the evaluator
exited with status 0. Otherwise:

- A result that cannot be trusted is stored as an incorrect program with score
  0.0 and `metadata.evaluation_failed`: a file is missing or unreadable,
  `correct` is not a boolean, a correct program has unusable metrics, or a
  local evaluator reported a correct program and exited non-zero. For
  generations after the seed, the diagnostic and the evaluator's own
  `text_feedback` or `error` go to `text_feedback`, and what the evaluator
  reported is kept in `metadata.rejected_evaluation` (`correct`, `metrics`,
  `exit_status`). A failed seed stores the diagnostic only.
- A program reported as incorrect is stored as reported, with its feedback and
  metrics, whatever the exit status. Unusable metric values are replaced by
  defaults (score 0.0, empty `public` or `private`) and noted in `stderr_log`.
  The seed is the exception: a seed evaluator that exits non-zero is stored
  as a failed evaluation whatever it reported. An evaluator that exits
  non-zero for an incorrect program therefore loses its feedback on the seed.

A finished job is kept when its results cannot be read or stored at first:

- A results read that raises is retried, five attempts in all.
- A results read that hangs, for example on a stalled network file system, is
  waited for once (30 seconds) and then checked without blocking for up to 150
  seconds in all, so it holds up the other jobs only during those first 30
  seconds.
- Storing the program row is retried the same way, five attempts in all.

The attempts follow each other within a fraction of a second, so they cover a
momentary failure, not an outage. When they run out, the generation is recorded
as a failed program (`metadata.results_missing` and
`metadata.evaluation_failed`), also when its results had been read. If that row
cannot be written either, the job is logged as `JOB LOST` and has no row.

#### Slurm seed evaluation

Generation-zero Slurm evaluations require a verified terminal state of
`COMPLETED` and `ExitCode=0:0` before accepting result files. Verification
matches the numeric job ID and the unique name set at submission, so a
retained record from an earlier allocation with the same ID cannot validate it.
Each requeued execution writes in its own directory. Result loading also
requires the latest execution's completion receipt, so earlier files cannot
validate a restart that produced no results. Verification remembers the latest
restart count seen in Slurm and waits for its receipt within the retry window.
Once `squeue` reports a terminal or absent job, `sacct` checks the allocation.
If accounting cannot verify it, `scontrol --oneliner show job` checks the
controller's retained job record in the same attempt. After a requeue, the
controller can identify the current execution by its restart count when
accounting still describes an earlier execution. A terminal failure verified
for the current execution is rejected. This supports clusters without
accounting while the controller still retains the completed job. Missing,
inaccessible, ambiguous,
or otherwise unverified status from both sources fails the seed with a
diagnostic; partial artifacts are never evidence of successful completion.
Slurm local fallback jobs instead use the actual subprocess exit status.

Synchronous Slurm evaluations limit each `sbatch`, `squeue`, `sacct`, and `scontrol`
command to 30 seconds. Three consecutive status lookup errors fail the seed;
a successful lookup resets that counter. Pending, running, and suspended jobs
continue waiting in the queue. Slurm applies the `time` setting to job execution.
Docker image preparation for synchronous submission also limits each cold-cache
`docker pull` and `docker save` to 30 seconds. If either times out, the batch job
uses its existing image-load or pull path.
The `squeue` error `Invalid job id specified` triggers terminal verification
through accounting or the controller. Verification allows six rounds, ten
seconds apart, with at most one accounting and one
controller command per round, including command errors and timeouts.

#### Slurm evaluations after the seed

Later generations are submitted with `sbatch` and tracked through the queue:

- Tracked jobs are polled together with one `squeue` call, at most every 5
  seconds and with a 30-second timeout.
- A job is finished when it has left the queue. A job id the controller no
  longer knows (`Invalid job id specified`) counts as finished.
- A failed lookup finishes no job. While the controller is down or `squeue`
  times out, every tracked job stays active; the run waits and goes on when the
  controller answers. The outage is logged as a warning, then as an error every
  5 minutes while it lasts.
- A job that is held, or has spent 30 minutes in the queue in any state other
  than `RUNNING`, is reported in a warning with its state and reason, for
  example `12345 PENDING (Priority) for 30 min`. The warning repeats every 30
  minutes. The runner does not cancel the job: it holds its evaluation slot
  until Slurm starts it or removes it from the queue, or until you cancel it
  with `scancel`.
- The results of a job that has left the queue are read and checked as
  described under [Evaluation results](#evaluation-results). Slurm reports no
  exit status to the runner for these jobs, so the exit-status rule does not
  apply to them.

---

## Hydra Presets

### Evolution Presets

All `shinka/configs/evolution/*.yaml` set runner-level concurrency at the top level and override `EvolutionConfig` defaults only for listed `evo_config` keys.

#### `shinka/configs/evolution/small_budget.yaml`

```yaml
max_evaluation_jobs: 1
max_proposal_jobs: 2
max_db_workers: 2

evo_config:
  patch_types: ["diff", "full"]
  patch_type_probs: [0.5, 0.5]
  num_generations: 20
  max_patch_attempts: 10
  llm_models: ["gpt-4.1"]
  llm_dynamic_selection: null
  embedding_model: "text-embedding-3-small"
  enable_controlled_oversubscription: false
  results_dir: ${output_dir}
```

#### `shinka/configs/evolution/medium_budget.yaml`

```yaml
max_evaluation_jobs: 4
max_proposal_jobs: 6
max_db_workers: 2

evo_config:
  patch_types: ["diff", "full", "cross"]
  patch_type_probs: [0.6, 0.3, 0.1]
  num_generations: 50
  max_patch_resamples: 3
  max_patch_attempts: 1
  llm_models:
    - "gpt-5-mini"
    - "gemini-3-flash-preview"
    - "gemini-3.1-pro-preview"
    - "gpt-5.4"
  llm_dynamic_selection: ucb
  llm_dynamic_selection_kwargs:
    cost_aware_coef: 0.5
  llm_kwargs:
    temperatures: [0.0, 0.5, 1.0]
    max_tokens: 16384
  meta_rec_interval: 10
  embedding_model: "text-embedding-3-small"
  code_embed_sim_threshold: 0.99
  enable_controlled_oversubscription: false
  results_dir: ${output_dir}
```

#### `shinka/configs/evolution/large_budget.yaml`

```yaml
max_evaluation_jobs: 6
max_proposal_jobs: 8
max_db_workers: 2

evo_config:
  patch_types: ["diff", "full", "cross"]
  patch_type_probs: [0.4, 0.4, 0.2]
  num_generations: 300
  max_patch_resamples: 3
  max_patch_attempts: 3
  llm_models:
    - "gpt-4.1"
    - "gpt-4.1-mini"
    - "gpt-4.1-nano"
    - "us.anthropic.claude-sonnet-4-6-v1:0"
    - "o4-mini"
  llm_dynamic_selection: ucb
  llm_kwargs:
    temperatures: [0.0, 0.5, 1.0]
    max_tokens: 16384
  meta_rec_interval: 10
  meta_llm_models: ["gpt-4.1"]
  meta_llm_kwargs:
    temperatures: [0.0]
  embedding_model: "text-embedding-3-small"
  enable_controlled_oversubscription: false
  results_dir: ${output_dir}
```

### Controlled Oversubscription

When proposal generation is slower than evaluation, Shinka can keep extra
proposal tasks in flight so evaluation workers spend less time idle.

- `max_evaluation_jobs` still caps evaluation concurrency.
- `max_proposal_jobs` becomes the hard ceiling for proposal generation tasks.
- the controller raises the proposal target above evaluation concurrency only
  when observed `sampling_seconds > evaluation_seconds`
- the oversubscription is bounded by `proposal_buffer_max`,
  `proposal_target_ratio_cap`, `proposal_target_hard_cap`, and
  `max_proposal_jobs`

Recommended starting point:

```yaml
max_evaluation_jobs: 5
max_proposal_jobs: 7
max_db_workers: 2

evo_config:
  enable_controlled_oversubscription: true
  proposal_target_mode: adaptive
  proposal_target_min_samples: 5
  proposal_target_ratio_cap: 2.0
  proposal_buffer_max: 2
  proposal_target_hard_cap: 7
  proposal_target_ewma_alpha: 0.3
```

Use `max_proposal_jobs: 1` if you want sync-like proposal behavior with no
proposal backlog.

### Database Presets

All `shinka/configs/database/*.yaml` override `DatabaseConfig` defaults only for listed keys.

#### `shinka/configs/database/island_small.yaml`

```yaml
db_config:
  num_islands: 2
  archive_size: 20
  exploitation_ratio: 0.2
  elite_selection_ratio: 0.3
  num_archive_inspirations: 4
  num_top_k_inspirations: 2
  migration_interval: 10
  migration_rate: 0.1
  island_elitism: true
  db_path: null
```

#### `shinka/configs/database/island_medium.yaml`

```yaml
db_config:
  num_islands: 2
  archive_size: 40
  elite_selection_ratio: 0.3
  num_archive_inspirations: 1
  num_top_k_inspirations: 1
  migration_interval: 10
  migration_rate: 0.0
  island_elitism: true
  enforce_island_separation: true
  parent_selection_strategy: "weighted"
  parent_selection_lambda: 10.0
  db_path: null
```

#### `shinka/configs/database/island_large.yaml`

```yaml
db_config:
  num_islands: 5
  archive_size: 40
  elite_selection_ratio: 0.3
  num_archive_inspirations: 4
  num_top_k_inspirations: 2
  migration_interval: 10
  migration_rate: 0.1
  island_elitism: true
  parent_selection_strategy: "weighted"
  exploitation_alpha: 1.0
  exploitation_ratio: 0.2
  parent_selection_lambda: 10.0
  db_path: null
```

### Cluster Presets

- `shinka/configs/cluster/local.yaml`
  - `job_config: LocalJobConfig`
  - `job_config.eval_program_path: ${distributed_job_config.eval_program_path}`
  - `evo_config.job_type: "local"`
- `shinka/configs/cluster/remote.yaml`
  - `job_config: ${distributed_job_config}`
- `shinka/configs/cluster/gcp.yaml`
  - inherits `remote`
  - overrides `distributed_job_config.partition: "a3,aisci"`

### Task Presets (Current)

Only these task files currently exist:

- `shinka/configs/task/circle_packing.yaml`
- `shinka/configs/task/novelty_generator.yaml`

Both define task-specific `evaluate_function`, `distributed_job_config`, and `evo_config` task prompt/init path.

---

## Current Hydra Composition Defaults

`shinka/configs/config.yaml` defaults chain:

```yaml
defaults:
  - _self_
  - database@_global_: island_medium
  - evolution@_global_: medium_budget
  - task@_global_: circle_packing
  - cluster@_global_: local
  - variant@_global_: default
```

So default `shinka_launch` behavior is a neutral medium shared baseline on the
`circle_packing` task with `variant=default`. Example-heavy stacks remain
available via explicit variants such as `variant=circle_packing_example`.

---

## `shinka_run` Config File Schema

`shinka_run --config-fname <yaml>` accepts:

- Namespaces: `evo`, `db`, `job` (aliases: `evo_config`, `db_config`, `job_config`)
- Runner keys: `max_evaluation_jobs`, `max_proposal_jobs`, `max_db_workers`, `verbose`, `debug`

Precedence for `shinka_run`:

1. defaults from CLI builder
2. config YAML (`--config-fname`)
3. `--set` overrides
4. authoritative flags:
   - `--results_dir` always sets `evo.results_dir`
   - `--num_generations` always sets `evo.num_generations`

---

## Config Directory Structure

```text
shinka/configs/
├── config.yaml
├── cluster/
│   ├── gcp.yaml
│   ├── local.yaml
│   └── remote.yaml
├── database/
│   ├── island_large.yaml
│   ├── island_medium.yaml
│   └── island_small.yaml
├── evolution/
│   ├── large_budget.yaml
│   ├── medium_budget.yaml
│   └── small_budget.yaml
├── task/
│   ├── circle_packing.yaml
│   └── novelty_generator.yaml
└── variant/
    ├── circle_packing_example.yaml
    ├── default.yaml
    └── novelty_generator_example.yaml
```

---

## Quick Valid Overrides

Hydra launch:

```bash
shinka_launch \
  task=novelty_generator \
  database=island_medium \
  evolution=medium_budget \
  cluster=local \
  evo_config.num_generations=50 \
  evo_config.max_api_costs=25.0
```

`shinka_run`:

```bash
shinka_run \
  --task-dir examples/circle_packing \
  --results_dir results/circle_agent \
  --num_generations 40 \
  --max-evaluation-jobs 6 \
  --set evo.llm_models='["gpt-5-mini","gemini-3-flash-preview"]' \
  --set evo.llm_dynamic_selection=ucb \
  --set db.num_islands=2
```
