# Configuration Guide

This document is synced to the current code + config files in this repo.

---

## Default Layers (Source of Truth)

Configuration values are resolved in this order (later wins):

1. Dataclass defaults in code:
   - `shinka/core/config.py` (`EvolutionConfig`)
   - `shinka/database/dbase.py` (`DatabaseConfig`)
   - `shinka/launch/scheduler.py` (`LocalJobConfig`, `SlurmDockerJobConfig`, `SlurmCondaJobConfig`)
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
| `llm_kwargs` | `dict` | `{'temperatures': [0.0, 0.5, 1.0], 'max_tokens': 16384}` | kwargs forwarded to LLM calls. |
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
| `max_api_costs` | `Optional[float]` | `None` | Budget in USD for the paid API calls of a run, counted across resumes. Proposals start only while it covers a reservation for them; it is not a hard limit. See [API Cost Cap](#api-cost-cap). |
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

- Only the file name counts. The database always sits directly in
  `results_dir`, because the prompt database, the meta files, the web UI and
  the analysis helpers look next to it. An absolute path or a path with
  directories is reduced to its file name and the runner logs a warning: set
  `results_dir` to move a run.
- Give the name a `.sqlite` or `.db` suffix. The web UI, `load_programs_to_df`
  and the resume rule below recognise database files by it; the runner warns
  about any other name. `prompts.sqlite` and `prompts.db`
  belong to the prompt database and are refused.
- A run never starts at generation 0 next to programs it did not ask for. If
  the configured file has no programs but `programs.sqlite` or a database of the
  first releases (`evolution_db.sqlite`) does, the run resumes in that one and
  logs a warning. Use a new `results_dir` to start over.
- Programs under any other name are not adopted. If the configured file has no
  programs and another database in `results_dir` does (the run was started with
  a `db_path` the configuration no longer has, or somebody left a copy there),
  the runner stops with an error that lists the files: set `db_path` to the one
  to continue.

`load_programs_to_df(results_dir)` and `python -m shinka.plots.code_path_anim`
look in the same places. Because reading changes nothing, they also load a
database under an unknown name when it is the only one in the directory. A
`ProgramDatabase` you open yourself uses `db_path` as the full file path.

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
| `extra_cmd_args` | `Dict[str, Any]` | `{}` | Extra CLI args forwarded to eval script. |
| `eval_verbose` | `bool` | `True` | Print the per-run evaluation banners to stdout. Passed to the evaluation process as `SHINKA_EVAL_VERBOSE`. |
| `numeric_threads_per_job` | `Optional[int]` | `None` | Thread cap for numeric libraries (OMP, BLAS, MKL, ...) in each evaluation process. `None` leaves the libraries' defaults on Slurm; for local jobs the runner sets it to `cpu_count // max_evaluation_jobs`. |

`LocalJobConfig` adds:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | `Optional[str]` | `None` | Optional timeout (`HH:MM:SS`). |
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
| `time` | `str` | `'01:00:00'` | SLURM time limit. |
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
| `time` | `str` | `'01:00:00'` | SLURM time limit. |
| `cpus` | `int` | `1` | CPU request. |
| `gpus` | `int` | `1` | GPU request. |
| `mem` | `Optional[str]` | `'8G'` | Memory request. |

`conda_env` and `activate_script` are mutually exclusive.

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
Ordinary asynchronous result retrieval retains its existing contract.

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

### API Cost Cap

`max_api_costs` is a budget in USD for every paid API call of a run: proposals,
embeddings, novelty checks, meta summaries and prompt evolution. A resumed run
starts from the charges already stored in its results directory.

What it guarantees:

- No paid request starts once the charges the run knows of have reached the
  cap. A charge is known when the reply arrives, not when the request is sent.
- A proposal starts only if the cap covers the known charges plus one
  reservation for the new proposal and one for every proposal still in flight.
  The reservation is the cost of the most expensive of the last 50 proposals
  this results directory has paid for, failed ones included. An unusually
  expensive proposal therefore holds proposals back only until 50 newer ones
  have finished. What a proposal in flight has already been charged counts on
  top of its reservation.
- When no proposal is in flight and the cap cannot cover even one more
  reservation, the run finishes its evaluations and stops. The log line names
  the amounts, for example `API cost budget cannot cover another proposal:
  $0.3978 charged + $0.1497 reserved (the most expensive of the last 50
  proposals) > $0.45`. A run can therefore end with up to one reservation
  unspent. To use the rest, resume it with a cap that covers the charges so
  far plus that reservation: a resumed run reserves from the last 50 stored
  proposals, and with a smaller raise it stops again before its first request.

What it does not guarantee: that the total stays below the cap. The price of a
request is known only after the reply, and a request that is in flight when
the known charges reach the cap is still billed. The total passes the cap by
at most the cost of those requests. That happens when

- a proposal costs more than its reservation, that is, more than each of the
  50 proposals before it. A proposal joins that history when its program is
  handed to evaluation or when it has failed for good;
- the first `max_proposal_jobs` proposals of a new results directory start.
  They have no cost history yet and so no reservation;
- a meta summary, a prompt evolution step or an LLM-generated initial program
  runs. These calls are not reserved. They start whenever the known charges
  are below the cap, and what they cost is then missing for the proposals in
  flight.

One LLM request costs at most about `max_tokens` output tokens at the model's
output price, plus its prompt; reasoning tokens count as output. With
`max_tokens: 32768`, one `gpt-5.4-mini` reply that spends the whole output
budget costs $0.15. To keep the total below a fixed limit, set the cap lower
than the limit by that amount for every request that can be in flight at
once, or lower `max_tokens` and the reasoning effort. In flight at once are

- one request for each of the `max_proposal_jobs` proposal workers;
- the requests of a meta summary. It sends one request for every program
  evaluated since the last summary, all at the same time: `meta_rec_interval`
  of them (10 by default), or more when several evaluations finish together;
- the request of a prompt evolution step.

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
