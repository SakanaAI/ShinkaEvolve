# Circle Packing Experiment Thread Audit

Date: 2026-07-10  
Audited task: `Run circle packing experiment` (`019f3e52-ab25-7681-b96a-7ae1ab0c9419`)  
Base repository commit: `fa8ff96`  
Headless CLI reviewed: `@roberttlange/headless` 0.4.0

## Executive conclusion

The prior agent correctly recognized the required artifact boundary at first: a seed git repository, a coding agent editing an isolated worktree, and an evaluator outside the evolved repository. It also found several real repo-mode defects that should remain fixed.

It later made a central conceptual error. It interpreted “put the single-file seed into a dedicated repo” as “the coding agent may only change that one file.” It then prohibited helper files, tests, terminal commands, optimizers, searches, and background work. Those restrictions converted the experiment back into constrained single-file mutation and directly opposed the purpose of moving mutation from raw LLM output to autonomous coding agents.

The resulting run should be treated as an integration-debugging artifact, not as a completed evolutionary experiment:

- At least 15 `shinka_run` invocations occurred: 13 distinct results directories and two resumes in the final directory.
- W&B created 13 local run directories, fragmenting one logical experiment across many runs.
- The final database assigned generation IDs through 152 but stored only 24 programs, including two seed-island copies.
- It recorded 130 failed proposal events.
- All 22 non-seed stored programs came from Antigravity Gemini. Cursor, Antigravity Opus, and Codex produced no accepted program in the final database.
- The best score was `2.632697290721954`, from Antigravity Gemini 3.1 Pro at generation 29.
- No experiment process is currently running.
- The thread ended by user interruption while the meta model was repeatedly hitting Gemini free-tier quota errors.

The project direction should be corrected now: when `mutable_paths` is unspecified or empty, the agent should be able to mutate nearly the whole repository. Explicit user-provided mutable or immutable paths should narrow that authority. System-owned paths such as `.git`, Shinka policy metadata, hidden evaluator assets, and paths escaping the worktree should remain protected.

## Evidence reviewed

Repository documents:

- [Repo-Level ShinkaEvolve Architecture](repo_evolution_architecture.md)
- [Repo-Only Transformation Plan](repo_evolution_transformation_plan.md)
- [Repo-Level Transformation Status Report](repo_evolution_status_report.md)
- [Core Concepts](core_concepts.md)
- [Configuration Guide](configuration.md)
- [Agentic Usage Guide](agentic_usage.md)
- [Shinka Run Skill](../skills/shinka-run/SKILL.md)

Headless sources:

- [Headless usage guide](https://github.com/RobertTLange/headless-cli/blob/main/docs/usage.md)
- [Headless CLI source](https://github.com/RobertTLange/headless-cli/blob/main/src/cli.ts)
- [Headless agent adapters](https://github.com/RobertTLange/headless-cli/blob/main/src/agents.ts)
- [Headless package metadata](https://github.com/RobertTLange/headless-cli/blob/main/package.json)

Runtime evidence:

- The local task transcript JSONL was used to reconstruct the full interrupted turn, including sections hidden by context compaction.
- Every uncommitted change in the working tree was reviewed.
- Every circle-packing result database was inspected.
- Current Headless, Antigravity, Cursor, and Codex availability checks were run without invoking a mutation model.
- The full test suite was run: 512 tests passed and one stale legacy CLI test failed.

## Intended product behavior

The repository documents and the clarified project goal imply this model:

1. An evolutionary individual is a git-backed repository state, not a source string.
2. A coding agent runs with full tools in an isolated child worktree.
3. The agent decides how to inspect, edit, test, and validate its candidate.
4. The authoritative evaluator is outside the evolved repository or hidden from the agent.
5. The user defines mutable, immutable, and hidden paths when task-specific restrictions are needed.
6. If the user does not specify a mutable allow-list, the default should be nearly whole-repository mutation.
7. Shinka validates policy, records the diff and summary, commits the candidate, evaluates it, and persists lineage.

The current code is close to supporting the desired default already. `WorktreeManager.enforce_mutability()` only applies an allow-list when `self.mutable_paths` is non-empty. The runner separately rejects an empty list. Removing that rejection and documenting empty as “whole repository except protected/immutable/hidden paths” would align behavior with the project goal.

The current design documents are inconsistent with the clarified goal. The architecture and transformation plan say `mutable_paths` should be required unless a separate whole-repo opt-in is added. The configuration guide also calls it a required allow-list. Those statements should be changed, not treated as product requirements.

## Actual experiment outcome

### Final database

The final database is `results/circle_packing_repo_requested_20260707_174133/programs.sqlite`.

| Model | Stored programs | Correct | Best score |
|---|---:|---:|---:|
| Antigravity Gemini 3.5 Flash High | 14 | 13 | 2.632539806752 |
| Antigravity Gemini 3.1 Pro High | 8 | 8 | 2.632697290722 |
| Seed island copies | 2 | 2 | 0.959764216996 |
| Antigravity Opus 4.6 Thinking | 0 | 0 | n/a |
| Cursor GPT-5.4 | 0 | 0 | n/a |
| Cursor GPT-5.1 Nano | 0 | 0 | n/a |
| Codex | 0 | 0 | n/a |

Failed proposal events by configured route:

| Model route | Failed events |
|---|---:|
| Antigravity Gemini 3.5 Flash High | 35 |
| Antigravity Gemini 3.1 Pro High | 27 |
| Antigravity Opus 4.6 Thinking | 26 |
| Cursor GPT-5.4 | 21 |
| Cursor GPT-5.1 Nano | 21 |

The database reaches generation ID 152 because failed proposals still consumed generation IDs. The highest generation with a stored program is 43. Increasing the resume target from 150 to 153 did not create 150 evaluated programs; it only extended the generation-ID range.

### Provider health observed in this audit

- Headless 0.4.0 sees Antigravity 1.0.16, Codex 0.141.0, and Cursor 2026.07.01.
- `agy models` confirms both requested Gemini High variants and Claude Opus 4.6 Thinking.
- `headless --check` reports a Cursor OAuth credential signal.
- Direct `agent models` currently returns “Authentication required.”

This discrepancy matters. Headless documents `--check` as checking binaries, versions, and local credential signals; it is not a real authenticated model invocation. The prior agent treated `--check` as sufficient model validation and then interpreted Cursor's hanging `create-chat` as a process-management problem. Authentication should have been validated directly before changing Shinka's process model.

## Restart ledger

The restarts were not one repeated problem. They came from four classes: uncaught baseline defects, self-imposed experiment restrictions, provider integration failures, and quota/lifecycle failures.

| # | Results directory or resume | Trigger | Assessment |
|---:|---|---|---|
| 1 | `..._165013` | `LocalJobConfig.numeric_threads_per_job` missing; no DB created | Restart justified. The defect was real, but a config-construction smoke test should have caught it before a 150-generation launch. |
| 2 | `..._165057` | `LocalJobConfig.eval_verbose` missing; seed stored with score zero | Restart justified. Again avoidable with a generation-zero smoke test before the main run. |
| 3 | `..._smoke_165249` | `stdout_log` referenced before assignment in seed setup | Correctly moved to a smoke run. The code fix is valid. |
| 4 | `..._smoke_165334` | Async meta helper lacked the `msg_history` parameter | Correct real bug fix. A focused meta test should have existed already. |
| 5 | `..._smoke_165428` | Meta finalization hung; the user also replaced Codex with Cursor | Stopping was reasonable. Replacing Codex was explicitly user-directed. |
| 6 | `..._165803` | Active prompts contained committed merge-conflict markers | Restart required because agents received a malformed prompt. Entirely avoidable with `git diff --check`, a conflict-marker scan, and one fake-agent end-to-end preflight. |
| 7 | `..._170113` | Agent created `src/best_vars.npy`; proposal was rejected by binary policy | The restart decision was wrong. This was evidence that the coding agent was using the repository as intended. The right response was to review/expose artifact policy, not collapse mutation to one file. |
| 8 | `..._170511` | Headless's internal default timeout fired at 300 seconds; Cursor also stalled | Passing an explicit timeout to Headless was correct. Forbidding terminal/search workflows and stopping after only a few minutes was not. |
| 9 | `..._171237` | All attempts hit a new 180-second limit; an Opus worktree had a valid partial improvement | The short timeout caused the failure. Fabricating a summary to admit partial, timed-out state was the wrong recovery mechanism. |
| 10 | `..._172129` | Cursor child processes remained after timeout; a Gemini result reached 2.6078 | Process ownership was a real issue. Stopping to prevent leaks was justified. The implementation chosen later was unsafe under concurrency. |
| 11 | `..._172959` | Opus final-message extraction failed; other agents timed out | The agent should have changed the Headless output contract or tested raw JSON. It instead made per-agent timeouts more aggressive. |
| 12 | `..._173816` | Opus and Cursor failures remained invisible to UCB | Adding failure accounting was directionally reasonable, but transport/harness health should not be treated as model-quality reward. |
| 13 | `..._174133`, first launch | Productive Gemini run; stopped to make “no response” terminal and cap Cursor at 15 seconds | The early terminal classification can be useful. The 15-second Cursor cap made a successful coding-agent proposal practically impossible. |
| 14 | Same directory, first resume | Resumed with target 153; later stalled on bursty Gemini meta calls and 429s | Reusing the DB was correct. The target arithmetic was wrong. Quota demand should have been computed before launch. |
| 15 | Same directory, second resume | Added 13-second batch pacing; then hit the hard daily Gemini quota; user interrupted | Pacing correctly addressed requests per minute but could never solve requests per day. The run should have stopped with a clear quota blocker. |

The main systemic mistake was using repeated production-scale launches as the integration test. The transformation plan explicitly identifies a fake-agent end-to-end evolution as the highest-leverage prerequisite. A proper sequence would have been: full tests, fake-agent two-generation test, one canary per real provider, a short multi-agent canary, then the frozen 150-generation run.

## Chronological decision audit

### 1. Inspect the repository and use the local run guidance

Verdict: **Keep, but update the guidance.**

Inspecting the example, CLI, and model resolver was correct. The local `shinka-run` skill is stale: its frontmatter and early workflow still describe `evaluate.py + initial.<ext>`, while the current CLI requires `evaluate.py + seed_repo/`. That stale guidance contributed to the initial attempt to use classic single-file evolution.

### 2. Consider the classic `examples/circle_packing/run_evo.py` path

Verdict: **Reverse.**

This contradicted the project’s repo-only direction. The user corrected it immediately. Future task detection should fail fast if a request explicitly requires repo mode.

### 3. Convert the existing single-file seed into a dedicated git repository

Verdict: **Keep.**

This was exactly the right interpretation of the seed artifact. Starting with one source file does not imply that future mutations must remain one-file-only.

### 4. Put `evaluate.py` outside `seed_repo/` and pass `--repo_path`

Verdict: **Keep, with evaluator hardening.**

This correctly isolates scoring from mutation. The new evaluator is less robust than the existing pipeline-test evaluator: malformed tuple shapes or validation-time exceptions can escape without writing result files. It should coerce values defensively, catch the whole candidate/validation path, and have evaluator tests.

### 5. Map the requested evolutionary and database configuration

Verdict: **Mostly keep, document ignored fields.**

The archive, island, selection, patch-probability, concurrency, W&B, and score settings were mapped closely. Important caveats were not surfaced:

- Headless mutation calls ignore API-style temperatures and max-token settings in the current provider path.
- Cursor GPT-5.1 Nano does not receive an xhigh variant in Headless 0.4.0; the adapter leaves that model name unchanged.
- `max_novelty_attempts=None` was replaced with 3 because the implementation requires an integer.
- Repo-agent `diff/full/cross` currently influences prompt selection, not literal code-string patch application.

These are acceptable compatibility compromises only if recorded in the run manifest.

### 6. Force W&B offline because `wandb status` did not see a key

Verdict: **Reverse; the later correction was right.**

The project root `.env` contained the key. The agent checked the wrong environment surface and changed requested behavior prematurely. Loading `.env` through the same startup path as the runner was the correct follow-up.

### 7. Add `numeric_threads_per_job` and `eval_verbose` to `LocalJobConfig`

Verdict: **Keep and document.**

Both fields were already consumed elsewhere and their absence caused real startup failures. They need explicit unit/config round-trip coverage and configuration documentation.

### 8. JSON-encode non-scalar W&B table cells

Verdict: **Keep with a regression test.**

This is a reasonable way to keep W&B table column types stable when metadata shapes vary. The existing fake-W&B test does not assert nested row stability, so the exact failure should become a test.

### 9. Move initial metric extraction outside the embedding/verbose branch

Verdict: **Keep.**

This fixes a genuine control-flow bug. Metrics and logs must be extracted regardless of embedding or verbosity.

### 10. Make W&B helpers silently no-op on partially constructed runners

Verdict: **Modify.**

Production `__init__` always creates a disabled or enabled W&B logger. The change exists mainly for tests that instantiate a runner with `object.__new__`. Tests should construct a valid disabled logger instead of weakening production invariants. A narrow cleanup guard is acceptable, but silently accepting missing `db` and `prompt_db` can hide initialization defects.

### 11. Add the missing async `msg_history` parameter

Verdict: **Keep.**

This directly aligns the helper signature with its caller and fixes a real runtime failure.

### 12. Replace Codex routes with Cursor routes

Verdict: **Keep as a user-directed change.**

The user explicitly requested this because Codex usage was nearly exhausted. It should not be described as an autonomous design choice. It also means the sustained experiment cannot be evidence about Codex mutation behavior.

### 13. Remove merge-conflict markers and keep the direct-repository prompt

Verdict: **Keep.**

Choosing the direct-edit side of the conflict matches the repo-agent architecture. Launching before detecting committed conflict markers was the error.

### 14. Allow the whole `src/` directory initially

Verdict: **Too narrow for the product default, but better than the later restriction.**

For this minimal seed, `src/` was a plausible explicit task policy. For a project-level default, the agent should receive the whole repository except protected, immutable, or hidden paths.

### 15. Treat `best_vars.npy`, helper scripts, and local optimizers as invalid agent behavior

Verdict: **Reverse.**

Creating tools, tests, generated data, or an optimizer is normal coding-agent behavior and especially natural for circle packing. The candidate should be judged on its final committed repository and evaluator result. Temporary artifacts can be cleaned by the agent; required artifacts can be committed if policy permits. Binary, large-file, deletion, and lockfile policy should be user-configurable rather than silently converted into a one-file task.

### 16. Narrow `mutable_paths` to `src/packing.py`

Verdict: **Reverse.**

This is the clearest off-path decision. The user asked for repo mode, not a single-file allow-list. The phrase “throw the single file code into a dedicated repo” described seed conversion. The agent should have asked before making a material change to mutation authority.

### 17. Add broad ignore patterns for tests, optimizers, scratch files, NumPy arrays, and logs

Verdict: **Reverse most of them.**

Ignore patterns are dangerous when evaluation can consume ignored files: the files evade policy/diff capture and may disappear when agent-view changes are copied. Keep only deterministic system noise such as bytecode caches where appropriate. Do not ignore categories of legitimate repo artifacts merely to make import validation pass.

### 18. Tell agents not to run terminal commands, tests, optimizers, or searches

Verdict: **Reverse.**

This directly contradicts Headless `--allow yolo`, the coding-agent architecture, and `core_concepts.md`, which says the coding agent decides which cheap validation or smoke checks to run. It also caused the experiment to measure constrained response latency rather than agentic repository mutation.

### 19. Turn debug logging off because output was noisy

Verdict: **Reverse for this experiment.**

The user explicitly asked to debug the full run and the purpose was to surface integration errors. Debug output should have remained in a file while console verbosity was filtered. Disabling debug reduced the evidence available for the exact experiment meant to exercise failures.

### 20. Run independent Headless sanity checks after stalls

Verdict: **Good instinct, incomplete execution.**

Provider canaries were the right tool, but they should have happened before the full run and should have tested:

- authenticated model listing or a real no-op call;
- working-directory mutation;
- raw output and exit status;
- named session creation and resume;
- cleanup after timeout.

The agent inferred too much from an empty worktree and a visible `create-chat` process. It did not check `agent models`, which exposes the current Cursor authentication failure.

### 21. Pass Shinka's timeout explicitly as Headless `--timeout`

Verdict: **Keep, with a grace-period redesign.**

Headless has its own timeout configuration. Passing the desired value explicitly prevents an unexpected internal default from ending the agent early. The outer Shinka timeout should be longer than the inner Headless timeout by a cleanup grace period, not exactly equal to it.

### 22. Reduce the global Headless timeout from 1200 seconds to 180 seconds

Verdict: **Reverse.**

Three minutes was inconsistent with high-reasoning coding agents and with a task where local numerical search is a legitimate strategy. The timeout created the partial-state problem that subsequent patches tried to solve. Startup/auth health timeouts and proposal wall-clock budgets should be separate.

### 23. Salvage timed-out worktrees by fabricating `.shinka/individual.md`

Verdict: **Reverse as default behavior; optional quarantine only.**

The summary is intended to be the agent-authored compact representation used for lineage, novelty, and future context. The fallback contains a generic performance hypothesis, overwrites any partial summary, records zero cost, and can admit an inconsistent mid-edit repository. A timeout candidate may be preserved for forensic inspection, but it should not enter the normal archive unless a resumed agent completes it or an explicit salvage policy revalidates and summarizes it.

### 24. Start Headless in a new process group and terminate the group on outer timeout

Verdict: **Keep the ownership goal; modify the implementation.**

Headless 0.4.0's local timeout calls `SIGTERM`/`SIGKILL` on the immediate agent child. A wrapper-level process group is a reasonable defense. The current implementation races Headless's identical timeout and only cleans the group when the outer wait raises. If Headless exits with code 124 first, the cleanup path may not run. Prefer an upstream Headless process-tree fix or use an inner timeout plus outer grace and cleanup on timeout exit status.

### 25. Find Cursor processes globally with `ps` and kill every newly observed matching PID

Verdict: **Revert immediately.**

This is unsafe under the requested parallelism. If two Cursor proposals overlap, each timeout snapshot can classify the other proposal's child as “new” and kill an unrelated active agent. It can also affect a user-owned Cursor process started during the window. Process ownership must be explicit through PIDs, process groups, native session IDs, or a Headless-managed run—not inferred from a global command-line substring scan.

### 26. Treat Cursor `create-chat` as abnormal daemonization

Verdict: **Reverse the diagnosis.**

Headless documents and implements `create-chat` as the normal way to mint a Cursor session ID before resume. A long-running or stuck `create-chat` can still be a bug, but the first hypotheses should have been authentication and native session setup. The current direct CLI authentication failure supports that interpretation.

### 27. Investigate Opus “could not extract final message” but continue in text mode

Verdict: **Incomplete; fix the integration before scoring the model.**

Headless explicitly says to rerun with `--json` when final-message extraction fails. Shinka ignores the final assistant message and treats the worktree as the result, so it should have a machine/raw-trace mode that does not make final-text extraction a success condition. Headless 0.4.0 forbids `--json` with `--usage`, so the durable solution is a structured Headless envelope or upstream support for usage in machine mode. Until then, mark usage unknown rather than declaring Opus unhealthy.

### 28. Add per-agent timeout environment variables

Verdict: **Keep the capability in configuration, not as an undocumented environment-only patch.**

Different agent CLIs can need different budgets. The 360-second Antigravity and 45/15-second Cursor values were not reasonable measurements of equivalent coding work. Timeouts must be persisted in the run config and W&B manifest.

### 29. Penalize terminal proposal failures in UCB

Verdict: **Modify into two layers.**

The quality bandit should learn from valid evaluated candidates. Authentication errors, wrapper parsing errors, process timeouts, and policy-engine defects are route-health failures, not model-quality rewards. Add a provider-health circuit breaker outside the quality bandit. It may temporarily disable an unhealthy route while preserving a separate failure count. Invalid evaluated candidates can reasonably receive the bandit's worst reward.

### 30. Stop patch-attempt retries immediately when the provider already returned no response after its own retries

Verdict: **Keep the concept with explicit failure taxonomy.**

Multiplying provider retries by patch retries by novelty retries caused large delays. A terminal transport/auth failure should stop that proposal. A valid agent response with an invalid summary or policy violation should still use the same native session for repair. These cases should not share one `terminal_model_failure` flag.

### 31. Resume the productive DB rather than start another fresh run

Verdict: **Keep.**

Once the run had valid persisted candidates, resume was better than discarding them. However, resuming after changing core runner semantics mixes different algorithms in one experiment. That DB should be labeled diagnostic, not a controlled run.

### 32. Raise the target to 153 to compensate for failed early generations

Verdict: **Reverse.**

The runner targets generation IDs/proposals, not successful stored evaluations. The final state proves the arithmetic wrong: IDs reached 152, but only 24 programs were stored. Decide and document whether “150 generations” means 150 proposal IDs, 150 completed proposals, or 150 evaluated candidates. If evaluated candidates are required, the scheduler must target that count directly.

### 33. Remove incomplete worktrees and delete partial generation result directories before resume

Verdict: **Modify.**

Removing registered worktrees through git was good hygiene. Deleting failure directories and partial evidence was contrary to the debugging purpose. Persist an abandoned/failed snapshot, logs, status, and diff first; then detach and clean the worktree safely.

### 34. Add a global async batch concurrency/start-interval throttle through environment variables

Verdict: **Replace.**

Spacing requests by 13 seconds correctly addressed a 5 RPM limit. The implementation applies to every `AsyncLLMClient` batch and is not keyed by provider, model, or request class. It can unintentionally throttle mutation and novelty calls and has no daily-budget accounting. Add provider/model-specific rate-limit configuration and a shared limiter. Prefer reducing the meta summarizer to fewer calls rather than issuing ten independent requests.

### 35. Continue retrying after the hard Gemini daily quota was identified

Verdict: **Stop instead.**

Request pacing cannot fix a daily quota. The run's required meta schedule was infeasible on a 20-request daily allowance. The agent should have computed the worst-case call demand before launch and asked for a paid quota, a one-call meta implementation, or an explicitly approved config change.

### 36. Patch the framework while the experiment was running and resume the same scientific run

Verdict: **Do not do this for the next controlled experiment.**

It was useful for integration debugging, but every live patch changed the system under test. Scores and failure rates across directories—and even within the final resumed DB—do not share one implementation. Freeze a commit, config, Headless version, environment manifest, and evaluator before the production run.

### 37. Configure five proposal/evaluation workers but accept a runtime cap of four proposal workers

Verdict: **Reasonable guardrail, but record the effective configuration.**

The requested maximum was five, not a requirement to oversubscribe an eight-core host unsafely. Letting resource validation cap proposal concurrency was reasonable. The run manifest and W&B config should record both requested and effective worker counts so throughput comparisons are not made against the wrong denominator.

### 38. Use `--no-verbose` for meta smoke runs

Verdict: **Modify.**

Reducing console noise for a smoke run was reasonable, but it made the known-unstable meta finalization path opaque. For an error-surfacing experiment, keep detailed file logs and make console verbosity a separate setting.

### 39. Manually evaluate agent worktrees while the Headless process was still editing them

Verdict: **Use snapshots instead.**

The manual checks correctly established that partial proposals could improve the score. Evaluating a live worktree can race ongoing edits and create cache files that alter status. Copy or commit a diagnostic snapshot first, disable bytecode writes, and evaluate the snapshot without admitting it to the archive.

### 40. Give every proposal a stable named Headless session and reuse it for repair calls

Verdict: **Keep.**

This matches Headless's documented session model and the repo architecture. The session should remain associated with one proposal/worktree. Cursor's normal session-minting step must be allowed to complete after authentication is verified.

### 41. Reuse a fixed W&B display name across fresh runs and create new W&B IDs on resume

Verdict: **Modify.**

The local artifacts contain 13 W&B run directories, including three W&B IDs for the final results database. A resume should reuse the logical W&B run ID. A framework/config change should create a new versioned experiment name with the prior run linked as diagnostic lineage.

## File-by-file change audit

Exact uncommitted shared-file inventory:

- `shinka/core/async_runner.py`: fixed seed metric extraction; added timeout-summary synthesis, synthetic `QueryResult`, terminal no-response handling, failed-proposal UCB updates, and W&B guards.
- `shinka/launch/scheduler.py`: added `numeric_threads_per_job` and `eval_verbose` to `LocalJobConfig`.
- `shinka/llm/llm.py`: added the missing async `msg_history` parameter and global environment-controlled batch throttling.
- `shinka/llm/providers/headless.py`: passed `--timeout`; added per-agent timeout variables, process groups, group termination, and Cursor global PID cleanup.
- `shinka/prompts/prompts_diff.py`: resolved committed merge-conflict markers in favor of direct repository edits.
- `shinka/wandb_logging.py`: converted non-scalar table cells to JSON strings.
- `tests/test_headless_provider.py`: added only an assertion that `--timeout 10` reaches the fake Headless CLI.

The experiment also added `examples/circle_packing_repo_experiment/` with an external evaluator, seed repository, README, and run YAML.

| File/change | Verdict | Required action |
|---|---|---|
| `examples/circle_packing_repo_experiment/seed_repo/` | Keep | Preserve the dedicated git seed. Do not equate the seed's initial size with future mutation scope. |
| `examples/circle_packing_repo_experiment/evaluate.py` | Modify | Keep external repo evaluation; use the more defensive pipeline-test coercion/error handling and add malformed-output tests. |
| `examples/circle_packing_repo_experiment/shinka_requested.yaml` model/database/W&B settings | Mostly keep | Preserve user-directed model swap and requested hyperparameters; record unsupported/ignored Headless kwargs. |
| YAML `mutable_paths: [src/packing.py]` | Reverse | Omit/empty should mean nearly whole repo, or explicitly use a broad policy only if the user requests it. |
| YAML artifact ignore list | Reverse most | Keep only deterministic noise exclusions; do not hide legitimate agent-created repo artifacts from validation and persistence. |
| YAML terminal/helper/optimizer bans | Reverse | Let the coding agent use its tools and repository. Require cleanup and final validity, not a prescribed workflow. |
| YAML `debug: false` | Reverse for error-surfacing runs | Keep debug logs in files; tune console handlers separately. |
| `LocalJobConfig.numeric_threads_per_job` | Keep | Add docs/config round-trip coverage. |
| `LocalJobConfig.eval_verbose` | Keep | Add docs and a direct scheduler-config test. |
| Initial repo metric-extraction indentation fix | Keep | Add regression coverage with embeddings disabled and verbosity variants. |
| W&B missing-logger guards | Modify | Prefer valid test construction; do not hide production initialization defects. |
| W&B non-scalar table serialization | Keep | Add a nested heterogeneous metadata regression test. |
| Async LLM `msg_history` signature fix | Keep | Add a direct test that history reaches the provider. |
| Global async batch throttle | Replace | Implement provider/model/request-class rate limiting and daily demand checks. |
| Prompt conflict resolution | Keep | Add conflict-marker/preflight checks to CI and the run skill. |
| Headless explicit `--timeout` | Keep | Persist it in config and use an outer cleanup grace period. |
| Headless process-group lifecycle | Modify | Handle exit code 124/nonzero cleanup, cancellation, concurrency, and cross-platform behavior; prefer fixing Headless upstream. |
| Cursor global PID discovery/termination | Revert | Never kill by global command-line matching. Track owned native sessions/processes. |
| Timeout fallback summary and synthetic `QueryResult` | Revert by default | Preserve partial candidates as quarantined diagnostics; resume the agent to completion instead. |
| Failure update into UCB | Modify | Separate route health from evaluated model quality. |
| Immediate terminal no-response shortcut | Modify/keep | Use typed auth/transport/timeout/extraction failures and provider-level circuit breaking. |
| Test change that only asserts `--timeout` | Insufficient | Add process-tree, concurrent Cursor, raw-output, session-resume, timeout-code, and unrelated-process safety tests. |

The visible patch set adds 432 lines and removes 121 lines across shared framework files, while adding only two test lines. The full suite currently reports 512 passing tests and one failing stale single-file CLI test (`_detect_initial_program`). Most new timeout, salvage, rate-limit, Cursor cleanup, and bandit-failure behavior has no committed regression coverage.

## Documentation changes required

1. Change `mutable_paths` semantics everywhere:
   - omitted or empty: all repository paths are mutable except system-protected, immutable, and hidden paths;
   - non-empty: explicit allow-list;
   - `immutable_paths` always overrides mutable scope.
2. Update the architecture and transformation plan, which currently require a mutable allow-list.
3. Update the configuration guide, which calls `mutable_paths` required.
4. Update the `shinka-run` skill from `initial.<ext>` to `seed_repo/` and add a mutation canary preflight.
5. Document which API-style LLM kwargs Headless routes ignore.
6. Document Headless session behavior, including Cursor `create-chat` and Antigravity transcript-backed resume.
7. Define “generation” versus “evaluated candidate” for run targets and resume behavior.
8. Add a run-manifest section covering git commit, dirty diff hash, config hash, evaluator hash, Headless version, native CLI versions, provider health, timeouts, quotas, and W&B run identity.

## Recommended corrected architecture

### Mutation policy

Use these defaults:

```text
mutable_paths omitted/empty  => whole repository
immutable_paths              => explicit user deny-list
agent_hidden_paths           => evaluator/private assets removed from agent view
always protected             => .git, Shinka-owned policy state, path escapes
```

Deletions, lockfile changes, binaries, and large files should be explicit policy knobs. For the intended high-control default, ordinary deletions and dependency changes should be allowed. Symlink escapes, submodule boundary violations, and evaluator access should remain protected regardless of user omission.

### Provider execution

1. Preflight every configured Headless route with a real throwaway worktree mutation.
2. Validate native authentication directly; do not rely only on credential signals from `--check`.
3. Give each proposal one named Headless session and reuse it for repairs.
4. Use a machine-output contract that does not require a final prose message when the worktree is the artifact.
5. Track process/session ownership explicitly.
6. Use typed failures: auth, unavailable model, timeout, extraction, process leak, no diff, invalid summary, policy violation, evaluator failure.
7. Put auth/transport failures behind a route-health circuit breaker and keep them out of quality reward.

### Run lifecycle

1. Do not launch the full run from a dirty, changing framework worktree.
2. Pass the full suite or record approved known failures.
3. Pass a two-generation fake-agent end-to-end test.
4. Pass one real canary per configured route, including session resume and workdir mutation.
5. Run a short multi-agent/W&B canary.
6. Freeze the code/config/evaluator/provider versions.
7. Run the full experiment without framework edits.
8. If a framework defect appears, stop and label that run diagnostic. Fix it, create a new versioned run, and do not combine its statistics with the old run.

### Quotas and logging

1. Estimate total mutation, novelty, meta, embedding, and retry calls before launch.
2. Reject a config whose minimum required demand exceeds a known daily quota.
3. Use per-provider/model rate limiters rather than global environment throttles.
4. Use one W&B logical run per experiment, with proper resume identity if the process restarts.
5. If a code/config change forces a new experiment, create a new W&B run with a new version tag rather than reusing a generic name.

## Prioritized correction plan

### P0: Restore architectural intent

1. Revert the experiment's single-file mutable allow-list and workflow bans.
2. Make empty/omitted `mutable_paths` mean nearly whole-repository mutation.
3. Update the architecture, transformation plan, configuration guide, README, examples, and skills to match.
4. Expose deletion, lockfile, binary, and size policies through `EvolutionConfig` instead of hard-coded restrictive defaults.

### P0: Remove unsafe integration patches

1. Remove Cursor global PID scanning and killing.
2. Disable timeout-summary fabrication by default.
3. Separate provider health from UCB reward.
4. Replace the global async batch throttle with provider-aware rate limiting.

### P0: Fix Headless integration at the contract boundary

1. Reproduce Cursor with direct authentication and model checks.
2. Reproduce Opus with Headless raw JSON and a named session.
3. Decide whether Shinka should use raw JSON with unknown normalized usage or contribute a structured machine-result mode upstream.
4. Fix process-tree cleanup upstream or implement owned process/session cleanup with an outer grace period.
5. Add integration tests for concurrent proposals and unrelated-process safety.

### P1: Harden the experiment harness

1. Replace the bespoke evaluator validation path with robust coercion and total error capture.
2. Define the 150-run completion criterion explicitly.
3. Add quota feasibility checks.
4. Add one logical W&B resume identity and a complete run manifest.
5. Preserve failed proposal artifacts before cleaning worktrees.

### P1: Freeze and rerun

1. Run full tests.
2. Run fake-agent end-to-end.
3. Run one canary for Antigravity Gemini, Antigravity Opus, Cursor, and Codex if all are in the new scope.
4. Run a 5-10 generation multi-agent canary.
5. Run the full 150-target experiment from a frozen commit.

## Acceptance criteria for the next full experiment

- The agent can create, modify, rename, and delete normal repository files when the user did not specify a narrow policy.
- The evaluator and private assets are inaccessible to the agent.
- Every configured Headless route passes a real authenticated workdir/session canary.
- No provider depends on final prose extraction when repository state is the result.
- No cleanup code can kill an unrelated concurrent agent or user process.
- Timeouts are long enough for coding work, recorded in config, and separated from auth/startup health checks.
- The meaning of the 150 target is explicit and verified by a test.
- The run's minimum meta/novelty demand fits available quotas.
- W&B contains a single coherent experiment lineage with code/config/provider hashes.
- Debug logs are retained without flooding the interactive console.
- No framework code changes occur during the controlled run.

## Bottom line

The thread was valuable as a stress test because it exposed real defects in repo initialization, async meta calls, prompt integrity, W&B table logging, Headless timeouts, process cleanup, provider output handling, failure accounting, resume behavior, and quota planning.

It did not validate the intended product. The most important correction is to stop treating coding-agent autonomy as an obstacle to throughput. Agents should be allowed to use the repository and their tools; Shinka should enforce only the user-defined artifact boundary and system safety boundary, then evaluate the final repository state.
