# Repo-Only Transformation Plan

## Target Outcome

The target is a runnable repo-only ShinkaEvolve system where each individual is still named a `Program` in the database, but the executable artifact is a git commit in a worktree. Each individual is represented by a compact summary file, mutated by a Headless coding agent, evaluated through a repository-level evaluator, and stored with full lineage and metrics.

Minimum acceptable MVP:

1. The package imports cleanly.
2. A seed repository can be initialized and evaluated.
3. A fake or real headless agent can mutate a child worktree.
4. Mutability policy is enforced before evaluation.
5. The child worktree is committed.
6. Evaluation receives `repo_path`.
7. The database stores commit, parent commit, summary, changed files, Headless session metadata, metrics, correctness, and embedding on `Program` rows.
8. A future generation can sample that child as a parent.
9. A focused end-to-end test passes.

## Phase 0: Make Repo Mode The Only Mode

This repository should not support the legacy single-file mode. `Program` remains the database naming convention for an individual, but every active run requires a seed git repository, worktree mutation, and evaluator support for `--repo_path`.

Tasks:

1. Remove `artifact_mode`/dual-mode branching from the plan and active code.
2. Require `seed_repo_path`.
3. Require Headless models for mutation.
4. Keep database class/table/method naming as `Program`/`programs`.
5. Keep repo-specific artifact fields such as `repo_commit`, `repo_summary`, and `repo_path` where they describe the artifact.

## Phase 1: Restore An Importable Baseline

Fix import and type-level failures before changing behavior.

Tasks:

1. Fix `shinka/database/async_dbase.py` to import `Program` and `ProgramDatabase` from `shinka/database/dbase.py`; do not create a separate repo individual model.
2. Fix `shinka/database/__init__.py` so importing `shinka.database.dbase` does not fail through broken async imports.
3. Restore `shinka/prompts/prompts_base.py` exports or stop importing dead prompt symbols.
4. Align `QueryResult` with all providers. Either restore fields such as `content` and `new_msg_history`, or update every provider and caller.
5. Fix `query_headless_async()` so `usage` is defined.
6. Fix `_sample_kwargs_query_async_with_retry()` so it does not reference undefined `msg_history`.
7. Add a smoke test that imports:
   - `shinka.database`
   - `shinka.database.dbase`
   - `shinka.database.async_dbase`
   - `shinka.core.async_runner`
   - `shinka.llm.llm`
   - `shinka.prompts`

Acceptance criteria:

```bash
python -c "import shinka.core.async_runner"
pytest -q tests/test_repo_agent_evolution.py
```

Both should at least collect and run without import-time failure.

## Phase 2: Define The Stable Repo Contract

Make the data model and config contract explicit.

Tasks:

1. Standardize the summary path as `.shinka/individual.md`.
2. Remove hardcoded `summary.md` expectations from active code.
3. Make `seed_repo_path` required.
4. Define omitted or empty `mutable_paths` as whole-repository mutation except protected, immutable, and hidden paths.
5. Decide whether `agent_model` is real or redundant with `llm_models`.
6. Store `repo_summary` as the main text used by embeddings and compatibility `code`.
7. Add database-level handling on `Program` rows for:
   - `repo_commit`
   - `repo_parent_commit`
   - `repo_diff`
   - `repo_summary`
   - `summary_version`
   - `changed_files`
   - `artifact_uri`
   - `mutable_paths`
   - `immutable_paths`
8. Stop running source-code complexity analysis on markdown summaries, or replace it with summary-specific stats.

Acceptance criteria:

1. One `Program` object can round-trip through sync and async database APIs without losing repo artifact fields.
2. Config validation rejects ambiguous single-file runs.

## Phase 3: Harden Worktree Safety

The current worktree manager is a good base. Extend it for repo-level adversarial safety.

Tasks:

1. Check symlink targets for changed symlinks.
2. Detect submodule pointer changes.
3. Detect binary and oversized files.
4. Reject `.git` and `.shinka` changes unless explicitly allowed.
5. Reject path traversal and absolute-path artifacts.
6. Add policy for deleted files inside mutable paths.
7. Add policy for dependency lockfiles.
8. Include untracked files in enforcement.
9. Add cleanup and retention settings for worktrees and branches.
10. Write policy files before agent invocation and verify they were not modified.

Acceptance criteria:

1. Tests cover allowed change, immutable change, outside-mutable change, symlink escape, no-change proposal, and summary-only proposal.
2. Policy violations fail before evaluation.

## Phase 4: Keep The Active Proposal Path

Make repo proposal generation a single coherent path.

Implementation direction:

1. Keep `_run_patch_async()` as the active mutation path for now.
2. Keep `PromptSampler.sample()` as the repo prompt source for now, including fix-mode attempts.
3. Do not define or use a custom `RepoContext`; Headless already receives the rendered system prompt plus patch message through the provider prompt file.
4. Create the child worktree from `parent.repo_commit`.
6. Write `.shinka` policy and context files.
7. Invoke the headless agent with `headless_work_dir=child_worktree.path`.
8. Ensure explicit working directory wins over client defaults.
9. Validate summary schema.
10. Enforce mutability.
11. Commit the child worktree.
12. Embed `summary_text`.
13. Submit evaluation with `repo_path`.
14. Populate every repo artifact field in `AsyncRunningJob`.

Acceptance criteria:

1. A fake agent that edits one allowed file and writes `.shinka/individual.md` produces one committed child.
2. The resulting evaluation job receives the child worktree path.
3. The completed database row has the same individual id, commit, parent commit, summary, and changed-file list produced during proposal generation.

## Phase 5: Make Agent Sessions Real

Each proposal should own one fresh but persistent session.

Tasks:

1. Use Headless `--session` for per-proposal session names.
2. Store session id/name, provider, model, prompt logs, and response logs with the `Program`.
3. Use provider-specific resume flags through Headless when available.
4. Reuse the same session for repair prompts on invalid summary, policy failure, or missing validation.
5. Remove global serialization except for providers that require it.
6. Track unknown token usage explicitly instead of pretending usage is zero.

Acceptance criteria:

1. A proposal retry can follow up in the same worktree.
2. Session metadata is persisted with proposal metadata.
3. Concurrent proposals can run in separate worktrees for providers that support it.

## Phase 6: Fix Evaluation For Long Jobs

Repo-level inference benchmarks may run for a long time. Treat evaluation as durable work.

Tasks:

1. Persist job state before launching evaluation.
2. Store `job_id`, `worktree_path`, `repo_commit`, `results_dir`, and `status` in job metadata.
3. Support recovery of pending/running jobs on process restart.
4. Avoid short global timeouts for evaluation completion.
5. Retry result-file loading after scheduler completion.
6. Keep worktrees until persistence succeeds.
7. Distinguish evaluator failure, timeout, missing result, invalid metrics, and policy violation.
8. Add optional max wall-clock timeout per task.

Acceptance criteria:

1. A fake long-running evaluator can complete after delayed polling.
2. A simulated restart can recover an in-flight job record.
3. A completed job never loses its commit or summary if database persistence is retried.

## Phase 7: Repair Database, Islands, Archive, And Novelty

Tasks:

1. Update async database APIs to use the same `Program` model as sync APIs.
2. Update island copy and spawn SQL to preserve all repo fields.
3. Update archive queries to display and sample summaries.
4. Ensure `get_best`, `get_top`, and parent sampling return valid `Program` objects.
5. Embed `repo_summary` everywhere.
6. Update LLM novelty judge to compare candidate summaries against summary references.
7. Store diffs outside the summary and avoid huge SQLite rows when possible.
8. Add migration handling for existing databases.

Acceptance criteria:

1. Island migration preserves repo commit identity.
2. Novelty checks operate on summaries.
3. Archive and best-program APIs use Program naming.

## Phase 8: Repair CLI, Configs, Docs, And Examples

Tasks:

1. Fix `shinka/cli/run.py` argument flow.
2. Add a repo-mode task template.
3. Document evaluator contract with `--repo_path`.
4. Document seed repository setup.
5. Document mutable and immutable path policy.
6. Update README and configuration docs that still describe legacy single-file task inputs.
7. Update skills and example tasks.
8. Add a minimal ML inference pipeline example with correctness and latency metrics.

Acceptance criteria:

1. A new user can run a local repo-mode example from docs.
2. CLI defaults match the config schema.
3. Old single-file docs are clearly marked legacy if retained.

## Phase 9: Build The Test Matrix

Add tests in increasing scope.

Unit tests:

1. Config validation for repo mode.
2. Summary schema validation.
3. Worktree creation and commit.
4. Mutability enforcement.
5. Symlink and submodule policy.
6. Database round-trip.
7. Async database imports.
8. QueryResult/provider compatibility.
9. Headless working-directory override.

Integration tests:

1. Seed repo initialization.
2. Fake-agent child proposal.
3. Invalid summary repair or rejection.
4. Immutable file violation.
5. Evaluation with `repo_path`.
6. Completed job persistence.
7. Island copy preservation.
8. Novelty embedding from summary.
9. Long-running evaluator simulation.
10. Resume from pending job.

End-to-end tests:

1. One generation with a fake agent.
2. Two generations with parent selection from a prior child.
3. Parallel proposals in separate worktrees.
4. A reward-hacking attempt that edits evaluator files and is rejected.

## Phase 10: Add ML Inference Pipeline Benchmark Template

This is the motivating use case and should become a first-class example after the core loop works.

Template structure:

```text
examples/inference_pipeline_repo/
  seed_repo/
  evaluate.py
  config.yaml
  README.md
```

Evaluator metrics:

1. Correctness against reference outputs.
2. Latency p50, p90, and p99.
3. Throughput.
4. Peak memory.
5. Compile or graph-capture time.
6. Numerical tolerance failures.
7. Stability across repeated runs.

Recommended mutable paths:

```text
src/
inference/
configs/runtime/
```

Recommended immutable paths:

```text
evaluate.py
tests/
benchmarks/
data/
reference_outputs/
```

Acceptance criteria:

1. The example can run locally with a fake agent.
2. The evaluator rejects correctness regressions.
3. The database stores performance metrics suitable for parent selection.

## Cleanup

After the repo path is green:

1. Remove or quarantine dead `_generate_repo_proposal()` logic.
2. Remove unused `shinka/agents` code if it is not integrated.
3. Rename database-layer variables back to `program` where they drifted to `repo`.
4. Keep compatibility aliases only at stable external boundaries if needed.
5. Delete stale tests or split them into legacy and repo-mode suites.
6. Update docs to match actual behavior.
7. Add a migration note for old run databases.

## Suggested Next Ten Commits

1. Fix imports and `QueryResult` compatibility.
2. Fix async headless provider and working-directory override.
3. Standardize summary path to `.shinka/individual.md`.
4. Add repo-only config validation.
5. Keep `_run_patch_async()` as the active repo proposal path.
6. Populate repo fields in `AsyncRunningJob` and persistence.
7. Submit evaluation with `repo_path`.
8. Fix database island copy and async round-trip for repo fields.
9. Add fake-agent end-to-end repo evolution test.
10. Update CLI and docs for one runnable repo-mode example.
