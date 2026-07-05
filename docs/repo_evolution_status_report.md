# Repo-Level Transformation Status Report

## Executive Summary

The repository is in an invasive mid-refactor. The direction is right: worktree management, repo summaries, repo-specific database fields, and scheduler support for `repo_path` are partially present. However, the current code is not runnable as a repo-evolution system. Package imports fail, prompt imports fail, the LLM result type is inconsistent with providers, the active async runner path still uses old code-string proposal machinery, and completed repo proposals would lose critical commit and summary metadata.

The most important architectural gap is that there are two competing proposal paths. `_generate_repo_proposal()` looks closer to the desired design but is not the active path. `_run_patch_async()` is active, but it still uses old prompt sampling, hardcodes a different summary filename, does not reliably run the agent in the worktree, and submits evaluation as a generated program path rather than a repository path.

## Verification Performed

1. Reviewed the upstream SakanaAI/ShinkaEvolve repository: https://github.com/SakanaAI/ShinkaEvolve.
2. Cloned upstream locally to compare structure and behavior.
3. Scanned the current repository files under `shinka/`, `docs/`, `tests/`, `examples/`, and config files.
4. Ran import smoke checks for database, async runner, and prompts.
5. Ran the focused repo-agent test file.

Observed failures:

```text
pytest -q tests/test_repo_agent_evolution.py
ModuleNotFoundError: No module named 'shinka.database.individuals'
```

Additional import checks showed:

```text
import shinka.database.dbase -> fails through shinka.database.__init__
import shinka.database.async_dbase -> fails on shinka.database.individuals
import shinka.core.async_runner -> fails on shinka.database.individuals
import shinka.prompts -> fails because prompts_base.py is empty
```

## Successfully Transformed Areas

### Worktree Layer

`shinka/repo/worktree.py` is a solid start. It introduces:

1. `WorktreeManager`.
2. `RepoWorktree`.
3. `WorktreeSnapshot`.
4. Policy files under `.shinka/`.
5. Child worktree creation from parent commits.
6. Changed-file detection.
7. Mutability enforcement.
8. Commit creation for child individuals.

This is the right foundation for git-backed individuals.

### Summary Schema

`shinka/repo/summary.py` defines a clear `repo-individual-v1` summary shape with required sections. This matches the intended compact representation for embeddings, novelty, and agent context.

### Repo Context

`shinka/repo/context.py` introduces structured repo-agent context with task objective, parent summary, metrics, feedback, archive summaries, mutable paths, immutable paths, and summary filename. This is the right replacement for code-string mutation prompts.

### Config Direction

`shinka/core/config.py` now includes repo-oriented settings such as:

1. `seed_repo_path`.
2. `worktree_root`.
3. `base_ref`.
4. `mutable_paths`.
5. `immutable_paths`.
6. `ignore_paths`.
7. `summary_filename`.
8. `summary_max_chars`.
9. `agent_model`.

The fields are directionally correct, though not consistently used.

### Database Schema Direction

`shinka/database/dbase.py` introduces a `Repo` dataclass and repo-specific columns:

1. `individual_type`.
2. `repo_commit`.
3. `repo_parent_commit`.
4. `repo_diff`.
5. `repo_summary`.
6. `summary_version`.
7. `changed_files`.
8. `artifact_uri`.
9. `mutable_paths`.
10. `immutable_paths`.

The database should use `Program` and `ProgramDatabase` directly; repo-specific fields belong on `Program` rows.

### Scheduler Direction

The evaluation scheduler has started to accept `repo_path` and run evaluators with the repository as context. This is necessary for whole-repo evolution.

### Novelty Direction

The novelty judge has been partially adapted to use summaries, especially `.shinka/individual.md`, rather than raw program strings.

## Partially Transformed Areas

### Async Runner

`shinka/core/async_runner.py` contains many repo concepts, but the active path is not coherent. The runner initializes a seed repo, creates worktrees, and has repo fields in `AsyncRunningJob`, but the main proposal path still behaves like the old patch-based system in several critical places.

### LLM Headless Provider

`shinka/llm/providers/headless.py` has been adapted to use `.shinka` logs and headless CLIs, but it does not yet provide the needed per-proposal session model. The async path also has a runtime bug.

### Database Layer

The main table schema is repo-aware, but async imports, island copying, complexity analysis, and naming are still mixed with program-string assumptions.

### CLI And Docs

Some docs mention repo mode, but README, configuration docs, task layout assumptions, and CLI behavior still conflict with the new architecture.

### Tests

There is a focused repo-agent test file, but it does not currently collect because imports are broken. The broader test suite likely still assumes upstream single-file behavior in many places.

## Blocking Issues

### 1. Database Package Imports Are Broken

`shinka/database/async_dbase.py` imports:

```python
from .individuals import Repo
```

No separate repo individual model should exist. `shinka/database/__init__.py` should export the `Program` and `AsyncProgramDatabase` APIs used by the async runner.

Impact: the system cannot start.

### 2. Prompt Package Imports Are Broken

`shinka/prompts/prompts_base.py` is empty, while `shinka/prompts/__init__.py` imports names from it.

Impact: prompt-related imports fail immediately.

### 3. `QueryResult` API Does Not Match Providers

Current providers instantiate `QueryResult` with fields such as `content` and `new_msg_history`, but `shinka/llm/providers/result.py` no longer defines those constructor parameters.

Impact: LLM calls will fail after import issues are fixed.

### 4. Async Headless Provider References Undefined `usage`

`query_headless_async()` returns a result with `usage=usage`, but `usage` is not defined.

Impact: successful async headless calls raise `NameError`.

### 5. Headless Working Directory Is Overridden

`AsyncLLMClient._attach_headless_work_dir()` always overwrites explicit `headless_work_dir` with the client default. The runner initializes that default as `results_dir`, so agent calls intended for a child worktree can be redirected to the results directory.

Impact: agents do not reliably edit the candidate worktree.

### 6. Active Proposal Path Is The Wrong One

`_generate_evolved_proposal()` calls `_run_patch_async()`. That function still uses `PromptSampler.sample()`, old patch-style prompts, and hardcoded `summary.md` handling. The cleaner `_generate_repo_proposal()` path is not the active path.

Impact: repo-level mutation is not actually wired into the main loop.

### 7. Undefined Variables In Active Runner Path

`_run_patch_async()` references undefined values such as `goal_path` and `repo_prompt` in success/failure handling and retry logic.

Impact: proposal generation can fail even after an agent succeeds.

### 8. Summary Filename Is Inconsistent

Config defaults to `.shinka/individual.md`. `summary.py` and context rendering use the configured path. `_run_patch_async()` requires `summary.md` at the repository root and only later copies it.

Impact: agents receive inconsistent instructions, validations disagree, and summaries may be missed.

### 9. Evaluation Is Submitted As A Program Path

The active path calls evaluation with generated file names such as `gen_N/main.ext`, not with the child worktree `repo_path`.

Impact: repo-level candidates are not evaluated as repositories.

### 10. Accepted Jobs Lose Repo Metadata

The active path builds `AsyncRunningJob` without critical fields:

1. `individual_id`.
2. `repo_path`.
3. `repo_commit`.
4. `repo_parent_commit`.
5. `repo_summary`.
6. `summary_version`.
7. `changed_files`.

`_persist_completed_job()` then creates a new random id and cannot preserve the real artifact identity.

Impact: lineage, reproducibility, archive behavior, and future parent selection are compromised.

### 11. Embeddings Use The Wrong Text

The active path embeds `exec_fname`, which is a generated path string, instead of embedding `summary_text`.

Impact: novelty detection and archive diversity become meaningless.

### 12. Initial Evaluation Failure Can Be Marked Correct

In seed setup, `correct` can become true when the initial evaluation failed.

Impact: failed seed repositories can enter the population as valid individuals.

### 13. CLI Is Broken

`_build_default_evo_values()` now expects `language` and `seed_repo_path`, but `main()` calls it with only `results_dir` and `num_generations`. Some tests still expect old helpers such as `_detect_initial_program()`.

Impact: the documented entry point is unreliable.

### 14. Island Copying Drops Repo Fields

`shinka/database/islands.py` still inserts old column sets when copying or spawning islands.

Impact: repo commit identity, summary, changed files, mutable policy, and artifact metadata can be lost.

### 15. Headless Calls Are Serialized

The async headless path uses a global lock around the CLI call.

Impact: concurrent worktree proposals can become serialized, reducing one of the main benefits of repo-backed agents.

### 16. Agent Sessions Are Not Persistent

The architecture needs one persistent session per proposal, but the current headless provider is essentially one CLI invocation per query with logs.

Impact: follow-up repair prompts cannot reliably resume the same agent context.

### 17. Path Safety Is Too Shallow For Adversarial Repo Mutation

Current mutability enforcement checks changed file paths, but does not fully address symlink targets, submodules, binary files, generated files, dependency lockfiles, ignored files consumed by evaluation, or deletions of required files.

Impact: reward hacking and accidental evaluator tampering remain possible.

### 18. Complexity And Analytics Remain Code-Centric

The database still runs code complexity analysis on `repo.code`, which is now summary markdown.

Impact: analytics become noisy or misleading.

### 19. `shinka/agents` Appears Dead

`shinka/agents/validation.py` is not integrated with the active runner. It may contain useful validation-tier ideas, but it is currently extra surface area.

Impact: maintainability suffers.

## Edge Cases That Need Explicit Handling

1. Agent creates no diff.
2. Agent creates a diff but forgets the summary.
3. Agent creates an invalid or oversized summary.
4. Agent edits only the summary but no mutable code.
5. Agent changes files outside mutable paths.
6. Agent changes immutable evaluator files.
7. Agent changes symlinks to point outside the repo.
8. Agent changes submodule pointers.
9. Agent creates ignored files that evaluation consumes.
10. Agent deletes required files inside mutable paths.
11. Agent changes dependency lockfiles to bypass benchmark intent.
12. Parent commit is missing because a worktree or branch was cleaned.
13. Branch names collide across concurrent proposals or resumed runs.
14. Evaluation runs longer than the main process lifetime.
15. Evaluation finishes but result files are written slowly or partially.
16. Candidate commit succeeds but database insert fails.
17. Database insert succeeds but archive update fails.
18. Worktree cleanup happens before result persistence.
19. Novelty judge receives no valid reference summaries.
20. All candidates fail and fix mode has no valid parent.
21. Agent token usage is unknown, so budget accounting is inaccurate.
22. Multiple headless providers have different session semantics.
23. Web UI tries to display repo summaries as source code.
24. Large diffs or logs bloat SQLite.

## Overall Code Quality Assessment

The current codebase has useful pieces, but the transformation is not yet integrated. The main quality risks are:

1. Dead and duplicate paths.
2. Mixed terminology between `Program` and `Repo`.
3. Import-time failures.
4. Runtime-only bugs in rarely tested async paths.
5. Stale tests and docs.
6. Unclear mode boundary between legacy single-file evolution and repo evolution.
7. Incomplete reward-hacking controls.

The highest-leverage next step is not adding more features. It is restoring an importable baseline and making one minimal fake-agent repo evolution pass end to end.
