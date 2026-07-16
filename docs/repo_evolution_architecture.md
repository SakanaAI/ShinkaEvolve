# Repo-Level ShinkaEvolve Architecture

## Purpose

This document describes the intended architecture for transforming ShinkaEvolve from a system that evolves single program strings into a system that evolves whole repositories with coding agents. The goal is to preserve as much of the original ShinkaEvolve loop as possible so results remain comparable to the upstream system, while changing the artifact being evolved from "one code string" to "one git-backed repository state".

The upstream baseline is SakanaAI/ShinkaEvolve: https://github.com/SakanaAI/ShinkaEvolve. In that system, individuals are programs, proposal generation asks an LLM for edits to a code string, evaluation runs generated program files, and the database stores program text plus metrics, embeddings, lineage, islands, and metadata. Repo-level ShinkaEvolve should keep the evolutionary algorithm, scheduling model, archive, novelty logic, and metrics flow recognizable. The main change is the individual representation and mutation executor.

## Design Principles

1. Keep the evolutionary loop stable.
   Parent selection, islands, archive behavior, evaluator metrics, novelty scoring, and run bookkeeping should remain close to upstream.

2. Change the artifact boundary, not the whole system.
   The evolved artifact becomes a git commit in a worktree, represented compactly by a summary file. The rest of the system should interact with that artifact through stable interfaces.

3. Agents edit files directly.
   Coding agents such as Codex or Cursor should run inside an isolated child worktree and modify files there. ShinkaEvolve should not ask agents to emit diffs and should not apply patches to code strings.

4. Summaries represent individuals.
   Each individual must include a compact `summary.md`-style document that captures the mutation idea, changed files, validation, risks, and lineage. The system should embed and compare this summary, not the full repository.

5. Git is the source of truth for artifacts.
   Each individual should correspond to a commit. The database stores commit identity, summary text, changed files, metrics, and lineage metadata.

6. Evaluation must be isolated from mutation.
   Agents should only be able to edit approved mutable paths. Evaluation scripts, scoring logic, and other immutable files must be protected before evaluation starts.

7. Long evaluations are first-class.
   Evaluation may take minutes, hours, or longer. Job state, worktree paths, commits, result locations, and summaries must be persisted enough to recover after process restarts.

## Upstream Baseline

The legacy single-file system had this flow:

1. Load an initial single-file candidate from the legacy config key.
2. Store a `Program` in the database with code, metrics, lineage, and embedding.
3. Sample parent programs from islands and archive state.
4. Build prompts containing current code, prior attempts, evaluator feedback, and inspirations.
5. Ask an LLM to produce a patch, full replacement, crossover, or fix.
6. Apply the generated code to a generated file such as `gen_N/main.py`.
7. Submit an evaluation job with the legacy program-path flag.
8. Read metrics, correctness, and feedback from evaluator output.
9. Store the accepted child program, update islands, update embeddings, and continue.

Repo-level ShinkaEvolve should keep steps 2, 3, 7, 8, and 9 conceptually intact, but replace code-string prompting and patch application with worktree mutation by an agent.

## Repo Individual Model

A repo-level individual should be modeled as:

```text
RepoIndividual
  id
  parent_id
  generation
  island
  repo_path or artifact_uri
  repo_commit
  repo_parent_commit
  changed_files
  repo_summary
  summary_version
  metrics
  correct
  embedding
  logs
  metadata
```

The database can keep compatibility aliases such as `Program = Repo` and `code = repo_summary`, but the semantic artifact is no longer a code string. `repo_summary` is the searchable, embeddable individual representation. `repo_commit` is the reproducible executable artifact.

## Worktree Lifecycle

### Seed setup

1. Validate `seed_repo_path` is a git repository.
2. Resolve `base_ref` to a seed commit.
3. Create a generation 0 worktree from the seed commit.
4. Generate or read the initial summary file.
5. Evaluate the generation 0 repository.
6. Store a database row with the seed commit, summary, metrics, and embedding.

### Child proposal

1. Sample a parent individual from the database.
2. Create a child branch and worktree from `parent.repo_commit`.
3. Write policy/context files into `.shinka/`:
   - mutable paths
   - immutable paths
   - parent id
   - task objective
   - parent summary
   - evaluator feedback
   - archive or inspiration summaries
4. Start a fresh coding-agent session in the child worktree.
5. Ask the agent to mutate only approved paths and update the summary.
6. Validate the worktree before evaluation:
   - the summary exists and matches schema
   - changed files are inside mutable paths
   - immutable paths are untouched
   - symlinks, submodules, generated files, and binary files are policy-compliant
   - the proposal made a real change
7. Commit the child worktree.
8. Embed the summary and optionally run novelty checks.
9. Submit evaluation with `repo_path` and the committed worktree.
10. Persist the completed result with commit, changed files, summary, metrics, logs, and lineage.

## Summary File

The summary file is the compact representation of an individual. The project should standardize one path. The current code mostly points toward `.shinka/individual.md`, while some active paths expect `summary.md`. One path should be chosen and used everywhere.

Recommended default:

```text
.shinka/individual.md
```

Recommended schema:

```markdown
# Individual Summary

Schema-Version: repo-individual-v1

## Parent

## Core Idea

## Lineage Context

## Changed Files

## Validation Performed

## Performance Hypothesis

## Risks and Followups

## Minimal Snippets
```

The summary should:

1. Explain the mutation in natural language.
2. Name the files changed and why.
3. Include only minimal code snippets.
4. State what validation was performed.
5. State the performance hypothesis.
6. Mention risk areas and follow-up ideas.
7. Stay under `summary_max_chars`.

The summary should be embedded for novelty search. It should also be passed to future agents as compact context.

## Agent Session Model

Each proposal should get one fresh agent session tied to one child worktree. That session should remain resumable until the proposal is accepted, rejected, or abandoned.

Required session metadata:

```text
proposal_id
generation
parent_id
worktree_path
agent_provider
agent_model
agent_session_id
prompt_log_path
response_log_path
started_at
completed_at
status
```

The agent provider must honor the requested working directory. A per-call `headless_work_dir` should override any client default. If the underlying CLI supports resume/session ids, store and reuse them for repair prompts. If it does not, retries should still reuse the same worktree and include the transcript or prior failure details.

Concurrency should be per proposal. A global lock around all headless agent calls defeats the purpose of worktree isolation and should only be used for providers that truly require serialization.

## Context Given To Agents

The repo prompt should be built from structured context rather than the old code-string prompt sampler. It should include:

1. Task objective.
2. Evaluator contract.
3. Parent summary.
4. Parent metrics and evaluator feedback.
5. Archive or top-k inspiration summaries.
6. Mutable paths.
7. Immutable paths.
8. Required summary schema.
9. Validation and commit requirements.

The agent should be instructed to edit files directly in the current worktree and produce no patch text for ShinkaEvolve to apply.

## Mutability And Reward-Hacking Controls

Repo-level evolution increases the attack surface. The system should treat path policy as part of the core architecture, not a prompt-only instruction.

Minimum enforcement:

1. Treat omitted or empty `mutable_paths` as whole-repository mutation. A non-empty list is an explicit user allow-list.
2. Keep evaluation code outside mutable paths.
3. Mark scoring scripts, datasets, expected outputs, `.git`, and `.shinka` as immutable or ignored as appropriate.
4. Check changed files before evaluation and again before commit.
5. Reject changes outside mutable paths.
6. Reject changes to immutable paths even if they also match mutable paths.
7. Reject path traversal, suspicious symlinks, submodule pointer changes, and policy-file edits.
8. Record all changed files in the database.
9. Keep evaluator inputs read-only where practical.
10. Prefer evaluator code outside the evolved repository when benchmarking reward-hacking-sensitive tasks.

Important edge cases:

1. Symlinks inside mutable paths pointing outside the repo.
2. Submodule commit changes.
3. Generated files that affect evaluation.
4. Binary or very large files.
5. Deleted required files.
6. Agents editing dependency lockfiles to avoid real work.
7. Agents modifying benchmark fixtures or cached expected outputs.
8. Ignored files that are consumed by evaluation but not visible in `git diff`.

## Evaluation Contract

Repo-level evaluators should receive a repository path, not a generated program path.

Recommended evaluator CLI:

```bash
python evaluate.py --repo_path /path/to/worktree --results_dir /path/to/results
```

Required outputs:

```text
metrics.json
correct.json
```

Optional outputs:

```text
feedback.txt
logs/
artifacts/
```

The scheduler may keep backward-compatible `program_path` support for upstream-style tasks, but repo mode should submit `repo_path` and run with the worktree as the relevant execution directory.

For ML inference pipeline tasks, the evaluator should separate correctness, latency, throughput, memory, compile time, and stability. It should warm up models, control seeds, report hardware details, and use repeated measurements to reduce noise.

## Long-Running Evaluation

Long evaluations require durable state. The runner should persist enough information before submitting the job:

```text
job_id
individual_id
generation
parent_id
worktree_path
repo_commit
summary_path
results_dir
submitted_at
status
```

The scheduler should support:

1. Jobs that run much longer than one process lifetime.
2. Restart recovery from pending/running job records.
3. Periodic status polling without fixed short timeouts.
4. Result loading after completion with robust retries.
5. Worktree retention until the database row is safely persisted.
6. Clear failure states for timeout, evaluator error, missing results, and policy violation.

Short fixed timeouts are acceptable for small file reads after a job finishes, but not for the evaluation itself.

## Embeddings And Novelty

Embeddings should be computed from `repo_summary`, not from paths, diffs, or full source trees. The full diff should be stored for audit and reproduction, but novelty should compare compact summaries.

LLM-based novelty checks should also use summaries:

1. Parent summary.
2. Candidate summary.
3. Relevant island or archive summaries.
4. Optional changed-file list.
5. Optional evaluator feedback.

This preserves the original novelty mechanism while making it scale to repositories.

## Database And Archive

The database should keep the original evolutionary concepts:

1. Island membership.
2. Parent lineage.
3. Metrics and score.
4. Correctness.
5. Embeddings.
6. Archive membership.
7. Generation and iteration.
8. Sampling metadata.

Repo-specific fields should be preserved during all operations:

1. Insert.
2. Async insert.
3. Island migration.
4. Island copy.
5. Spawned islands.
6. Best/top-k queries.
7. Resume/recovery.
8. Export.
9. Web UI display.

Compatibility aliases are fine, but no path should silently drop commit, summary, or changed-file fields.

## Module Responsibilities

### `shinka/repo`

Owns worktree creation, path policy, summary schema validation, context rendering, and commit snapshots. It should be the only layer that understands git worktree mechanics.

### `shinka/core/async_runner.py`

Owns orchestration:

1. Initialize seed repo.
2. Sample parents.
3. Create child worktrees.
4. Invoke agents.
5. Validate and commit proposals.
6. Submit evaluation.
7. Persist completed results.
8. Manage async concurrency and shutdown.

It should not contain old code-string patch parsing in the repo proposal path.

### `shinka/llm`

Owns provider invocation. For repo mode, it should expose a reliable agent-call interface with explicit working directory and session metadata. It should not override per-proposal working directories.

### `shinka/database`

Owns persistent individual metadata. It can preserve old names for compatibility, but repo mode must store summary text, commit identity, parent commit, changed files, artifact URI, and mutable policy.

### `shinka/evaluation`

Owns job submission and result loading. It should support both `program_path` for legacy tasks and `repo_path` for repo tasks.

### `shinka/core/novelty_judge.py`

Owns novelty scoring from summaries. It should not depend on generated program paths.

### CLI, docs, examples, and skills

These should teach repo mode directly: seed repository, mutable paths, evaluator contract, summary schema, and agent provider setup.

## Fair Comparison With Upstream

To compare fairly:

1. Keep task objective, evaluator, metrics, population size, island count, archive behavior, and selection logic as close as possible.
2. Change only mutation machinery and artifact representation.
3. Store enough metadata to separate agent improvements from evolutionary algorithm changes.
4. Support a legacy single-file mode or preserve an upstream branch for direct comparison.
5. Report both evaluation metrics and system metrics such as agent tokens, wall time, failed proposals, policy violations, and evaluation cost.

## Open Decisions

1. Whether the canonical summary path is `.shinka/individual.md` or repository-root `summary.md`.
2. Whether the codebase should remain dual-mode or become repo-only.
3. Which headless agent providers have real session/resume support.
4. How aggressively to clean old worktrees and branches.
5. Which task-specific repository paths should be immutable or hidden; whole-repository mutation is the default.
6. How prompt evolution maps onto repo-agent context templates.
7. How much evaluator isolation is required for reward-hacking-sensitive benchmarks.
8. How to store large logs, diffs, and artifacts without bloating the main SQLite database.
