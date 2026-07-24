# Repo-Level Transformation Status Report

## Status At A Glance

The repo-only core is implemented and testable on `main`. As verified on 2026-07-21, `python3 -m pytest -q` completed with `777 passed` (plus 15 non-failing warnings). The earlier report that imports, prompts, and repo-agent tests were broken is obsolete.

The project is not yet production-ready for sensitive or long-running evaluation. The current mainline evaluator is trusted-local, and real model-backed evolution has not yet been run end to end.

## Verified Mainline Capabilities

1. The package imports and the async runner, database package, prompt package, and Headless provider load successfully.
2. Repo-only configuration requires a seed git repository and evaluates candidates with `repo_path`.
3. A `Program` database row retains repo commit, parent commit, summary, changed-file, metric, and session metadata.
4. The active proposal path creates a child worktree, invokes Headless in that worktree, validates `.shinka/individual.md`, enforces mutability policy, commits the result, and embeds the summary.
5. The worktree layer covers policy tampering, path traversal, changed symlinks, untracked files, binary/oversized files, deletions, protected paths, and no-change/summary-only proposals.
6. Fake-agent repository evolution and delayed/recovered evaluation paths are covered by the test suite.
7. The repository includes a documented fair-comparison methodology in [Repo-Agent Evaluation](repo_evolution_evaluation.md).

## Boundaries And Known Gaps

### Trusted-local evaluation is not sealed evaluation

`agent_hidden_paths`, immutable paths, Git worktrees, and evaluator path hiding are useful policy controls. They do not protect evaluator repositories, private data, credentials, SQLite state, or result stores from an untrusted agent or candidate with host access. Do not use the current path for hidden tests or reward-hacking-sensitive benchmarks.

### Real agent operation remains unverified

The completed core tests use a fake Headless agent. A real model/provider run needs an explicit small-budget canary, with the provider/model, authentication, image, and expected accounting selected beforehand.

### Session metadata is not durable conversation continuity

Program rows record Headless session names and IDs. Real repair-turn continuity across disposable containers also needs a private, per-proposal agent home that is mounted only for that proposal chain. This depends on the Headless/container integration work.

### Secure runtime work must be extracted before merge

`codex/sandbox-eval` contains a secure mutation/evaluation implementation, agent-image work, tests, documentation rewrites, and unrelated benchmark additions. It was built from an older base and is not safe to merge wholesale. Rebase it onto current `main`, split it into reviewable feature branches, and preserve the current repo-agent evaluation guide during that process.

### Benchmarks are not yet integrated

The paper-task/open-problem catalog is entangled in `codex/sandbox-eval`. The Stockfish NNUE evaluator and harness are in `codex/assess-shinkaevolve-for-nnue`. Both need independent rebase, review, and test runs before they become mainline examples.

## Next Objectives

1. Split and rebase the secure-runtime branch.
2. Land a minimal secure vertical slice: candidate artifact, mutation container, evaluator/candidate process boundary, durable local job record, and cleanup/recovery tests.
3. Publish and pin a universal Headless-agent image by digest; add persistent per-proposal agent homes.
4. Run a low-budget real-agent canary against a public evaluator.
5. Integrate NNUE and the benchmark catalog as independent changes.
6. Run the paired, pre-registered repo-agent evaluation pilot after sealed evaluation is available.

## Non-Blocking Cleanup

The remaining source TODOs are mostly legacy or broad infrastructure debt. The notable repo-mode items are simplifying the summary schema, removing/quarantining legacy `PaperEdit` support, and deciding whether multi-harness sampling belongs in scope. Structured-output gaps in non-Headless providers and visualization `pass` handlers are not blockers for the repo-agent milestone.
