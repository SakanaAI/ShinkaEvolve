# Evaluating Repo-Agent Shinka Against Original Shinka

## What This Comparison Can Claim

The repo-agent system may generate better proposals per attempt, but that is a
hypothesis—not a result implied by using Codex, Cursor, or a repository. Treat
the comparison as an experiment with a declared target:

| Target | Valid claim |
|---|---|
| Historical reproduction | Whether the paper configuration can be reproduced with its frozen code, models, task, and evaluator. |
| Modern re-benchmark | Which system works better today with a new, explicitly named model stack. |
| Component ablation | Which of the mutation harness, repository artifact, and summary representation explains a difference. |
| Native end-to-end comparison | Which complete system offers a better quality/cost/time trade-off. |

Do not call a run with newer models a reproduction of the paper. The paper used
specific model pools, temperatures, mutation mixes, and task settings; its core
method also embeds code for novelty rejection. If a paper-era endpoint or model
behavior is unavailable, retain the published numbers as historical context and
call the new result a *modern re-benchmark*. The paper and its task-specific
configuration tables are the source for the historical settings: [ShinkaEvolve
paper](https://arxiv.org/abs/2509.19349).

Pin an immutable upstream commit, task artifact, dependency environment, and
evaluator hash. Do not use a moving `upstream/main` as “original Shinka”: it
already has post-paper changes, including Headless support.

## What Remains the Same, and What Does Not

The core loop is recognizably retained: archive/island populations, parent and
inspiration selection, scalar and textual evaluator feedback, novelty
rejection, model selection, and asynchronous proposal/evaluation/persistence
flow. The following differences are important confounders beyond the three
headline changes.

| Area | Original string-program loop | Repo-agent implementation | Why it matters |
|---|---|---|---|
| Mutation | Model emits a diff, full rewrite, or crossover text that Shinka applies. | A Headless coding agent edits an isolated worktree with terminal/tool access. | Measures a different model+harness product unless the same underlying model and tool policy are controlled. |
| Candidate boundary | One program string/file; evaluator receives `program_path`. | Git worktree/commit; evaluator receives `repo_path`. | Repository state, file layout, dependencies, and multi-file edits enlarge the search space. |
| Context | Parent and inspirations are source text. | The agent sees the worktree; prompts give parent/inspiration **summaries**. | The information representation, especially crossover context, changes. |
| Novelty | Embeddings and LLM novelty assessment use mutable code. | Embeddings and novelty assessment use `.shinka/individual.md`. | A semantic summary can change both false rejections and diversity, independently of mutation quality. |
| Validity gate | Parse/apply and immutable-block checks precede evaluation. | A nonempty, schema-valid summary, policy checks, hidden-path checks, mutability checks, and a real diff are required before evaluation. | Proposal failures and the denominator for “sample efficiency” change. |
| Search permissions | `EVOLVE-BLOCK` markers define the editable code. | `mutable_paths` controls files; empty defaults permit the whole repo, with deletions, lockfile changes, and binaries allowed by default. | Extra degrees of freedom can produce apparent gains or reward hacking. |
| Agent visibility | The model receives generated text only. | The agent can inspect files and run cheap tests; evaluator/private paths can be hidden and immutable files made read-only. | Useful capability and safety control, but neither is present in the old interface. |
| Accounting and scheduling | API token/call accounting around text proposals. | Headless sessions, worktree setup/copy/commit, route health/rate limits, two-hour default proposal timeout, and possibly unknown subscription usage. | Cost and latency are not directly comparable without full accounting. |
| Persistence | Program text is the executable artifact. | Commit is executable; summary, diff, session metadata, and embedding are database artifacts. | Reproduction requires the commit *and* run database/export. |

The summary is especially easy to misstate. `.shinka/individual.md` is a
required per-worktree sidecar and its text is persisted in the database, but
`.shinka/` is ignored and the summary is not included in the child Git commit.
Archive the SQLite database (or an equivalent summary export), prompt and agent
logs, and commits together.

## Recommended Experiment Set

Use a small ablation ladder when attribution matters. Each row changes one
boundary from the preceding row.

| Arm | Mutation interface | Candidate artifact and novelty text | Purpose |
|---|---|---|---|
| O-historical | Original text diff/full/crossover loop | Single file; code | Historical reference, only if the paper stack is runnable. |
| O-modern | Original loop with the current, pinned model stack | Single file; code | Separates model progress from framework progress. |
| R-raw | Raw model emits repo patches under the repo path policy | Worktree; canonical diff or code manifest | Isolates repository scope from an agent harness. |
| R-agent | Coding agent edits the same constrained worktree | Worktree; canonical diff or code manifest | Tests the agent harness with the representation held as close as possible. |
| R-full | Current repo-agent system | Worktree; required Markdown summary | Measures the shipped system, including summary-based search and policy gates. |

If only two arms are affordable, run O-modern and R-full and label the result a
**native end-to-end system comparison**. It cannot establish that the agent
alone caused a gain. If the identical base model is not available through both
the raw API and agent harness, label the treatment “model + harness,” not
“agent versus raw LLM.”

## What Must Be Held Constant

Pre-register a single task manifest and use it for every arm.

| Hold constant | Practical rule |
|---|---|
| Starting point | Same seed implementation/commit, task description, dependencies, and allowed libraries. For a single-file baseline, place that exact implementation in one mutable repo file. |
| Authoritative evaluation | Same evaluator commit, input distribution, scoring transform, time/memory limits, and correctness gate. Keep scoring code and hidden/private tests outside the candidate and invisible to every mutator. |
| Search budget | Same cap on raw mutation requests, valid proposals, evaluated candidates, and retries; report all four rather than silently choosing the favorable denominator. |
| Evolution algorithm | Same islands, archive size, parent/inspiration sampler, mutation-type mix, novelty threshold, meta-memory settings, and random seeds. Disable adaptive model selection for a clean mutator ablation, or use the same fixed model pool and update rule. |
| Mutation capability | For an isolation study, match base model/version, temperature/reasoning setting, max output/turn budget, tool policy, editable files, and cheap-test budget. For native comparison, allow native tools but disclose them. |
| Repository freedom | Set `mutable_paths` to the original evolvable block or an equivalent fixed file set; normally forbid evaluator, tests, data, lockfiles, binaries, generated result files, and dependency changes. Do not leave the repo defaults unconstrained. |
| Runtime | Same container/image, hardware, network policy, parallel proposal/evaluation slots, queue policy, and warm-up. Compare both serial and fixed-parallelism results when wall time matters. |
| Tuning | Give each arm the same tuning budget and use validation tasks/seeds distinct from the final benchmark. Freeze configuration before final runs. |

Use the same *semantic* task information in an isolation study. Exact prompt
length need not match: a repo contract and tool instructions require different
serialization. But one arm may not receive extra algorithms, benchmark hints,
failure information, or private-test access. Log input/output tokens, prompt
hashes, retrieved files, commands, tool calls, and agent transcripts. A useful
report has both a matched-context ablation and a native-system comparison.

## Metrics and Denominators

Log the proposal funnel for every run:

```text
raw mutation requests
  -> valid/materialized proposals
  -> novelty-accepted candidates
  -> authoritative evaluations
  -> correct candidates
  -> target-reaching candidates
```

Report best-so-far quality against each of the following x-axes:

| Question | Primary measurement |
|---|---|
| Sample efficiency | Evaluated candidates to reach predeclared quality thresholds; area under the best-so-far quality versus evaluation curve; also raw requests to threshold. |
| Cost efficiency | Cost to threshold and quality versus total cost. Include mutation, retries, embeddings, novelty/meta calls, agent subscription or billed usage, evaluation compute, and infrastructure. Report unknown agent usage explicitly rather than treating it as zero. |
| Time efficiency | Wall-clock time to threshold, proposal latency, evaluation latency, throughput, and makespan under both one worker and the same fixed parallelism. |
| Accuracy/quality | Correctness rate plus the task score on a sealed holdout evaluator. For noisy scores, reevaluate finalists with common independent seeds and report uncertainty. |
| Robustness | Pass rate on hidden cases, regressions, repeated evaluations, and transfer to a second task/seed distribution. |
| Search behavior | Diversity/coverage, duplicate/novelty-rejection rate, parent concentration, diff size, and mutation failure taxonomy. |
| Integrity and usability | Reward-hacking/policy-violation rate, reproducibility from archived artifacts, human setup/tuning effort, and retained storage. |

For each arm, use multiple independent search seeds (twenty is a reasonable
starting target when budget permits), plot every run plus median and bootstrap
confidence intervals, and compare paired seeds when possible. Decide thresholds
and the primary metric before inspecting final outcomes.

## Interpreting Outcomes

Higher accuracy at a higher cost is not, by itself, “better.” It is a point on
a quality–cost–time frontier. It is unambiguously better only when it dominates
the comparator (at least as accurate, no more costly, and no slower, with one
strict improvement), or when it wins under a declared operating constraint,
such as “maximum quality under $X,” “first correct solution within Y minutes,”
or “quality at 100 authoritative evaluations.” Otherwise report the trade-off
and let the application choose the operating point.

Likewise, a longer prompt is fair in a native end-to-end comparison if it is
part of the system being evaluated and contains no privileged benchmark
information. It is not fair evidence that the *agent mutator itself* is better
unless the raw-loop control receives the same additional semantic information
or a matched-context ablation shows that length/context does not explain the
effect.

## Minimum Run Manifest

Persist per run: baseline and repo commit hashes; evaluator and dataset hashes;
container/hardware; model/provider/harness versions; model settings; prompt and
tool-policy hashes; all budgets; random seeds; concurrency; full proposal
funnel; token/billing/compute accounting; worktree commits; summaries; diffs;
agent logs; evaluator outputs; and the final configuration. This makes a claim
about sample, cost, time, or quality efficiency auditable rather than anecdotal.
