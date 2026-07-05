# Inference Pipeline Repo Example

This is a repo-mode task. The candidate is a git worktree, not a single source file.

## Setup

Initialize the seed repository once:

```bash
cd examples/inference_pipeline_repo/seed_repo
git init
git add src/pipeline.py
git -c user.name=Test -c user.email=test@example.invalid commit -m "seed"
```

Run a fake-agent smoke test without model credentials:

```bash
cd ../../..
SHINKA_HEADLESS_COMMAND="python3 examples/inference_pipeline_repo/fake_headless.py" \
shinka_run \
  --task-dir examples/inference_pipeline_repo \
  --config-fname shinka.yaml \
  --results_dir results/inference_pipeline_repo \
  --num_generations 2 \
  --no-verbose
```

The evaluator receives `--repo_path`, imports `src/pipeline.py` from that worktree, and writes `metrics.json` plus `correct.json`.
