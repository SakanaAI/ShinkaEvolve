# Getting Started

Shinka combines Large Language Models with evolutionary algorithms to drive
scientific discovery. This guide covers installation, first run, and API usage.

---

## What is Shinka?

Shinka enables automated exploration and improvement of scientific code:

- **Evolutionary Search** — Maintains a population of programs that evolve over generations
- **LLM-Powered Mutations** — Uses LLMs as intelligent mutation operators for code improvement
- **Parallel Evaluation** — Supports parallel evaluation locally or on SLURM clusters
- **Knowledge Transfer** — Archives of successful solutions for cross-pollination between islands
- **Scientific Focus** — Optimized for tasks with verifiable correctness and performance metrics

Best suited for optimization problems, algorithm design, and scientific computing
tasks with clear evaluation criteria.

---

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Git
- Either uv (recommended) or conda/pip

### Option 1: uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a modern, fast Python package installer.

#### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### Clone and install

```bash
git clone <shinka-repository-url>
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
uv pip install -e .
```

### Option 2: conda/pip

```bash
conda create -n shinka python=3.11
conda activate shinka
git clone <shinka-repository-url>
cd ShinkaEvolve
pip install -e .
```

### Set up credentials

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here              # Optional
OPENROUTER_API_KEY=sk-or-v1-...                        # Optional
LOCAL_OPENAI_API_KEY=local                             # Optional
CUSTOM_API_KEY=...                                     # Optional
```

### Verify installation

```bash
shinka_launch --help
python -c "from shinka.core import ShinkaEvolveRunner; print('OK')"
```

### Optional: Agent skills

Install bundled Shinka skills for Claude Code or Codex:

```bash
npx skills add SakanaAI/ShinkaEvolve --skill '*' -g -a claude-code -a codex -y
```

See [Usage Guide](agentic_usage.md) for per-skill walkthroughs.

### Advanced uv features

```bash
# Lockfile for reproducible environments
uv pip compile pyproject.toml --output-file requirements.lock
uv pip install -r requirements.lock

# Dev dependencies
uv pip install -e ".[dev]"

# Sync environment to exact spec
uv pip sync pyproject.toml
```

---

## Basic Usage

### CLI Quick Start

```bash
# Default baseline
shinka_launch

# Custom parameters
shinka_launch \
    task=circle_packing \
    database=island_small \
    evolution=small_budget \
    cluster=local \
    evo_config.num_generations=5
```

The shorthand group syntax (`task=...`, `database=...`, `evolution=...`,
`cluster=...`, `variant=...`) maps to presets under `shinka/configs/`.

Custom Hydra presets from a PyPI install:

```bash
mkdir -p ~/my-shinka-configs/variant
$EDITOR ~/my-shinka-configs/variant/my_variant.yaml
shinka_launch --config-dir ~/my-shinka-configs variant=my_variant
```

### Agent-Friendly CLI (`shinka_run`)

Direct task-directory launcher for agents:

```bash
shinka_run --help

# Minimal async run
shinka_run \
    --task-dir examples/circle_packing \
    --results_dir results/circle_agent_run \
    --num_generations 20

# With namespaced overrides
shinka_run \
    --task-dir examples/circle_packing \
    --results_dir results/circle_agent_custom \
    --num_generations 40 \
    --max-evaluation-jobs 6 \
    --set db.num_islands=3 \
    --set job.time=00:10:00
```

| Constraint | Rule |
|------------|------|
| `--task-dir` | Must contain `evaluate.py` and `initial.<ext>` |
| `--set` | Strict namespaces: `evo.<field>`, `db.<field>`, `job.<field>` |
| `--results_dir` / `--num_generations` | Always authoritative |

### Python API

```python
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(
    eval_program_path="examples/circle_packing/evaluate.py",
    activate_script=".venv/bin/activate",
)

db_config = DatabaseConfig(
    archive_size=40,
    num_archive_inspirations=1,
    num_islands=2,
    migration_interval=10,
)

evo_config = EvolutionConfig(
    num_generations=50,
    llm_models=["gpt-5-mini", "gemini-3-flash-preview"],
    init_program_path="examples/circle_packing/initial.py",
    language="python",
    task_sys_msg="You are optimizing circle packing...",
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
    max_evaluation_jobs=1,
    max_proposal_jobs=1,
)
runner.run()
```

Dynamic backend model formats:

```python
evo_config = EvolutionConfig(
    llm_models=[
        "openrouter/qwen/qwen3-coder",
        "local/qwen2.5-coder@http://localhost:11434/v1",
        "local/dummy-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY",
    ],
    embedding_model="local/text-embeddings-inference@http://localhost:8080/v1",
)
```

See the [Configuration Guide](configuration.md) for detailed options.

---

## Circle Packing Example

Recommended first example: optimize arrangement of 26 circles in a unit square
to maximize the sum of radii.

### File structure

```
examples/circle_packing/
  initial.py       Seed solution
  evaluate.py      Evaluation harness
  run_evo.py       Direct Python runner
```

### Run it

```bash
shinka_launch

# Custom settings
shinka_launch \
    task=circle_packing \
    cluster=local \
    evo_config.num_generations=20 \
    db_config.num_islands=4

# Direct Python
python run_evo.py
```

### Initial code structure

The `EVOLVE-BLOCK-START/END` markers define mutable code regions:

```python
# EVOLVE-BLOCK-START
def construct_packing():
    """Construct arrangement of 26 circles in unit square"""
    n = 26
    centers = np.zeros((n, 2))
    # ... placement logic ...
    return centers, radii
# EVOLVE-BLOCK-END
```

### Evaluation script

Uses `run_shinka_eval` to test and score evolved solutions:

```python
from shinka.core import run_shinka_eval

def main(program_path: str, results_dir: str):
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_packing",
        num_runs=1,
        run_workers=1,
        get_experiment_kwargs=get_kwargs_fn,
        validate_fn=validation_function,
        aggregate_metrics_fn=metrics_function,
    )
```

`run_workers` controls repeated runs *inside one evaluation call* — separate
from evolution-level concurrency (`max_evaluation_jobs`).

### Validation function

```python
def validate_packing(run_output):
    """Returns (is_valid: bool, error_msg: str or None)"""
    centers, radii, reported_sum = run_output
    if constraint_violated:
        return False, "Specific error description"
    return True, None
```

### Metrics aggregation

```python
def aggregate_metrics(results, results_dir):
    centers, radii, reported_sum = results[0]
    return {
        "combined_score": float(reported_sum),    # PRIMARY FITNESS (higher = better)
        "public": {                               # Visible in WebUI/logs
            "num_circles": len(centers),
            "centers_str": format_centers(centers)
        },
        "private": {                              # Internal analysis only
            "reported_sum_of_radii": float(reported_sum),
            "computation_time": 0.15
        }
    }
```

### What `run_shinka_eval` returns

| Return value | Type | Description |
|-------------|------|-------------|
| `metrics` | `dict` | `combined_score` (fitness), `public` (WebUI), `private` (internal) |
| `correct` | `bool` | `True` = valid, can reproduce; `False` = discarded |
| `error_msg` | `str` or `None` | Error description if validation failed |

**Public** metrics appear in WebUI and logs. **Private** metrics are for
internal analysis and debugging only.

---

## Other Examples

| Example | Description | Use Case |
|---------|-------------|----------|
| [Circle Packing](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/circle_packing) | 26 circles in unit square; maximize sum of radii | Geometric optimization |
| [2048](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/game_2048) | Evolve a policy to play 2048 | Game-playing / heuristic optimization |
| [Julia Prime Counting](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/julia_prime_counting) | Optimize Julia prime-count queries | Algorithmic optimization |
| [Novelty Generator](https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/novelty_generator) | Diverse outputs scored by LLM-as-a-judge | Open-ended exploration |
| [Tutorial Notebook](https://github.com/SakanaAI/ShinkaEvolve/blob/main/examples/shinka_tutorial.ipynb) | Guided walkthrough of Circle Packing and Novelty Generator | Interactive onboarding |

---

## Advanced Usage

### Resuming experiments

When you specify an existing `results_dir` containing a database, Shinka will
detect the previous run, restore the population and history, and resume from the
last completed generation.

#### CLI (Hydra)

```bash
shinka_launch \
    variant=default \
    evo_config.results_dir=results_20250101_120000 \
    evo_config.num_generations=50
```

#### Python API

```python
evo_config = EvolutionConfig(
    num_generations=50,
    results_dir="results_20250101_120000",
)
runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(archive_size=20, num_islands=2),
    max_proposal_jobs=1,
)
runner.run()
```

| Rule | Detail |
|------|--------|
| `num_generations` | Set to **total** desired, not additional (completed 20, want 30 more = set 50) |
| DB config | Must match the original run |
| Prior state | Best solutions and meta-recommendations are preserved |

### Environment management

| Mode | Config |
|------|--------|
| Current env (default) | `LocalJobConfig(eval_program_path="evaluate.py")` |
| Sourced env | `LocalJobConfig(..., activate_script=".venv/bin/activate")` |
| Conda env | `LocalJobConfig(..., conda_env="my_project_env")` |

`conda_env` and `activate_script` are mutually exclusive.

### Creating custom tasks

1. **Define the problem** — task config in `shinka/configs/task/my_task.yaml`
2. **Initial solution** — `initial.py` with `EVOLVE-BLOCK` markers
3. **Evaluation script** — `evaluate.py` with validation logic
4. **Variant config** — combine settings in `shinka/configs/variant/my_variant.yaml`

See the [Configuration Guide](configuration.md) for parameter explanations.

### Code evolution animation

```bash
python code_path_anim.py --results_dir examples/circle_packing/results_20250101_120000
```

---

## Troubleshooting

### Import errors

```bash
uv pip install -e .          # or: pip install -e .
python -c "import shinka; print(shinka.__file__)"
```

### API key issues

```bash
cat .env
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Evaluation failures

- Check evaluation script function signatures
- Verify `EVOLVE-BLOCK` markers are placed correctly
- Ensure evaluation function returns expected types

### Memory issues

- Reduce `max_evaluation_jobs`
- Increase memory allocation for cluster jobs
- Monitor database size and archive settings

### uv issues

```bash
uv --version
which python           # Should point to .venv/bin/python
rm -rf .venv && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
uv cache clean
```

### Conda environment issues

```bash
conda env list
conda run -n my_env python --version
conda run -n my_env python -c "import shinka; print('OK')"
```

### Debug mode

```bash
shinka_launch variant=my_variant verbose=true
```

---

## Next Steps

1. **Run the examples** — Circle packing is the best starting point
2. **Explore the WebUI** — see the [WebUI Guide](webui.md) for live visualization
3. **Create custom tasks** — adapt the framework to your optimization problems
4. **Scale up** — deploy on clusters for large-scale experiments
