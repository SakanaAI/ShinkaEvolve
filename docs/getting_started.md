# Getting Started with Shinka 🧬

Shinka is a framework that combines Large Language Models (LLMs) with evolutionary algorithms to drive scientific discovery. This guide will help you get started with installing, configuring, and running your first evolutionary experiments.

![](../docs/conceptual.png)

## Table of Contents

1. [What is Shinka?](#what-is-shinka)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Examples](#examples)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

## What is Shinka?

Shinka enables automated exploration and improvement of scientific code by:

- **Evolutionary Search**: Maintains a population of programs that evolve over generations
- **LLM-Powered Mutations**: Uses LLMs as intelligent mutation operators to suggest code improvements
- **Parallel Evaluation**: Supports parallel evaluation locally or on Slurm clusters
- **Knowledge Transfer**: Maintains archives of successful solutions for cross-pollination between evolutionary islands
- **Scientific Focus**: Optimized for tasks with verifiable correctness and performance metrics

The framework is particularly well-suited for optimization problems, algorithm design, and scientific computing tasks where you can define clear evaluation criteria.

## Installation

### Prerequisites

- Python 3.10+ (Python 3.11 recommended)
- Git
- Either uv (recommended) or conda/pip for environment management

### Option 1: Using uv (Recommended - Faster) ⚡

[uv](https://docs.astral.sh/uv/) is a modern, fast Python package installer and environment manager that's significantly faster than pip.

#### Step 1: Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### Step 2: Clone and Install Shinka

```bash
git clone <shinka-repository-url>
cd ShinkaEvolve

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install Shinka in development mode
uv pip install -e .
```

### Option 2: Using conda/pip (Traditional)

#### Step 1: Create Environment

```bash
conda create -n shinka python=3.11
conda activate shinka
```

#### Step 2: Clone and Install

```bash
git clone <shinka-repository-url>
cd ShinkaEvolve
pip install -e .
```

### Step 3: Set Up Credentials

Create a `.env` file in the project root with your API keys:

```bash
# .env file
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here  # Optional
```

### Step 4: Verify Installation

```bash
# Test the CLI launcher
shinka_launch --help

# Test Python imports
python -c "from shinka.core import EvolutionRunner; print('Installation successful!')"
```

### Advanced uv Features (Optional)

If you're using uv, you can take advantage of additional features:

#### Create a lockfile for reproducible environments:
```bash
# Generate uv.lock file
uv pip compile pyproject.toml --output-file requirements.lock

# Install from lockfile
uv pip install -r requirements.lock
```

#### Install development dependencies:
```bash
# Install with dev dependencies (includes pytest, black, etc.)
uv pip install -e ".[dev]"
```

#### Sync environment to exact specifications:
```bash
# Sync environment to match pyproject.toml exactly
uv pip sync pyproject.toml
```

## Basic Usage

### Quick Start with CLI

The easiest way to get started is using the Hydra-based CLI launcher:

```bash
# Run circle packing example with default settings
shinka_launch variant=circle_packing_example

# Run with custom parameters
shinka_launch \
    task=circle_packing \
    database=island_small \
    evolution=small_budget \
    cluster=local \
    evo_config.num_generations=5
```

### Python API Usage

For more control, you can use the Python API directly:

```python
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Configure the job execution environment
job_config = LocalJobConfig(
    eval_program_path="examples/circle_packing/evaluate.py",
    conda_env="my_special_env",  # Optional: run in specific conda environment
)

# Configure the evolution database
db_config = DatabaseConfig(
    archive_size=20,
    num_archive_inspirations=4,
    num_islands=2,
    migration_interval=10,
)

# Configure the evolution parameters
evo_config = EvolutionConfig(
    num_generations=10,
    max_parallel_jobs=1,
    llm_models=["azure-gpt-4.1"],
    init_program_path="examples/circle_packing/initial.py",
    language="python",
    task_sys_msg="You are optimizing circle packing...",
)

# Run the evolution
runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)
runner.run()
```

For detailed configuration options and advanced settings, see the [Configuration Guide](configuration.md).

## Examples

### Circle Packing Example

This example demonstrates optimizing the arrangement of 26 circles in a unit square to maximize the sum of their radii.

#### Files Structure
```
examples/circle_packing/
├── initial.py     # Starting solution
├── evaluate.py    # Evaluation script
└── run_evo.py     # Direct Python runner
```

#### Running the Example

```bash
# Using CLI launcher (recommended)
shinka_launch variant=circle_packing_example

# Or with custom settings
shinka_launch \
    task=circle_packing \
    cluster=local \
    evo_config.num_generations=20 \
    db_config.num_islands=4

# Or just via the python API
python run_evo.py
```

#### Understanding the Initial Code Solution

The `initial.py` contains the code that will be evolved:

```python
# EVOLVE-BLOCK-START
def construct_packing():
    """Construct arrangement of 26 circles in unit square"""
    # This code will be modified by the LLM
    n = 26
    centers = np.zeros((n, 2))
    # ... placement logic ...
    return centers, radii
# EVOLVE-BLOCK-END
```

The `EVOLVE-BLOCK-START/END` markers define which parts of the code can be modified during evolution.

#### Understanding the Evaluation Script

The `evaluate.py` script uses Shinka's `run_shinka_eval` function to test and score evolved solutions:

```python
from shinka.core import run_shinka_eval

def main(program_path: str, results_dir: str):
    """Main evaluation function called by Shinka"""

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_packing",        # Function to call in evolved code
        num_runs=1,                              # Number of test runs
        get_experiment_kwargs=get_kwargs_fn,     # Arguments for each run
        validate_fn=validation_function,         # Validation logic
        aggregate_metrics_fn=metrics_function,   # Metrics computation
    )
```

**Key Components:**

**1. Validation Function** - Checks if solutions meet constraints:
```python
def validate_packing(run_output):
    """Returns (is_valid: bool, error_msg: str or None)"""
    centers, radii, reported_sum = run_output

    # Check constraints (bounds, overlaps, etc.)
    if constraint_violated:
        return False, "Specific error description"

    return True, None  # Valid solution
```

**2. Metrics Aggregation Function** - Computes fitness and organizes results:
```python
def aggregate_metrics(results, results_dir):
    """Returns metrics dictionary with required structure"""

    # Extract data from results
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

**What run_shinka_eval Returns:**

The `run_shinka_eval` function returns three values:

1. **`metrics` (dict)**: Structured performance data
   - `combined_score`: Primary fitness value (higher = better)
   - `public`: Metrics shown in WebUI and logs
   - `private`: Internal metrics for analysis

2. **`correct` (bool)**: Whether solution passed validation
   - `True`: Solution is valid, can reproduce
   - `False`: Solution failed, will be discarded

3. **`error_msg` (str or None)**: Error description if validation failed

**Public vs Private Metrics:**
- **Public**: Displayed in WebUI, included in logs, used for monitoring
- **Private**: Internal analysis, debugging, not shown in main interface



### Other Available Examples

| Example | Description | Use Case |
|---------|-------------|----------|
| **Circle Packing** | Optimize circle arrangements | Geometric optimization |
| **Agent Design** | Design AI agent scaffolds | Algorithm architecture |
| **ALE-Bench** | Optimize competitive programming solutions | Code optimization |
| **Novelty Generator** | Generate diverse creative outputs | Open-ended exploration |



## Advanced Usage

### Resuming Experiments

If you need to pause and resume an evolutionary run, or extend a completed run with more generations, Shinka supports seamless resumption from existing results.

#### How Resuming Works

When you specify an existing `results_dir` that contains a database, Shinka will:
- Detect the previous run automatically
- Restore the population database and all program history
- Resume meta-recommendations from the last checkpoint
- Continue from the last completed generation

#### Using the CLI (Hydra)

```bash
# Resume an existing run and extend to 50 generations
shinka_launch \
    variant=circle_packing_example \
    evo_config.results_dir=results_20250101_120000 \
    evo_config.num_generations=50

# Or with a custom task
shinka_launch \
    task=circle_packing \
    database=island_small \
    evolution=small_budget \
    cluster=local \
    evo_config.results_dir=path/to/previous/results \
    evo_config.num_generations=100
```

#### Using the Python API

```python
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Point to existing results directory
evo_config = EvolutionConfig(
    num_generations=50,  # Extend to 50 total generations
    results_dir="results_20250101_120000",  # Existing results
    # ... other config parameters ...
)

job_config = LocalJobConfig(
    eval_program_path="examples/circle_packing/evaluate.py",
)

db_config = DatabaseConfig(
    archive_size=20,
    num_islands=2,
)

# Run will automatically detect and resume
runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)
runner.run()
```

**Important Notes:**
- The `num_generations` parameter should be set to the **total** number of generations you want (not additional generations)
- For example, if you completed 20 generations and want 30 more, set `num_generations=50`
- The database configuration (number of islands, archive size, etc.) should match the original run
- All previous progress, including the best solutions and meta-recommendations, will be preserved

### Environment Management for Local Jobs

When running jobs locally, you have several options for managing Python environments:

#### Option 1: Use Current Environment (Default)
```python
job_config = LocalJobConfig(
    eval_program_path="evaluate.py"
)
# Uses the currently active Python environment
```

#### Option 2: Use Specific Conda Environment
```python
job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    conda_env="my_project_env"  # Runs in specified conda environment
)
```

This is particularly useful when:
- Different experiments require different dependency versions
- You want to isolate evaluation environments from your main development environment
- Testing compatibility across multiple Python/package versions

### Creating Custom Tasks

1. **Define the Problem**: Create task config in `configs/task/my_task.yaml`
2. **Initial Solution**: Write `initial.py` with `EVOLVE-BLOCK` markers
3. **Evaluation Script**: Create `evaluate.py` with validation logic
4. **Variant Config**: Combine settings in `configs/variant/my_variant.yaml`

For detailed configuration options, parameter explanations, and advanced patterns, see the [Configuration Guide](configuration.md).

### Code Evolution Animation

Generate animations showing how code evolves:

```bash
python code_path_anim.py --results_dir examples/circle_packing/results_20250101_120000
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# If using uv
uv pip install -e .
# If using pip
pip install -e .
# Check Python path
python -c "import shinka; print(shinka.__file__)"
```

**2. API Key Issues**
```bash
# Verify .env file exists and contains valid keys
cat .env
# Check environment variables
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

**3. Evaluation Failures**
- Check that your evaluation script has correct function signatures
- Verify the `EVOLVE-BLOCK` markers are properly placed
- Ensure the evaluation function returns expected data types

**4. Memory Issues**
- Reduce `max_parallel_jobs` for local execution
- Increase memory allocation for cluster jobs
- Monitor database size and archive settings

**5. uv-Specific Issues**
```bash
# Check uv version
uv --version

# Verify virtual environment is activated
which python  # Should point to .venv/bin/python

# Reset environment if needed
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Check uv cache if having dependency issues
uv cache clean
```

**6. Conda Environment Issues (Local Jobs)**
```bash
# Verify conda environment exists
conda env list

# Test conda environment works
conda run -n my_env python --version

# Check if required packages are installed in target environment
conda run -n my_env python -c "import shinka; print('OK')"

# Install shinka in specific conda environment
conda activate my_env
pip install -e .
conda deactivate
```

### Debug Mode

Enable verbose logging:
```bash
shinka_launch variant=my_variant verbose=true
```

### Getting Help

- Check the [examples](../examples/) directory for reference implementations
- See the [Configuration Guide](configuration.md) for detailed parameter explanations
- Examine the generated experiment logs in the results directory

## Next Steps

Now that you have Shinka running:

1. **Try the Examples**: Run the circle packing example to see evolution in action
2. **Explore the WebUI**: See the [WebUI Guide](webui.md) to visualize how solutions evolve
3. **Create Custom Tasks**: Adapt the framework to your specific optimization problems
4. **Scale Up**: Deploy on clusters for large-scale evolutionary experiments

Happy evolving! 🧬
