# Shinka Configuration Guide ⚙️

This guide covers the comprehensive configuration system in Shinka, including all parameters, file structures, and advanced configuration patterns.

## Table of Contents

1. [Core Configuration Components](#core-configuration-components)
2. [Configuration Parameters](#configuration-parameters)
3. [Pre-configured Variants](#pre-configured-variants)
4. [Configuration Structure](#configuration-structure)
5. [Creating Custom Configurations](#creating-custom-configurations)
6. [Advanced Configuration Patterns](#advanced-configuration-patterns)
7. [Configuration Examples](#configuration-examples)
8. [Configuration Best Practices](#configuration-best-practices)


## Core Configuration Components

### 1. Evolution Config (`evo_config`)

Controls the core evolutionary algorithm parameters:

```yaml
evo_config:
  _target_: shinka.core.EvolutionConfig
  num_generations: 20              # Number of evolution generations
  max_parallel_jobs: 1             # Maximum parallel evaluations
  max_patch_attempts: 10           # Max attempts to generate valid patches
  
  # LLM Configuration
  llm_models:                      # List of LLM models for mutations
    - "azure-gpt-4.1"
  llm_dynamic_selection: null      # Dynamic model selection strategy
  embedding_model: "text-embedding-3-small"
  
  # Patch Configuration
  patch_types:                     # Types of code modifications
    - "diff"                       # Diff-based patches
    - "full"                       # Full code replacement
  patch_type_probs:                # Probabilities for each patch type
    - 0.5
    - 0.5
  
  # Task Configuration
  language: "python"               # Programming language
  init_program_path: "???"         # Path to initial program
  task_sys_msg: "???"             # System message for LLM
  job_type: "local"                # Job execution type
  results_dir: ${output_dir}       # Results directory
```

### 2. Database Config (`db_config`)

Manages the evolutionary database and island topology:

```yaml
db_config:
  _target_: shinka.database.DatabaseConfig
  db_path: "evolution_db.sqlite"   # SQLite database path
  
  # Island Configuration
  num_islands: 2                   # Number of evolutionary islands
  island_elitism: true             # Enable elite preservation per island
  
  # Archive Configuration
  archive_size: 20                 # Size of elite solution archive
  num_archive_inspirations: 4      # Solutions drawn from archive
  num_top_k_inspirations: 2        # Solutions from current generation
  
  # Selection and Migration
  exploitation_ratio: 0.2          # Exploitation vs exploration balance
  elite_selection_ratio: 0.3       # Fraction of elites for selection
  migration_interval: 10           # Generations between migrations
  migration_rate: 0.1              # Fraction of population migrated
```

### 3. Job Config (`job_config`)

Defines the execution environment and resource requirements:

#### Local Execution
```yaml
job_config:
  _target_: shinka.launch.LocalJobConfig
  eval_program_path: "shinka/evaluate.py"
```

#### Slurm Cluster Execution
```yaml
job_config:
  _target_: shinka.launch.SlurmCondaJobConfig
  modules:                         # Environment modules
    - "cuda/12.4"
    - "cudnn/8.9.7"
    - "hpcx/2.20"
  eval_program_path: "shinka/utils/eval_hydra.py"
  conda_env: "shinka"              # Conda environment name
  time: "01:00:00"                 # Maximum job runtime
  cpus: 4                          # CPU cores per job
  gpus: 1                          # GPUs per job
  mem: "16G"                       # Memory per job
```

### 4. Task Config

Defines problem-specific settings and evaluation functions:

```yaml
# Task-specific evaluation function
evaluate_function:
  _target_: examples.my_task.evaluate.main
  program_path: ???               # Filled by runner
  results_dir: ???                # Filled by runner

# Job configuration for this task
distributed_job_config:
  _target_: shinka.launch.SlurmCondaJobConfig
  # ... resource requirements ...

# Evolution settings specific to this task
evo_config:
  task_sys_msg: |
    You are an expert in [domain].
    Key insights: [domain knowledge]
  language: "python"
  init_program_path: "examples/my_task/initial.py"
  job_type: "slurm_conda"

exp_name: "shinka_my_task"
```

## Configuration Parameters

### Evolution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | 20 | Number of evolutionary generations |
| `max_parallel_jobs` | int | 1 | Maximum concurrent evaluations |
| `max_patch_attempts` | int | 10 | Maximum attempts to generate valid patches |
| `llm_models` | list | `["azure-gpt-4.1"]` | LLM models for mutations |
| `patch_types` | list | `["diff", "full"]` | Types of code modifications |
| `patch_type_probs` | list | `[0.5, 0.5]` | Probabilities for patch types |
| `language` | str | `"python"` | Programming language |
| `embedding_model` | str | `"text-embedding-3-small"` | Model for code embeddings |

### Database Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_islands` | int | 2 | Number of evolutionary islands |
| `archive_size` | int | 20 | Size of elite solution archive |
| `num_archive_inspirations` | int | 4 | Solutions drawn from archive |
| `num_top_k_inspirations` | int | 2 | Solutions from current generation |
| `exploitation_ratio` | float | 0.2 | Balance between exploitation/exploration |
| `elite_selection_ratio` | float | 0.3 | Fraction of elites for selection |
| `migration_interval` | int | 10 | Generations between island migrations |
| `migration_rate` | float | 0.1 | Fraction of population migrated |
| `island_elitism` | bool | true | Preserve elites per island |
| `migration_adaptation` | dict | null | Optional per-island adaptive migration settings |

### Resource Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | str | `"01:00:00"` | Maximum job runtime (HH:MM:SS) |
| `cpus` | int | 4 | CPU cores per job |
| `gpus` | int | 0 | GPUs per job |
| `mem` | str | `"8G"` | Memory per job |
| `conda_env` | str | `"shinka"` | Conda environment name |
| `modules` | list | `[]` | Environment modules to load |

#### Adaptive Migration Settings

The optional `migration_adaptation` block enables automatic adjustment of
`migration_rate`, `migration_interval`, and `island_elitism` per island. Each
method (`success`, `diversity`, `bandit`) can be toggled independently and has a
dedicated configuration block. Refer to `docs/migration_adaptation.md` and the
sample `configs/database/island_adaptive.yaml` file for a complete walkthrough.

## Pre-configured Variants

Shinka uses [Hydra](https://hydra.cc/) for flexible, hierarchical configuration management. The system is designed around composable configuration files that can be mixed and matched to create different experimental setups.

Variants provide pre-configured combinations of settings for common use cases:

### Circle Packing Example
```yaml
# configs/variant/circle_packing_example.yaml
defaults:
  - override /database@_global_: island_large
  - override /evolution@_global_: large_budget
  - override /task@_global_: circle_packing
  - override /cluster@_global_: local
  - _self_

variant_suffix: "_example"
```

### Agent Design Example
```yaml
# configs/variant/agent_design_example.yaml
defaults:
  - override /database@_global_: island_medium
  - override /evolution@_global_: medium_budget
  - override /task@_global_: agent_design
  - override /cluster@_global_: local
  - _self_

evo_config:
  num_generations: 15

variant_suffix: "_agent_example"
```

## Configuration Structure

```
configs/
├── config.yaml           # Main config file with defaults
├── cluster/              # Execution environments
│   ├── local.yaml        # Local execution
│   ├── gcp.yaml          # Google Cloud Platform
│   └── remote.yaml       # Remote Slurm clusters
├── database/             # Evolution database settings
│   ├── island_small.yaml # Small-scale evolution (2 islands)
│   ├── island_medium.yaml# Medium-scale evolution (4 islands)
│   └── island_large.yaml # Large-scale evolution (8+ islands)
├── evolution/            # Evolution parameters
│   ├── small_budget.yaml # Few generations, quick runs
│   ├── medium_budget.yaml# Moderate computational budget
│   └── large_budget.yaml # Extensive evolution runs
├── task/                 # Problem definitions
│   ├── circle_packing.yaml
│   ├── agent_design.yaml
│   ├── bbo_search.yaml
│   ├── cifar10.yaml
│   ├── cuda_optim.yaml
│   ├── mad_moe.yaml
│   └── novelty_generator.yaml
└── variant/              # Pre-configured combinations
    ├── circle_packing_example.yaml
    ├── agent_design_example.yaml
    ├── mad_moe_example.yaml
    └── default.yaml
```

## Creating Custom Configurations

### Method 1: Custom Variant File

Create a new variant file combining existing components:

```yaml
# configs/variant/my_custom_variant.yaml
defaults:
  - override /database@_global_: island_small
  - override /evolution@_global_: small_budget
  - override /task@_global_: my_task
  - override /cluster@_global_: local
  - _self_

# Override specific parameters
evo_config:
  num_generations: 25
  max_parallel_jobs: 2

db_config:
  archive_size: 30
  migration_interval: 5

variant_suffix: "_custom"
```

Launch with:
```bash
shinka_launch variant=my_custom_variant
```

### Method 2: Command Line Overrides

Override parameters directly on the command line:

```bash
shinka_launch \
    task=circle_packing \
    database=island_large \
    evolution=medium_budget \
    cluster=local \
    evo_config.num_generations=50 \
    evo_config.max_parallel_jobs=4 \
    db_config.num_islands=6 \
    variant_suffix="_custom_run"
```

### Method 3: Custom Task Configuration

Create a new task configuration:

```yaml
# configs/task/my_optimization_task.yaml
evaluate_function:
  _target_: examples.my_optimization.evaluate.main
  program_path: ???
  results_dir: ???

distributed_job_config:
  _target_: shinka.launch.LocalJobConfig
  eval_program_path: "shinka/utils/eval_hydra.py"

evo_config:
  task_sys_msg: |
    You are an expert optimization researcher working on [specific problem].
    
    Key insights to consider:
    1. [Domain-specific insight 1]
    2. [Domain-specific insight 2]
    3. [Domain-specific insight 3]
    
    Focus on [specific optimization goals].
  language: "python"
  init_program_path: "examples/my_optimization/initial.py"
  job_type: "local"

exp_name: "shinka_my_optimization"
```

## Advanced Configuration Patterns

### Multi-Model Evolution

Use multiple LLM models with different strengths:

```yaml
evo_config:
  llm_models:
    - "azure-gpt-4.1"      # Strong reasoning
    - "claude-3-sonnet"    # Good at code
    - "azure-gpt-4o-mini"  # Fast iterations
  
  # Optional: Dynamic model selection
  llm_dynamic_selection:
    strategy: "performance_based"
    window_size: 10
```

## Configuration Examples

### Quick Prototyping Setup
```yaml
# Fast iteration for development
defaults:
  - override /database@_global_: island_small
  - override /evolution@_global_: small_budget
  - override /cluster@_global_: local

evo_config:
  num_generations: 5
  max_parallel_jobs: 1

db_config:
  num_islands: 1
  archive_size: 10

variant_suffix: "_prototype"
```

### Production Research Setup
```yaml
# Large-scale research experiment
defaults:
  - override /database@_global_: island_large
  - override /evolution@_global_: large_budget
  - override /cluster@_global_: remote

evo_config:
  num_generations: 100
  max_parallel_jobs: 8

db_config:
  num_islands: 8
  archive_size: 50
  migration_interval: 5

variant_suffix: "_production"
```

### Multi-Task Comparison
```yaml
# Configuration for comparing across tasks
defaults:
  - override /database@_global_: island_medium
  - override /evolution@_global_: medium_budget
  - override /cluster@_global_: local

# Standardized parameters for fair comparison
evo_config:
  num_generations: 30
  max_parallel_jobs: 2
  llm_models: ["azure-gpt-4.1"]

db_config:
  num_islands: 4
  archive_size: 25

variant_suffix: "_comparison"
```

## Configuration Best Practices

### 1. Start Small, Scale Up
- Begin with `island_small` and `small_budget` configurations
- Increase complexity as you understand the problem better

### 2. Use Meaningful Variant Suffixes
- Include key parameters in the suffix: `_gen50_islands4_gpt4`
- This helps identify experiments in results directories

### 3. Document Custom Configurations
- Add comments explaining parameter choices
- Include expected runtime and resource usage

### 4. Version Control Configurations
- Keep variant files in version control
- Tag configurations used for important results

### 5. Monitor Resource Usage
- Start with conservative resource allocations
- Monitor actual usage and adjust accordingly

For more examples and detailed parameter explanations, see the configuration files in the `configs/` directory and the [Getting Started Guide](getting_started.md).
