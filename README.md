<h1 align="center">
  <a href="shinka/favicon.png?raw=true"><img src="shinka/favicon.png?raw=true" width="180" /></a><br>
  <b><code>ShinkaEvolve</code>: Towards Open-Ended and Sample-Efficient Program Evolution 🧬</b><br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" />
  <a href="https://github.com/SakanaAI/ShinkaEvolve/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://arxiv.org/abs/2509.19349"><img src="http://img.shields.io/badge/paper-arxiv.2509.19349-B31B1B.svg" /></a>
  <a href="https://sakana.ai/shinka-evolve/"><img src="https://img.shields.io/badge/SakanaAI-0A66C2.svg" /></a>
  <a href="https://colab.research.google.com/github/SakanaAI/ShinkaEvolve/blob/main/examples/shinka_tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>


`shinka` is a framework that combines Large Language Models (LLMs) with evolutionary algorithms to drive scientific discovery. By leveraging the creative capabilities of LLMs and the optimization power of evolutionary search, `shinka` enables automated exploration and improvement of scientific code. The system is inspired by the [AI Scientist](https://sakana.ai/ai-scientist/), [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) and the [Darwin Goedel Machine](https://sakana.ai/dgm/): It maintains a population of programs that evolve over generations, with an ensemble of LLMs acting as intelligent mutation operators that suggest code improvements.

---

**Feb 2026 Update**: Added [`skills/shinka-setup/SKILL.md`](skills/shinka-setup/SKILL.md) for agentic task generation and verification.

**Feb 2026 Update**: ShinkaEvolve was accepted at ICLR 2026 and we have [released v1.1](docs/release_notes.md) with many new features.

**Nov 2025 Update**: Rob gave several public talks about our ShinkaEvolve effort ([Official](https://x.com/SakanaAILabs/status/1989352976792846356?s=20), [AutoML Seminar](https://www.youtube.com/watch?v=dAOIer_1INo)).

**Oct 2025 Update** ShinkaEvolve supported Team Unagi in winning the [ICFP 2025 Programming Contest](https://sakana.ai/icfp-2025/).

---

The framework supports **parallel evaluation of candidates** locally or on a Slurm cluster. It maintains an archive of successful solutions, enabling knowledge transfer between different evolutionary islands. `shinka` is particularly well-suited for scientific tasks where there is a verifier available and the goal is to optimize performance metrics while maintaining code correctness and readability.

![](docs/conceptual.png)

## Documentation 📝

| Guide | Description | What You'll Learn |
|-------|-------------|-------------------|
| 🚀 **[Getting Started](docs/getting_started.md)** | Installation, basic usage, and examples | Setup, first evolution run, core concepts |
| 📓 **[Tutorial](examples/shinka_tutorial.ipynb)** | Interactive walkthrough of Shinka features | Hands-on examples, configuration, best practices |
| ⚙️ **[Configuration](docs/configuration.md)** | Comprehensive configuration reference | All config options, optimization settings, advanced features |
| 🎨 **[WebUI](docs/webui.md)** | Interactive visualization and monitoring | Real-time tracking, result analysis, debugging tools | 
| ⚡ **[Async Evolution](docs/async_evolution.md)** | High-performance async pipeline (5-10x speedup) | Concurrent processing, performance tuning, migration guide | 
| 🧠 **[Local LLM](docs/support_local_llm.md)** | How to connect and use local LLMs with Shinka | Running open-source models, integration tips, performance notes |


## Installation & Quick Start 🚀

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install Shinka
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run your first evolution experiment
shinka_launch variant=circle_packing_example
```

For detailed installation instructions and usage examples, see the [Getting Started Guide](docs/getting_started.md).

## Examples 📖

| Example | Description | Environment Setup |
|---------|-------------|-------------------|
| ⭕ [Circle Packing](examples/circle_packing) | Optimize circle packing to maximize radii. | `LocalJobConfig` |
| 🎮 [Game 2048](examples/game_2048) | Optimize a policy for the Game of 2048. | `LocalJobConfig` |
| ∑ [Julia Prime Counting](examples/julia_prime_counting) | Optimize a Julia solver for prime-count queries. | `LocalJobConfig` |
| ✨ [Novelty Generator](examples/novelty_generator) | Generate creative, surprising outputs (e.g., ASCII art). | `LocalJobConfig` |


## `shinka` Run with Python API 🐍

For the simplest setup with default settings, you only need to specify the evaluation program:

<table>
<tr>
<td width="50%">

**`EvolutionRunner` - Synchronous**

```python
from shinka.core import EvolutionRunner
from shinka.core import EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Minimal - only specify what's required
job_conf = LocalJobConfig(
    eval_program_path="evaluate.py",
)
db_conf = DatabaseConfig()
evo_conf = EvolutionConfig(
    init_program_path="initial.py",
)

# Evolution: parallel evals & seq. gen.
runner = EvolutionRunner(
    evo_config=evo_conf,
    job_config=job_conf,
    db_config=db_conf,
)

runner.run()
```

</td>
<td width="50%">

**`AsyncEvolutionRunner` - Asynchronous**

```python
import asyncio
from shinka.core import AsyncEvolutionRunner, ...

async def main():
    # Same import & setup of hyperparams
    job_conf = LocalJobConfig(...)
    db_conf = DatabaseConfig()
    evo_conf = EvolutionConfig(...)
    
    # Async evolution with concurrent proposals
    runner = AsyncEvolutionRunner(
        evo_config=evo_conf,
        job_config=job_conf,
        db_config=db_conf,
        max_proposal_jobs=10,  # Proposals workers
        max_evaluation_jobs=10,  # Proposal workers
    )

    await runner.run()

# Asyncio execution
asyncio.run(main())
```
</td>
</tr>
</table>

<details>
<summary><strong>EvolutionConfig Parameters</strong> (click to expand)</summary>

| Key | Default Value | Type | Explanation |
|-----|---------------|------|-------------|
| `task_sys_msg` | `None` | `Optional[str]` | System message describing the optimization task |
| `patch_types` | `["diff"]` | `List[str]` | Types of patches to generate: "diff", "full", "cross" |
| `patch_type_probs` | `[1.0]` | `List[float]` | Probabilities for each patch type |
| `num_generations` | `10` | `int` | Number of evolution generations to run |
| `max_parallel_jobs` | `2` | `int` | Maximum number of parallel evaluation jobs |
| `max_patch_resamples` | `3` | `int` | Max times to resample a patch if it fails |
| `max_patch_attempts` | `5` | `int` | Max attempts to generate a valid patch |
| `job_type` | `"local"` | `str` | Job execution type: "local", "slurm_docker", "slurm_conda" |
| `language` | `"python"` | `str` | Programming language for evolution |
| `llm_models` | `["azure-gpt-4.1-mini"]` | `List[str]` | List of LLM models for code generation |
| `llm_dynamic_selection` | `None` | `Optional[Union[str, BanditBase]]` | Dynamic model selection strategy |
| `llm_dynamic_selection_kwargs` | `{}` | `dict` | Kwargs for dynamic selection |
| `llm_kwargs` | `{}` | `dict` | Additional kwargs for LLM calls |
| `meta_rec_interval` | `None` | `Optional[int]` | Interval for meta-recommendations |
| `meta_llm_models` | `None` | `Optional[List[str]]` | LLM models for meta-recommendations |
| `meta_llm_kwargs` | `{}` | `dict` | Kwargs for meta-recommendation LLMs |
| `meta_max_recommendations` | `5` | `int` | Max number of meta-recommendations |
| `embedding_model` | `None` | `Optional[str]` | Model for code embeddings |
| `init_program_path` | `"initial.py"` | `Optional[str]` | Path to initial program to evolve |
| `results_dir` | `None` | `Optional[str]` | Directory to save results (auto-generated if None) |
| `max_novelty_attempts` | `3` | `int` | Max attempts for novelty generation |
| `code_embed_sim_threshold` | `1.0` | `float` | Similarity threshold for code embeddings |
| `novelty_llm_models` | `None` | `Optional[List[str]]` | LLM models for novelty judgment |
| `novelty_llm_kwargs` | `{}` | `dict` | Kwargs for novelty LLMs |
| `use_text_feedback` | `False` | `bool` | Whether to use text feedback in evolution |

</details>

<details>
<summary><strong>DatabaseConfig Parameters</strong> (click to expand)</summary>

| Key | Default Value | Type | Explanation |
|-----|---------------|------|-------------|
| `db_path` | `None` | `Optional[str]` | Database file path (auto-generated if None) |
| `num_islands` | `4` | `int` | Number of evolution islands for diversity |
| `archive_size` | `100` | `int` | Size of program archive per island |
| `elite_selection_ratio` | `0.3` | `float` | Proportion of elite programs for inspiration |
| `num_archive_inspirations` | `5` | `int` | Number of archive programs to use as inspiration |
| `num_top_k_inspirations` | `2` | `int` | Number of top-k programs for inspiration |
| `migration_interval` | `10` | `int` | Generations between island migrations |
| `migration_rate` | `0.1` | `float` | Proportion of island population to migrate |
| `island_elitism` | `True` | `bool` | Keep best programs on their original islands |
| `enforce_island_separation` | `True` | `bool` | Enforce full separation between islands |
| `parent_selection_strategy` | `"power_law"` | `str` | Parent selection: "weighted", "power_law", "beam_search" |
| `exploitation_alpha` | `1.0` | `float` | Power-law exponent (0=uniform, 1=power-law) |
| `exploitation_ratio` | `0.2` | `float` | Chance to pick parent from archive |
| `parent_selection_lambda` | `10.0` | `float` | Sharpness of sigmoid for weighted selection |
| `num_beams` | `5` | `int` | Number of beams for beam search selection |

</details>

<details>
<summary><strong>JobConfig Parameters</strong> (click to expand)</summary>

**LocalJobConfig** (for local execution):
| Key | Default Value | Type | Explanation |
|-----|---------------|------|-------------|
| `eval_program_path` | `"evaluate.py"` | `Optional[str]` | Path to evaluation script |
| `extra_cmd_args` | `{}` | `Dict[str, Any]` | Additional command line arguments |
| `time` | `None` | `Optional[str]` | Time limit for job execution |
| `conda_env` | `None` | `Optional[str]` | Conda environment to run jobs in |

**SlurmDockerJobConfig** (for SLURM with Docker):
| Key | Default Value | Type | Explanation |
|-----|---------------|------|-------------|
| `eval_program_path` | `"evaluate.py"` | `Optional[str]` | Path to evaluation script |
| `extra_cmd_args` | `{}` | `Dict[str, Any]` | Additional command line arguments |
| `image` | `"ubuntu:latest"` | `str` | Docker image to use |
| `image_tar_path` | `None` | `Optional[str]` | Path to Docker image tar file |
| `docker_flags` | `""` | `str` | Additional Docker flags |
| `partition` | `"gpu"` | `str` | SLURM partition to use |
| `time` | `"01:00:00"` | `str` | Job time limit |
| `cpus` | `1` | `int` | Number of CPUs to request |
| `gpus` | `1` | `int` | Number of GPUs to request |
| `mem` | `"8G"` | `Optional[str]` | Memory to request |

**SlurmCondaJobConfig** (for SLURM with Conda):
| Key | Default Value | Type | Explanation |
|-----|---------------|------|-------------|
| `eval_program_path` | `"evaluate.py"` | `Optional[str]` | Path to evaluation script |
| `extra_cmd_args` | `{}` | `Dict[str, Any]` | Additional command line arguments |
| `conda_env` | `""` | `str` | Conda environment name |
| `modules` | `[]` | `Optional[List[str]]` | Environment modules to load |
| `partition` | `"gpu"` | `str` | SLURM partition to use |
| `time` | `"01:00:00"` | `str` | Job time limit |
| `cpus` | `1` | `int` | Number of CPUs to request |
| `gpus` | `1` | `int` | Number of GPUs to request |
| `mem` | `"8G"` | `Optional[str]` | Memory to request |

</details>

### Evaluation Setup & Initial Solution 🏃

To use EvolutionRunner, you need two key files: The **`evaluate.py`** script defines how to test and score your programs - it runs multiple evaluations, validates results, and aggregates them into metrics that guide the `shinka` evolution loop. The **`initial.py`** file contains your starting solution with the core algorithm that will be iteratively improved by LLMs across generations.

<table>
<tr>
<td width="50%">

**`evaluate.py` - Evaluation Script**

```python
from shinka.core import run_shinka_eval

def main(program_path: str,
         results_dir: str):
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=3, # Multi-evals to aggreg.
        run_workers=1,  # >1 enables per-run process parallelism
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,  # Optional
    )

def get_kwargs(run_idx: int) -> dict:
    return {"param1": "value", "param2": 42}

def aggregate_fn(results: list) -> dict:
    score = results[0]
    text = results[1]
    return {
        "combined_score": float(score),
        "public": {...},  # shinka-visible
        "private": {...},  # shinka-invisible
        "extra_data": {...},  # store as pkl
        "text_feedback": text,  # str fb
    }

if __name__ == "__main__":
    # argparse program path & dir
    main(program_path, results_dir)
```

</td>
<td width="50%">

**`initial.py` - Starting Solution**

```python
# EVOLVE-BLOCK-START
def advanced_algo():
    # This will be evolved
    return solution
# EVOLVE-BLOCK-END

def run_experiment(**kwargs):
    """Main called by evaluator"""
    result = solve_problem(kwargs)
    return result

def solve_problem(params):
    solution = advanced_algo()
    return solution
```

**Key Points:**
- Eval name matches `experiment_fn_name`
- Use `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` to mark evolution sections
- Return format matches validation expectations
- Dependencies must be available in env
- Results can be unpacked for metrics
- Auto-stores several results in `results_dir`
- Can add text feedback in `shinka` loop
- Higher `combined_score` values indicate better performance (maximization)

</td>
</tr>
</table>


## `shinka` Launcher with Hydra 🚀

`shinka` Launcher utilizes [Hydra](https://hydra.cc/) to configure and launch evolutionary experiments effortlessly. It supports concise configuration via Hydra's powerful override syntax, making it easy to manage and iterate scientific explorations.

```bash
# Run with pre-configured variant
shinka_launch variant=circle_packing_example

# Run with custom parameters
shinka_launch \
    task=circle_packing \
    database=island_large \
    evolution=small_budget \
    cluster=local \
    evo_config.num_generations=20
```

For comprehensive configuration options and advanced usage, see the [Configuration Guide](docs/configuration.md).


## Interactive WebUI 🎨

Monitor your evolution experiments in real-time with Shinka's interactive web interface! The WebUI provides live visualization of the evolutionary process, genealogy trees, and performance metrics.

![WebUI Screenshot](docs/webui.png)

### Quick Start

Launch the WebUI alongside your evolution experiment:

```bash
# Start your evolution experiment
shinka_launch variant=circle_packing_example

# In another terminal, launch the WebUI
shinka_visualize --port 8888 --open
```

For detailed WebUI documentation, see the [WebUI Guide](docs/webui.md).

## Related Open-Source Projects 🧑‍🔧

- [OpenEvolve](https://github.com/codelion/openevolve): An open-source implementation of AlphaEvolve
- [LLM4AD](https://github.com/Optima-CityU/llm4ad): A Platform for Algorithm Design with Large Language Model

## Citation ✍️

If you use `ShinkaEvolve` in your research, please cite it as follows:

```
@article{lange2025shinka,
  title={ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution},
  author={Lange, Robert Tjarko and Imajuku, Yuki and Cetin, Edoardo},
  journal={arXiv preprint arXiv:2509.19349},
  year={2025}
}
```
