# AEC Packing - Toilet Layout Optimization

This example demonstrates using ShinkaEvolve for **Architecture, Engineering, and Construction (AEC)** applications, specifically optimizing toilet layout packing in bathrooms to minimize space while satisfying design constraints.

## Problem Description

The goal is to find an optimal toilet layout that minimizes the total perimeter (and thus required space) while satisfying hard constraints related to accessibility, fixture placement, and corridor design.

### Design Elements

**Element Types:**
- **WC (Water Closet)**: Toilet fixtures
- **Urinal**: Urinal fixtures  
- **Wash Basin**: Sink fixtures
- **Corridor**: Passage areas for accessibility

**Representation:**
- **Blue (B)**: Fixture outlines
- **Red (R)**: Corridor/passage areas
- **Index**: Variations within the same type (e.g., WC-1, WC-2 with different door positions)

### Constraints

#### Hard Constraints (Must Satisfy)
1. **No B/B Overlap**: Blue fixtures cannot overlap with each other
2. **No B/R Overlap**: Blue fixtures cannot overlap with red corridors
3. **Type Adjacency**: Fixtures of the same type must be placed adjacent to each other
4. **Accessibility**: Corridors (R) cannot be completely enclosed by fixtures (B) - all fixtures must be accessible
5. **Perimeter Boundary**: Green line must border the perimeter

#### Soft Constraints (Optimization Goals)
1. **Minimize Total Perimeter**: Minimize sum of perimeters of all WC, Urinal, and Wash Basin elements
2. **R/R Overlap Preference**: Corridors should preferably overlap to consolidate passage space
3. **Minimize Corridor Count**: Use fewer corridors for area reduction

## Project Structure

```
aec_packing/
├── README.md           # This file
├── initial.py          # Initial packing solution (starting point)
├── evaluate.py         # Evaluation and validation logic
├── run_evo.py         # Evolution configuration and runner
└── results_cpack/     # Results directory (created during run)
    ├── evolution_db.sqlite  # Evolution database
    ├── logs/               # Execution logs
    └── programs/           # Generated program variants
```

## Prerequisites

### 1. Install ShinkaEvolve

```bash
# Navigate to ShinkaEvolve root directory
cd ShinkaEvolve

# Create virtual environment (using uv - recommended)
uv venv --python 3.11
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install ShinkaEvolve
uv pip install -e .
```

### 2. Set Up API Keys

Create a `.env` file in the ShinkaEvolve root directory:

```bash
# .env file
OPENAI_API_KEY=sk-proj-your-key-here       # For OpenAI models (text-embedding-3-small, gpt-4o, etc.)
GEMINI_API_KEY=your-gemini-key-here        # For Gemini models (gemini-2.5-flash, gemini-2.5-pro, etc.)
ANTHROPIC_API_KEY=your-anthropic-key-here  # For Anthropic models (claude-sonnet, etc.)
```

**Important**: You only need to provide the API key(s) for the model(s) you plan to use. By default, this example uses:
- `gemini-2.5-flash` for LLM operations (requires `GEMINI_API_KEY`)
- `text-embedding-3-small` for embeddings (requires `OPENAI_API_KEY`)

If you only have one provider's API key, edit `run_evo.py` to use models from that provider for both LLM and embedding operations. For example:
- **Gemini only**: Use `gemini-2.5-flash` for LLM and `gemini-embedding-001` for embeddings
- **OpenAI only**: Use `gpt-4o` for LLM and `text-embedding-3-small` for embeddings

**💡 Pro Tip - Model Selection Strategy**: Smaller, faster models (like `gemini-2.5-flash`) tend to be quick but less sophisticated, while larger models (like `gemini-2.5-pro` or `gpt-4o`) are slower but produce higher-quality solutions. ShinkaEvolve's strength is that you can use **both strategically**:
- **Early generations (Exploration)**: Use smaller/faster models to rapidly explore the solution space and generate diverse candidates
- **Later generations (Exploitation)**: Switch to larger/smarter models to refine and optimize the best solutions

You can configure this in `run_evo.py` by adjusting `llm_models` list or implementing custom logic to change models based on generation number.

## How to Run

### Step 1: Navigate to Example Directory

```bash
cd examples/aec_packing
```

### Step 2: (Optional) Test Initial Solution

```bash
python evaluate.py --program_path initial.py --results_dir test_results
```

This will evaluate the initial circle packing solution to ensure everything is set up correctly.

### Step 3: Run Evolution

```bash
python run_evo.py
```

This will:
- Initialize the evolution with the program in `initial.py`
- Run for 400 generations (configurable in `run_evo.py`)
- Use 2 evolutionary islands with population migration
- Maintain an archive of top 40 solutions
- Save results to `results_cpack/`

### Step 4: Monitor Progress

The evolution will print progress to the console:
```
Generation 0: Best score: 2.345
Generation 1: Best score: 2.567
...
```

Results are continuously saved to:
- `results_cpack/evolution_db.sqlite` - Database with all evolved programs
- `results_cpack/logs/` - Detailed logs
- `results_cpack/programs/` - Generated program files

## Configuration

Edit `run_evo.py` to customize the evolution:

### Evolution Strategy

Choose from multiple parent selection strategies:
```python
strategy = "weighted"  # Options: uniform, hill_climbing, weighted, power_law, beam_search
```

### Evolution Parameters

```python
evo_config = EvolutionConfig(
    num_generations=400,        # Number of evolution iterations
    max_parallel_jobs=5,        # Parallel evaluation workers
    patch_types=["diff", "full", "cross"],  # Mutation types
    patch_type_probs=[0.6, 0.3, 0.1],      # Probability distribution
    ...
)
```

### Database/Islands Configuration

```python
db_config = DatabaseConfig(
    num_islands=2,              # Number of evolutionary islands
    archive_size=40,            # Size of elite archive
    migration_interval=10,      # Generations between migrations
    migration_rate=0.1,         # Probability of migration
    ...
)
```

### LLM Configuration

```python
llm_models=[
    "gemini-2.5-flash",         # Primary model (fast and cheap)
    # "gemini-2.5-pro",         # More capable (slower, more expensive)
    # "gpt-4o",                 # OpenAI alternative
],
llm_kwargs=dict(
    temperatures=[0.0, 0.5, 1.0],              # Diversity control
    reasoning_efforts=["auto", "low", "medium", "high"],  # For reasoning models
    max_tokens=32768,
),
```

### Embedding Model

For code similarity detection:
```python
embedding_model="text-embedding-3-small",  # OpenAI (recommended)
# embedding_model="gemini-embedding-001",  # Gemini alternative
```

## Understanding the Code

### `initial.py` - Starting Solution

Contains the initial circle packing implementation with:
- `construct_packing()`: Creates the initial layout of 26 circles
- `compute_max_radii()`: Calculates maximum valid radii without overlaps
- `run_packing()`: Entry point called by the evaluator

**Evolution Target**: Only the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` is evolved.

### `evaluate.py` - Evaluation Logic

Defines how solutions are validated and scored:
- `adapted_validate_packing()`: Checks constraints (no overlaps, within bounds)
- `aggregate_circle_packing_metrics()`: Computes final score (sum of radii)
- `main()`: Entry point for evaluation

### `run_evo.py` - Evolution Runner

Configures and launches the evolution:
- System message defining the task
- Evolution parameters (generations, population, mutations)
- Database configuration (islands, archives)
- LLM models and parameters

## Expected Results

The evolution process should:
1. Start with the initial solution from `initial.py`
2. Generate variations using LLM-powered mutations
3. Evaluate each variant using `evaluate.py`
4. Select successful variants for further evolution
5. Progressively improve the sum of radii over generations

**Best Solution**: Saved in `results_cpack/evolution_db.sqlite` and accessible via the database

## Visualization

To visualize results, you can use the Shinka visualization tools:

```bash
# From ShinkaEvolve root directory
python -m shinka.plots.plot_improvement results_cpack/evolution_db.sqlite
python -m shinka.plots.plot_lineage_tree results_cpack/evolution_db.sqlite
```

Or explore using the Web UI (see `docs/webui.md`).

## Troubleshooting

### API Key Issues
```
Error: OPENAI_API_KEY not found
```
**Solution**: Ensure `.env` file is in the ShinkaEvolve root directory with valid API keys.

### Embedding Model Error
```
ValueError: Invalid embedding model: gemini-embedding-exp-03-07
```
**Solution**: The experimental model expired. Change to stable model in `run_evo.py`:
```python
embedding_model="text-embedding-3-small",  # Use this instead
```

### Out of Memory
If you encounter memory issues with large populations:
- Reduce `num_islands` (e.g., from 2 to 1)
- Reduce `archive_size` (e.g., from 40 to 20)
- Reduce `max_parallel_jobs` (e.g., from 5 to 2)

### Slow Evolution
To speed up evolution:
- Use faster models: `gemini-2.5-flash` instead of `gemini-2.5-pro`
- Reduce temperature variety: `temperatures=[0.5]`
- Increase `max_parallel_jobs` if you have spare CPU cores

## Advanced Usage

### Custom Evaluation Metrics

Edit `evaluate.py` to add custom metrics in `aggregate_circle_packing_metrics()`:

```python
public_metrics = {
    "centers_str": format_centers_string(centers),
    "num_circles": centers.shape[0],
    "custom_metric": your_calculation_here,
}
```

### Multi-Objective Optimization

Modify the `combined_score` calculation to balance multiple objectives:

```python
combined_score = alpha * sum_radii - beta * total_perimeter
```

### Constraint Relaxation

Adjust tolerance in validation:
```python
atol = 1e-6  # Increase for looser constraints
```

## References

- **ShinkaEvolve Documentation**: See `docs/` directory in root
- **Configuration Guide**: `docs/configuration.md`
- **Getting Started**: `docs/getting_started.md`
- **Circle Packing Example**: `examples/circle_packing/` (similar problem)

## Citation

If you use this example in your research, please cite the ShinkaEvolve framework:

```bibtex
@software{shinka2024,
  title={ShinkaEvolve: LLM-Driven Evolutionary Algorithms for Scientific Discovery},
  author={[Author names]},
  year={2024},
  url={[Repository URL]}
}
```

## License

This example is part of the ShinkaEvolve project. See `LICENSE` file in the root directory.
