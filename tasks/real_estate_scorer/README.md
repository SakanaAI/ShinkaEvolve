# Real Estate Scorer

ShinkaEvolve benchmark task: evolve a Python scoring function that ranks Buenos Aires apartment listings by investment quality.

## Task Description

Given a synthetic dataset of 50 Buenos Aires apartment listings, the goal is to evolve a `score_listing(listing) -> float` function that produces investment scores correlated with the ground-truth price-per-square-meter ranking.

### Features

Each listing dict contains:

| Feature | Type | Range | Description |
|---|---|---|---|
| `price_usd` | float | 10k+ | Apartment price in USD |
| `size_m2` | float | 25–200 | Size in square meters |
| `neighborhood` | str | 7 values | Buenos Aires neighborhood |
| `location_score` | int | 1–5 | Neighborhood desirability |
| `amenity_count` | int | 0–10 | Number of amenities |
| `floor` | int | 0–24 | Floor number |
| `age_years` | int | 0–79 | Building age |

### Fitness Metric

**Spearman rank correlation** between evolved scores and ground-truth `price_per_m2` rankings on a held-out test set of 10 listings. Range: [-1, 1], higher is better.

### Data Split

- Train: 40 listings (`listings_train.json`)
- Test: 10 listings (`listings_test.json`)
- Generated with `numpy.random.default_rng(seed=42)`

## Ingredients

| File | Purpose |
|---|---|
| `data.py` | Synthetic dataset generator |
| `initial.py` | Seed solution with `EVOLVE-BLOCK`; exposes `run_scoring()` |
| `evaluate.py` | Validator + scorer; runs `run_scoring`, computes Spearman corr, writes metrics |
| `baseline.py` | Standalone baseline evaluation script |
| `run_evo.py` | Async evolution runner |
| `shinka.yaml` | Run config (10 generations, small profile) |

## Execution

### Generate dataset

```bash
python -m tasks.real_estate_scorer.data
```

### Run baseline

```bash
python -m tasks.real_estate_scorer.baseline
```

### Single-program evaluation (no evolution)

```bash
cd tasks/real_estate_scorer
python evaluate.py --program_path initial.py --results_dir results/manual_eval
```

### Run ShinkaEvolve (10 generations)

```bash
cd tasks/real_estate_scorer
python run_evo.py --config_path shinka.yaml
```

Requires LLM API keys (OpenAI, Gemini, or Anthropic) configured in environment.

## Results

### Baseline vs ShinkaEvolve

| Method | Spearman Correlation | Generations |
|---|---|---|
| Baseline (linear weighted sum) | 0.9030 | — |
| ShinkaEvolve (best evolved) | *pending run* | 10 |

### Convergence Table

| Generation | Best Fitness (Spearman) |
|---|---|
| 0 (initial) | 0.9030 |
| 1 | *pending* |
| 2 | *pending* |
| 3 | *pending* |
| 4 | *pending* |
| 5 | *pending* |
| 6 | *pending* |
| 7 | *pending* |
| 8 | *pending* |
| 9 | *pending* |
| 10 | *pending* |

> Evolution results require LLM API keys — run `python run_evo.py` to populate.

### Baseline Details

The baseline uses a simple linear weighted sum:

```
score = 0.5 * (price_per_m2 / 5000) + 0.3 * (location_score / 5) + 0.2 * (amenity_count / 10)
```

This achieves Spearman correlation of **0.9030** (p=0.000344) on the 10-listing test set. The strong baseline is expected since price_per_m2 is both the dominant scoring signal and the ground truth, but evolution should discover better feature interactions (e.g., floor, age, neighborhood-specific weights).
