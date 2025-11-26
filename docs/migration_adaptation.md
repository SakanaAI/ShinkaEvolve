# Migration Adaptation

The migration adaptation option lets each island tune its migration schedule
and policy online. When `database.migration_adaptation.enabled=true`, the
evolution runner instantiates an adaptive controller that observes island
improvements and diversity after each generation and adjusts:

- `migration_rate` – fraction of the island population exported during a
  migration event.
- `migration_interval` – generations between migrations for the island.
- `island_elitism` – fraction of the island protected from migration.

## Methods

Three complementary signals can be combined by listing them in
`migration_adaptation.methods`:

1. **success** – tracks the relative improvement caused by recent migrations.
   When improvement exceeds `target_improvement`, the controller increases the
   migration rate, shortens intervals, and slightly raises the elitism ration.
2. **diversity** – computes a lightweight diversity score (score standard
   deviation of recent programs). Falling below `low_thresh` triggers more
   exploration; exceeding `high_thresh` stretches the interval and lowers the
   rate.
3. **bandit** – chooses migration policies (donor routing, payload selection,
   and migration size) using UCB1 or epsilon-greedy bandits. Rewards are the
   normalized improvements measured by the success tracker.

Each method has dedicated configuration blocks plus global `bounds` and
`weights` sections. Bounds clamp the adaptive parameters, while weights scale
how aggressively each method can move them.

## Logging

Adaptive decisions are recorded per generation in
`<results_dir>/migration_adaptation.csv` with columns for rate, interval,
elitism, EMA improvement, diversity, and the last bandit arm. These logs make
it easy to visualize how migration policies evolved across islands.

## Example configuration

See `configs/database/island_adaptive.yaml` for a complete Hydra configuration
that enables all three methods with reasonable defaults.
