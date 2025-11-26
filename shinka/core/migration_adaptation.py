"""Adaptive migration controller for Shinka evolution runs."""

from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from shinka.database.dbase import DatabaseConfig, MigrationAdaptationConfig
from shinka.database.islands import IslandMigrationParams, MigrationSummary

if TYPE_CHECKING:  # pragma: no cover - only for static checks
    from shinka.database.dbase import ProgramDatabase


@dataclass
class PendingSuccessEval:
    pre_best: float
    evaluate_after: int
    migration_generation: int
    arm_key: Optional[str] = None


@dataclass
class BanditArmState:
    count: int = 0
    total_reward: float = 0.0

    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_reward / self.count


@dataclass
class IslandAdaptationState:
    migration_rate: float
    migration_interval: int
    island_elitism: float
    impr_ema: float = 0.0
    diversity: float = 0.0
    pending_success: Optional[PendingSuccessEval] = None
    last_policy_key: Optional[str] = None


class MigrationAdaptationController:
    """Runtime controller that adapts migration parameters per island."""

    def __init__(
        self,
        db: "ProgramDatabase",
        config: DatabaseConfig,
        results_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db = db
        self.config = config
        self.adapt_config: Optional[MigrationAdaptationConfig] = (
            config.migration_adaptation
        )
        self.logger = logger or logging.getLogger(__name__)
        self.results_dir = Path(results_dir)

        self.enabled = bool(self.adapt_config and self.adapt_config.enabled)
        if not self.enabled:
            return

        self.methods = {
            method.lower() for method in (self.adapt_config.methods or [])
        }
        self.weights = self.adapt_config.weights
        self.bounds = self.adapt_config.bounds
        self.num_islands = max(0, config.num_islands)

        self.states: Dict[int, IslandAdaptationState] = {}
        self._initialize_states()

        self.bandit_enabled = "bandit" in self.methods
        self.bandit_algo = (self.adapt_config.bandit.algo or "ucb1").lower()
        self.bandit_ucb_c = self.adapt_config.bandit.ucb_c
        self.bandit_epsilon = self.adapt_config.bandit.epsilon
        self.bandit_arms: List[Tuple[str, Dict[str, str]]] = []
        self.bandit_stats: Dict[int, Dict[str, BanditArmState]] = {}
        if self.bandit_enabled:
            self._init_bandit_state()

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "migration_adaptation.csv"
        self._log_header_written = False

        self._register_with_islands()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def on_generation_completed(self, generation: int) -> None:
        if not self.enabled:
            return

        if "success" in self.methods:
            self._evaluate_success_metrics(generation)

        if "diversity" in self.methods:
            self._update_diversity_metrics()

        self._log_state(generation)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _register_with_islands(self) -> None:
        manager = getattr(self.db, "island_manager", None)
        if manager is None:
            self.logger.warning(
                "Migration adaptation enabled but no island manager is available."
            )
            self.enabled = False
            return

        params = {
            idx: IslandMigrationParams(
                migration_rate=state.migration_rate,
                migration_interval=state.migration_interval,
                island_elitism=state.island_elitism,
            )
            for idx, state in self.states.items()
        }
        manager.set_island_params_bulk(params)
        manager.set_migration_callback(self._on_migration_summary)
        if self.bandit_enabled:
            manager.register_policy_provider(self._policy_provider)

    def _initialize_states(self) -> None:
        base_rate = self._clamp_value(
            self.config.migration_rate, self.bounds.rate_min, self.bounds.rate_max
        )
        base_interval = max(2, self.config.migration_interval)
        base_elitism = self._normalize_elitism(self.config.island_elitism)

        for idx in range(self.num_islands):
            self.states[idx] = IslandAdaptationState(
                migration_rate=base_rate,
                migration_interval=base_interval,
                island_elitism=base_elitism,
            )

    def _init_bandit_state(self) -> None:
        arms_cfg = self.adapt_config.bandit.policy_arms
        donors = arms_cfg.donor or ["random"]
        payloads = arms_cfg.payload or ["random"]
        sizes = arms_cfg.size or ["medium"]
        for donor in donors:
            for payload in payloads:
                for size in sizes:
                    key = f"{donor}|{payload}|{size}"
                    policy = {"donor": donor, "payload": payload, "size": size}
                    self.bandit_arms.append((key, policy))

        for idx in range(self.num_islands):
            self.bandit_stats[idx] = {
                key: BanditArmState() for key, _ in self.bandit_arms
            }

    # ------------------------------------------------------------------
    # Success-based updates
    # ------------------------------------------------------------------
    def _on_migration_summary(self, summary: MigrationSummary) -> None:
        if not self.enabled:
            return
        for island_idx in summary.per_island.keys():
            state = self.states.get(island_idx)
            if state is None:
                continue
            pre_best = self._get_island_best_score(island_idx)
            state.pending_success = PendingSuccessEval(
                pre_best=pre_best,
                evaluate_after=summary.generation + self.adapt_config.success.window,
                migration_generation=summary.generation,
                arm_key=state.last_policy_key,
            )

    def _evaluate_success_metrics(self, generation: int) -> None:
        beta = self.adapt_config.success.ema_beta
        for island_idx, state in self.states.items():
            pending = state.pending_success
            if pending is None:
                continue
            if generation < pending.evaluate_after:
                continue

            post_best = self._get_island_best_score(island_idx)
            delta = self._relative_improvement(pending.pre_best, post_best)
            state.impr_ema = beta * state.impr_ema + (1 - beta) * delta
            self._apply_success_update(island_idx, state.impr_ema)

            if self.bandit_enabled and pending.arm_key:
                self._update_bandit_reward(island_idx, pending.arm_key, delta)

            state.pending_success = None

    def _apply_success_update(self, island_idx: int, improvement: float) -> None:
        state = self.states[island_idx]
        cfg = self.adapt_config.success
        weight = max(0.0, self.weights.success)

        if improvement >= cfg.target_improvement:
            rate_factor = cfg.step_up ** weight
            interval_factor = cfg.step_down ** weight
            elitism_delta = 0.02 * weight
        else:
            rate_factor = cfg.step_down ** weight
            interval_factor = cfg.step_up ** weight
            elitism_delta = -0.02 * weight

        new_rate = state.migration_rate * rate_factor
        new_interval = int(round(state.migration_interval * interval_factor))
        new_elitism = state.island_elitism + elitism_delta

        self._update_island_params(
            island_idx,
            migration_rate=new_rate,
            migration_interval=new_interval,
            island_elitism=new_elitism,
        )

    # ------------------------------------------------------------------
    # Diversity-based updates
    # ------------------------------------------------------------------
    def _update_diversity_metrics(self) -> None:
        cfg = self.adapt_config.diversity
        for island_idx, state in self.states.items():
            diversity = self._compute_diversity(island_idx)
            state.diversity = diversity
            strength = cfg.adjust_strength * max(0.0, self.weights.diversity)
            if diversity < cfg.low_thresh:
                new_rate = state.migration_rate * (1 + strength)
                new_interval = int(round(state.migration_interval * (1 - strength)))
                self._update_island_params(
                    island_idx,
                    migration_rate=new_rate,
                    migration_interval=max(2, new_interval),
                )
            elif diversity > cfg.high_thresh:
                new_rate = state.migration_rate * (1 - strength)
                new_interval = int(round(state.migration_interval * (1 + strength)))
                self._update_island_params(
                    island_idx,
                    migration_rate=new_rate,
                    migration_interval=new_interval,
                )

    # ------------------------------------------------------------------
    # Bandit policy selection
    # ------------------------------------------------------------------
    def _policy_provider(self, island_idx: int) -> Optional[Dict[str, str]]:
        if not self.bandit_enabled or not self.bandit_arms:
            return None
        policy, key = self._select_bandit_policy(island_idx)
        self.states[island_idx].last_policy_key = key
        return policy

    def _select_bandit_policy(self, island_idx: int) -> Tuple[Dict[str, str], str]:
        stats = self.bandit_stats[island_idx]
        if self.bandit_algo == "epsilon_greedy":
            if random.random() < self.bandit_epsilon:
                key, policy = random.choice(self.bandit_arms)
                return dict(policy), key
            key = max(self.bandit_arms, key=lambda item: stats[item[0]].average())[0]
            return dict(self._policy_from_key(key)), key

        total_plays = sum(state.count for state in stats.values()) or 1

        def ucb_score(arm_key: str) -> float:
            state = stats[arm_key]
            if state.count == 0:
                return float("inf")
            exploration = self.bandit_ucb_c * math.sqrt(
                math.log(total_plays + 1) / (state.count + 1e-9)
            )
            return state.average() + exploration

        key = max(self.bandit_arms, key=lambda item: ucb_score(item[0]))[0]
        return dict(self._policy_from_key(key)), key

    def _policy_from_key(self, key: str) -> Dict[str, str]:
        for arm_key, policy in self.bandit_arms:
            if arm_key == key:
                return dict(policy)
        # Fallback to default random policy
        return {"donor": "random", "payload": "random", "size": "medium"}

    def _update_bandit_reward(self, island_idx: int, arm_key: str, reward: float) -> None:
        stats = self.bandit_stats.get(island_idx)
        if not stats or arm_key not in stats:
            return
        stats[arm_key].count += 1
        stats[arm_key].total_reward += reward

    # ------------------------------------------------------------------
    # Parameter persistence
    # ------------------------------------------------------------------
    def _update_island_params(
        self,
        island_idx: int,
        *,
        migration_rate: Optional[float] = None,
        migration_interval: Optional[int] = None,
        island_elitism: Optional[float] = None,
    ) -> None:
        manager = getattr(self.db, "island_manager", None)
        if manager is None:
            return
        state = self.states[island_idx]

        if migration_rate is not None:
            migration_rate = self._limit_change(
                state.migration_rate,
                self._clamp_value(
                    migration_rate, self.bounds.rate_min, self.bounds.rate_max
                ),
            )
            state.migration_rate = migration_rate

        if migration_interval is not None:
            migration_interval = max(2, migration_interval)
            migration_interval = int(
                round(
                    self._limit_change(
                        float(state.migration_interval),
                        float(
                            self._clamp_value(
                                migration_interval,
                                self.bounds.interval_min,
                                self.bounds.interval_max,
                            )
                        ),
                    )
                )
            )
            state.migration_interval = migration_interval

        if island_elitism is not None:
            clamped = self._clamp_value(
                island_elitism, self.bounds.elitism_min, self.bounds.elitism_max
            )
            state.island_elitism = self._limit_change(state.island_elitism, clamped)

        manager.set_island_params(
            island_idx,
            migration_rate=state.migration_rate,
            migration_interval=state.migration_interval,
            island_elitism=state.island_elitism,
        )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _get_island_best_score(self, island_idx: int) -> float:
        cursor = self.db.cursor
        cursor.execute(
            "SELECT MAX(combined_score) as best FROM programs WHERE island_idx = ?",
            (island_idx,),
        )
        row = cursor.fetchone()
        return float(row["best"]) if row and row["best"] is not None else 0.0

    def _compute_diversity(self, island_idx: int, limit: int = 20) -> float:
        cursor = self.db.cursor
        cursor.execute(
            "SELECT combined_score FROM programs WHERE island_idx = ? "
            "ORDER BY generation DESC LIMIT ?",
            (island_idx, limit),
        )
        scores = [row["combined_score"] or 0.0 for row in cursor.fetchall()]
        if len(scores) <= 1:
            return 0.0
        std_dev = statistics.pstdev(scores)
        scale = max(max(abs(score) for score in scores), 1e-6)
        return min(1.0, float(std_dev / scale))

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _log_state(self, generation: int) -> None:
        if not self.log_path:
            return
        rows = []
        for island_idx, state in self.states.items():
            rows.append(
                {
                    "generation": generation,
                    "island": island_idx,
                    "migration_rate": f"{state.migration_rate:.4f}",
                    "migration_interval": state.migration_interval,
                    "island_elitism": f"{state.island_elitism:.4f}",
                    "impr_ema": f"{state.impr_ema:.5f}",
                    "diversity": f"{state.diversity:.5f}",
                    "policy": state.last_policy_key or "",
                }
            )

        write_header = not self._log_header_written and not self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "generation",
                    "island",
                    "migration_rate",
                    "migration_interval",
                    "island_elitism",
                    "impr_ema",
                    "diversity",
                    "policy",
                ],
            )
            if write_header:
                writer.writeheader()
                self._log_header_written = True
            for row in rows:
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    def _clamp_value(self, value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    def _limit_change(self, current: float, proposed: float) -> float:
        if current == 0:
            return proposed
        max_increase = current * 1.25
        max_decrease = current * 0.75
        return max(max_decrease, min(max_increase, proposed))

    @staticmethod
    def _relative_improvement(before: float, after: float) -> float:
        denom = max(abs(before), 1e-12)
        return (after - before) / denom

    @staticmethod
    def _normalize_elitism(value: object) -> float:
        if isinstance(value, bool):
            return 0.1 if value else 0.0
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0
