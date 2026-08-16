"""Optional Weights & Biases logging for evolution runs."""

from __future__ import annotations

import importlib
import logging
import math
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from shinka.database import Program, ProgramDatabase
from shinka.telemetry_safety import is_sensitive_telemetry_key
from shinka.wandb_metrics import (
    EVALUATED_COUNT_METRIC,
    GENERATION_METRIC,
    INDIVIDUAL_SCORE_METRIC,
    MAX_METRIC_KEY_LENGTH,
    MAX_PUBLIC_METRICS,
    PROGRAM_TABLE_COLUMNS,
    PROGRAM_TABLE_KEY,
    build_population_progress_payload,
    build_population_progress_payload_from_telemetry,
    build_program_log_payload,
    build_run_summary,
    flatten_numeric_metrics,
    is_island_copy,
    program_costs,
    program_table_row,
)

__all__ = [
    "EVALUATED_COUNT_METRIC",
    "GENERATION_METRIC",
    "INDIVIDUAL_SCORE_METRIC",
    "MAX_METRIC_KEY_LENGTH",
    "MAX_PUBLIC_METRICS",
    "PROGRAM_TABLE_COLUMNS",
    "PROGRAM_TABLE_KEY",
    "ShinkaWandbLogger",
    "build_population_progress_payload",
    "build_population_progress_payload_from_telemetry",
    "build_program_log_payload",
    "build_run_summary",
    "ensure_wandb_run_id",
    "flatten_numeric_metrics",
    "is_island_copy",
    "program_costs",
    "program_table_row",
]

logger = logging.getLogger(__name__)

WANDB_RUN_ID_FILENAME = ".wandb_run_id"
_REDACTED_VALUE = "<redacted>"
_EVOLUTION_CONFIG_FIELDS = (
    "num_generations",
    "max_patch_resamples",
    "max_patch_attempts",
    "job_type",
    "language",
    "meta_rec_interval",
    "meta_max_recommendations",
    "sample_single_meta_rec",
    "max_novelty_attempts",
    "code_embed_sim_threshold",
    "use_text_feedback",
    "max_api_costs",
    "enable_controlled_oversubscription",
    "proposal_target_mode",
    "proposal_target_min_samples",
    "proposal_target_ratio_cap",
    "proposal_buffer_max",
    "proposal_target_hard_cap",
    "proposal_target_ewma_alpha",
    "evolve_prompts",
    "prompt_evolution_interval",
    "prompt_archive_size",
    "prompt_ucb_exploration_constant",
    "prompt_epsilon",
    "prompt_evo_top_k_programs",
    "prompt_percentile_recompute_interval",
)
_DATABASE_CONFIG_FIELDS = (
    "num_islands",
    "archive_size",
    "max_stdout_log_chars",
    "elite_selection_ratio",
    "num_archive_inspirations",
    "num_top_k_inspirations",
    "migration_interval",
    "migration_rate",
    "island_elitism",
    "enforce_island_separation",
    "island_selection_strategy",
    "enable_dynamic_islands",
    "stagnation_threshold",
    "island_spawn_strategy",
    "island_spawn_subtree_size",
    "parent_selection_strategy",
    "exploitation_alpha",
    "exploitation_ratio",
    "parent_selection_lambda",
    "num_beams",
    "archive_selection_strategy",
)
_JOB_CONFIG_FIELDS = (
    "eval_verbose",
    "numeric_threads_per_job",
    "partition",
    "time",
    "cpus",
    "gpus",
    "mem",
)


def ensure_wandb_run_id(
    results_dir: Path,
    configured_id: Optional[str] = None,
) -> str:
    """Persist and return the W&B run ID associated with a results directory."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = results_dir / WANDB_RUN_ID_FILENAME
    run_id = str(configured_id).strip() if configured_id else ""

    if not run_id and run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        run_id = uuid.uuid4().hex

    run_id_path.write_text(f"{run_id}\n", encoding="utf-8")
    return run_id


class ShinkaWandbLogger:
    """Best-effort W&B sink that leaves database/WebUI logging unchanged."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._wandb: Optional[Any] = None
        self._run: Optional[Any] = None
        self._logged_program_ids: set[str] = set()
        self._last_population_evaluated_count: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.enabled and self._run is not None

    def start(
        self,
        *,
        evo_config: Any,
        db_config: Any,
        job_config: Any,
        results_dir: Path,
    ) -> None:
        if not self.enabled:
            return

        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError:
            logger.warning(
                "W&B logging is enabled, but wandb is not installed. "
                "Install shinka-evolve[wandb] to use it."
            )
            self.enabled = False
            return
        except Exception as exc:
            logger.warning("Failed to import W&B: %s", exc)
            self.enabled = False
            return

        try:
            extra_config = _json_safe(getattr(evo_config, "wandb_config", {}) or {})
            if not isinstance(extra_config, dict):
                raise ValueError("wandb_config must be a dictionary")

            run_id = ensure_wandb_run_id(
                results_dir,
                getattr(evo_config, "wandb_run_id", None),
            )
            evo_config.wandb_run_id = run_id
            config = {
                "evolution": _safe_config_fields(evo_config, _EVOLUTION_CONFIG_FIELDS),
                "database": _safe_config_fields(db_config, _DATABASE_CONFIG_FIELDS),
                "job": _safe_config_fields(job_config, _JOB_CONFIG_FIELDS),
            }
            config.update(extra_config)
            settings = self._wandb.Settings(
                console="off",
                disable_git=True,
                disable_code=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
                x_save_requirements=False,
            )
            init_kwargs = {
                "project": getattr(evo_config, "wandb_project", None)
                or "shinka-evolve",
                "entity": getattr(evo_config, "wandb_entity", None),
                "group": getattr(evo_config, "wandb_group", None),
                "name": getattr(evo_config, "wandb_name", None)
                or Path(results_dir).name,
                "mode": getattr(evo_config, "wandb_mode", None),
                "tags": getattr(evo_config, "wandb_tags", None) or None,
                "notes": getattr(evo_config, "wandb_notes", None),
                "dir": getattr(evo_config, "wandb_dir", None) or str(results_dir),
                "id": run_id,
                "resume": getattr(evo_config, "wandb_resume", "allow"),
                "save_code": False,
                "settings": settings,
                "config": config,
            }
            self._run = self._wandb.init(
                **{
                    key: value
                    for key, value in init_kwargs.items()
                    if value is not None
                }
            )
            self._define_metrics()
            logger.info("W&B logging initialized for '%s'", init_kwargs["project"])
        except Exception as exc:
            logger.warning("Failed to initialize W&B logging: %s", exc)
            self.finish()
            self.enabled = False

    def log_program(self, program: Program) -> None:
        if (
            not self.active
            or is_island_copy(program)
            or program.id in self._logged_program_ids
        ):
            return
        self.log_program_payload(program.id, build_program_log_payload(program))

    def log_program_payload(self, program_id: str, payload: dict[str, Any]) -> None:
        """Log one prebuilt compact candidate payload."""
        run = self._run
        if not self.enabled or run is None or program_id in self._logged_program_ids:
            return
        try:
            run.log(payload)
            self._logged_program_ids.add(program_id)
        except Exception as exc:
            logger.warning("Failed to log individual %s to W&B: %s", program_id, exc)

    def log_population_progress(
        self,
        *,
        db: Optional[ProgramDatabase],
    ) -> None:
        """Log a snapshot only when the evaluated population advances."""
        if not self.enabled or self._run is None or db is None:
            return
        try:
            snapshot = db.get_population_telemetry()
        except Exception as exc:
            logger.warning("Failed to query W&B population progress: %s", exc)
            return
        self.log_population_snapshot(snapshot)

    def log_population_snapshot(self, snapshot: list[dict[str, Any]]) -> None:
        """Log a precomputed snapshot on the W&B telemetry worker."""
        run = self._run
        if not self.enabled or run is None:
            return
        try:
            payload = build_population_progress_payload_from_telemetry(snapshot)
            evaluated_count = payload[EVALUATED_COUNT_METRIC]
            if (
                self._last_population_evaluated_count is not None
                and evaluated_count <= self._last_population_evaluated_count
            ):
                return
            run.log(payload)
            self._last_population_evaluated_count = evaluated_count
        except Exception as exc:
            logger.warning("Failed to log W&B population progress: %s", exc)

    def log_final(
        self,
        *,
        db: Optional[ProgramDatabase],
        total_proposals_generated: Optional[int] = None,
        total_api_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
    ) -> None:
        if not self.enabled or self._run is None or self._wandb is None or db is None:
            return
        try:
            programs = db.get_all_programs()
        except Exception as exc:
            logger.warning("Failed to query final W&B programs: %s", exc)
            return
        self.log_final_programs(
            programs,
            total_proposals_generated=total_proposals_generated,
            total_api_cost=total_api_cost,
            total_cost=total_cost,
        )

    def log_final_programs(
        self,
        programs: list[Program],
        *,
        total_proposals_generated: Optional[int] = None,
        total_api_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
    ) -> None:
        """Log final metrics and table from owner-thread materialized programs."""
        run = self._run
        wandb = self._wandb
        if not self.enabled or run is None or wandb is None:
            return
        try:
            final_summary = build_run_summary(
                programs,
                total_proposals_generated=total_proposals_generated,
                total_api_cost=total_api_cost,
                total_cost=total_cost,
            )
            final_payload = build_population_progress_payload(programs)
            final_payload.update(final_summary)
        except Exception as exc:
            logger.warning("Failed to prepare final W&B metrics: %s", exc)
            return

        try:
            run.log(final_payload)
        except Exception as exc:
            logger.warning("Failed to log final W&B history: %s", exc)
        self._last_population_evaluated_count = final_payload[EVALUATED_COUNT_METRIC]

        try:
            run.summary.update(final_summary)
        except Exception as exc:
            logger.warning("Failed to update final W&B summary: %s", exc)

        try:
            table = wandb.Table(
                columns=PROGRAM_TABLE_COLUMNS,
                data=[program_table_row(program) for program in programs],
            )
        except Exception as exc:
            logger.warning("Failed to construct final W&B table: %s", exc)
            return

        try:
            run.log({PROGRAM_TABLE_KEY: table})
        except Exception as exc:
            logger.warning("Failed to log final W&B table: %s", exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning("Failed to finish W&B run cleanly: %s", exc)
        finally:
            self._run = None

    def _define_metrics(self) -> None:
        run = self._run
        if run is None or not hasattr(run, "define_metric"):
            return
        run.define_metric(GENERATION_METRIC)
        run.define_metric(EVALUATED_COUNT_METRIC)
        run.define_metric(INDIVIDUAL_SCORE_METRIC)
        run.define_metric("run/*")
        for metric_glob in (
            "population/count",
            "population/correct_count",
            "population/correct_rate",
            "population/best_score",
            "population/best_correct_score",
            "population/mean_score",
            "island/*",
            "cost/*",
            "timing/*",
        ):
            run.define_metric(
                metric_glob,
                step_metric=EVALUATED_COUNT_METRIC,
            )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: (
                _REDACTED_VALUE
                if _is_sensitive_config_key(field.name)
                else _json_safe(getattr(value, field.name))
            )
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(key): (
                _REDACTED_VALUE if _is_sensitive_config_key(key) else _json_safe(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return f"<{type(value).__name__}>"


def _safe_config_fields(value: Any, allowed_fields: tuple[str, ...]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    missing = object()
    for field_name in allowed_fields:
        if isinstance(value, dict):
            field_value = value.get(field_name, missing)
        else:
            field_value = getattr(value, field_name, missing)
        if field_value is missing:
            continue
        if field_value is None or isinstance(field_value, (bool, int, float, str)):
            safe[field_name] = _json_safe(field_value)
    return safe


def _is_sensitive_config_key(value: Any) -> bool:
    return is_sensitive_telemetry_key(value)
