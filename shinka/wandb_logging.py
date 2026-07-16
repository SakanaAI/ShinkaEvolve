"""Weights & Biases logging for Shinka evolution runs."""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from shinka.database import Program, ProgramDatabase

logger = logging.getLogger(__name__)

LOGGING_METHOD_WEBUI = "webui"
LOGGING_METHOD_WANDB = "wandb"
VALID_LOGGING_METHODS = {LOGGING_METHOD_WEBUI, LOGGING_METHOD_WANDB}

COST_KEYS = ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
TIMING_KEYS = (
    "pipeline_seconds",
    "sampling_seconds",
    "evaluation_seconds",
    "postprocess_seconds",
    "post_eval_queue_wait_seconds",
    "postprocess_apply_wait_seconds",
    "postprocess_apply_seconds",
    "pipeline_unaccounted_seconds",
    "end_to_end_with_side_effects_seconds",
    "compute_time",
)
PROGRAM_TABLE_COLUMNS = [
    "id",
    "generation",
    "timestamp",
    "language",
    "parent_id",
    "archive_inspiration_ids",
    "top_k_inspiration_ids",
    "island_idx",
    "correct",
    "combined_score",
    "public_metrics",
    "private_metrics",
    "text_feedback",
    "children_count",
    "complexity",
    "metadata",
    "in_archive",
    "system_prompt_id",
    "code",
    "code_diff",
    "embedding",
    "embedding_pca_2d",
    "embedding_pca_3d",
    "embedding_cluster_id",
    "migration_history",
]


def resolve_logging_methods(config: Any) -> set[str]:
    """Resolve logging sink settings from list and legacy-style bool toggles."""
    raw_methods = getattr(config, "logging_methods", [LOGGING_METHOD_WEBUI])
    methods = _normalize_logging_methods(raw_methods)

    enable_webui = getattr(config, "enable_webui_logging", None)
    enable_wandb = getattr(config, "enable_wandb_logging", None)
    if enable_webui is not None:
        if enable_webui:
            methods.add(LOGGING_METHOD_WEBUI)
        else:
            methods.discard(LOGGING_METHOD_WEBUI)
    if enable_wandb is not None:
        if enable_wandb:
            methods.add(LOGGING_METHOD_WANDB)
        else:
            methods.discard(LOGGING_METHOD_WANDB)

    unknown = methods - VALID_LOGGING_METHODS
    if unknown:
        valid = ", ".join(sorted(VALID_LOGGING_METHODS))
        bad = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown logging method(s): {bad}. Valid methods: {valid}."
        )
    return methods


def _normalize_logging_methods(raw_methods: Any) -> set[str]:
    if raw_methods is None:
        return set()
    if isinstance(raw_methods, str):
        raw_items = [item.strip() for item in raw_methods.split(",")]
    elif isinstance(raw_methods, Iterable):
        raw_items = list(raw_methods)
    else:
        raise ValueError("logging_methods must be a list or comma-separated string.")
    return {str(item).strip().lower() for item in raw_items if str(item).strip()}


def json_safe(value: Any, *, max_string_length: Optional[int] = None) -> Any:
    """Convert common Python/numpy/dataclass values into JSON-safe data."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value), max_string_length=max_string_length)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict(), max_string_length=max_string_length)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item(), max_string_length=max_string_length)
        except Exception:
            pass
    if isinstance(value, str):
        if max_string_length is not None and len(value) > max_string_length:
            return value[:max_string_length] + "...[truncated]"
        return value
    if isinstance(value, dict):
        return {
            str(key): json_safe(item, max_string_length=max_string_length)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item, max_string_length=max_string_length)
            for item in list(value)
        ]
    return repr(value)


def compute_database_stats(
    db: Optional[ProgramDatabase],
    prompt_db: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute aggregate run stats matching the WebUI database summary."""
    if db is None:
        return {}

    programs = db.get_all_programs()
    program_count = len(programs)
    generations = [p.generation for p in programs]
    generation_count = len(set(generations))
    max_generation = max(generations) if generations else 0
    correct_programs = [p for p in programs if bool(p.correct)]
    scored_programs = [
        p for p in programs if isinstance(p.combined_score, (int, float))
    ]
    correct_scored_programs = [
        p for p in correct_programs if isinstance(p.combined_score, (int, float))
    ]

    best_score = None
    best_generation = None
    best_program_id = None
    if correct_scored_programs:
        best_score = max(float(p.combined_score) for p in correct_scored_programs)
        best_programs = [
            p for p in correct_scored_programs if float(p.combined_score) == best_score
        ]
        best_program = min(best_programs, key=lambda p: p.generation)
        best_generation = best_program.generation
        best_program_id = best_program.id

    first_update = min((p.timestamp for p in programs), default=None)
    last_update = max((p.timestamp for p in programs), default=None)
    first_pipeline_start = _min_metadata_value(programs, "pipeline_started_at")
    last_postprocess_finish = _max_metadata_value(programs, "postprocess_finished_at")
    runtime_start = (
        first_pipeline_start if first_pipeline_start is not None else first_update
    )
    runtime_end = (
        last_postprocess_finish
        if last_postprocess_finish is not None
        else last_update
    )
    total_runtime_seconds = None
    if runtime_start is not None and runtime_end is not None:
        total_runtime_seconds = max(0.0, float(runtime_end) - float(runtime_start))

    cost_breakdown = _aggregate_costs(programs)
    scores = [float(p.combined_score) for p in scored_programs]
    correct_count = len(correct_programs)
    incorrect_count = program_count - correct_count
    score_mean = sum(scores) / len(scores) if scores else None
    correct_rate = correct_count / program_count if program_count else 0.0
    throughput = None
    if total_runtime_seconds and total_runtime_seconds > 0:
        throughput = program_count / total_runtime_seconds

    archive_count = _count_archive(db)
    island_counts = _count_by_value(
        p.island_idx for p in programs if p.island_idx is not None
    )
    patch_type_counts = _count_by_value(
        (p.metadata or {}).get("patch_type")
        for p in programs
        if isinstance(p.metadata, dict)
    )
    model_counts = _count_by_value(
        model_name
        for model_name in (_program_model_name(program) for program in programs)
        if model_name
    )

    stats = {
        "program_count": program_count,
        "generation_count": generation_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "correct_rate": correct_rate,
        "best_score": best_score,
        "best_generation": best_generation,
        "best_program_id": best_program_id,
        "max_generation": max_generation,
        "last_update": last_update,
        "gens_since_improvement": (
            max_generation - best_generation
            if best_generation is not None
            else max_generation
        ),
        "total_cost": cost_breakdown["total"],
        "api_costs": cost_breakdown["api_costs"],
        "embed_cost": cost_breakdown["embed_cost"],
        "novelty_cost": cost_breakdown["novelty_cost"],
        "meta_cost": cost_breakdown["meta_cost"],
        "total_runtime_seconds": total_runtime_seconds,
        "throughput_programs_per_second": throughput,
        "score_mean": score_mean,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "archive_count": archive_count,
        "island_counts": island_counts,
        "patch_type_counts": patch_type_counts,
        "model_counts": model_counts,
        "prompt_count": 0,
        "prompt_evo_cost": 0.0,
        "has_prompt_evo": False,
    }
    stats.update(_prompt_stats(prompt_db))
    return json_safe(stats)


def build_program_log_payload(
    program: Program,
    db: Optional[ProgramDatabase],
    prompt_db: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create the scalar payload logged for one completed program."""
    program_dict = json_safe(program.to_dict())
    metadata = program.metadata or {}
    public_metrics = program.public_metrics or {}
    private_metrics = program.private_metrics or {}
    cost_breakdown = program_cost_breakdown(program)

    payload: Dict[str, Any] = {
        "generation": program.generation,
        "program/id": program.id,
        "program/parent_id": program.parent_id,
        "program/island_idx": program.island_idx,
        "program/correct": bool(program.correct),
        "program/combined_score": program.combined_score,
        "program/complexity": program.complexity,
        "program/children_count": program.children_count,
        "program/in_archive": bool(program.in_archive),
        "program/system_prompt_id": program.system_prompt_id,
        "program/patch_type": metadata.get("patch_type"),
        "program/patch_name": metadata.get("patch_name"),
        "program/model_name": _program_model_name(program),
        "cost/api": cost_breakdown["api_costs"],
        "cost/embed": cost_breakdown["embed_cost"],
        "cost/novelty": cost_breakdown["novelty_cost"],
        "cost/meta": cost_breakdown["meta_cost"],
        "cost/total": cost_breakdown["total"],
    }

    for key in TIMING_KEYS:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[f"timing/{key}"] = float(value)

    payload.update(flatten_scalars(public_metrics, "public_metrics"))
    payload.update(flatten_scalars(private_metrics, "private_metrics"))
    payload.update(flatten_scalars(metadata, "metadata"))
    payload["program/latest_json"] = json.dumps(program_dict, default=str)
    return json_safe(payload, max_string_length=8000)


def program_cost_breakdown(program: Program) -> Dict[str, float]:
    metadata = program.metadata or {}
    costs: Dict[str, float] = {}
    for key in COST_KEYS:
        costs[key] = _coerce_float(metadata.get(key), default=0.0)
    costs["total"] = sum(costs.values())
    return costs


def flatten_scalars(
    data: Any,
    prefix: str,
    *,
    max_depth: int = 4,
) -> Dict[str, Any]:
    """Flatten scalar dict values into slash-separated W&B metric keys."""
    flattened: Dict[str, Any] = {}

    def visit(value: Any, parts: List[str], depth: int) -> None:
        if depth > max_depth:
            return
        safe_value = json_safe(value)
        if _is_scalar(safe_value):
            flattened[_metric_key(prefix, parts)] = safe_value
            return
        if isinstance(safe_value, dict):
            for key, child in safe_value.items():
                visit(child, [*parts, _metric_segment(key)], depth + 1)
            return
        if isinstance(safe_value, list) and _is_short_scalar_list(safe_value):
            flattened[_metric_key(prefix, parts)] = json.dumps(safe_value)

    if isinstance(data, dict):
        for key, value in data.items():
            visit(value, [_metric_segment(key)], 1)
    return flattened


def program_table_rows(programs: Sequence[Program]) -> List[List[Any]]:
    return [program_table_row(program) for program in programs]


def program_table_row(program: Program) -> List[Any]:
    program_dict = json_safe(program.to_dict())
    return [_table_cell(program_dict.get(column)) for column in PROGRAM_TABLE_COLUMNS]


def _table_cell(value: Any) -> Any:
    safe_value = json_safe(value, max_string_length=12000)
    if _is_scalar(safe_value):
        return safe_value
    return json.dumps(safe_value, sort_keys=True, default=str)


class ShinkaWandbLogger:
    """Optional W&B sink for Shinka run metrics and WebUI-visible data."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._wandb: Optional[Any] = None
        self._run: Optional[Any] = None
        self._logged_program_ids: set[str] = set()
        self._program_rows: List[List[Any]] = []

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
                "W&B logging was requested but the 'wandb' package is not installed."
            )
            self.enabled = False
            return

        init_kwargs = {
            "project": getattr(evo_config, "wandb_project", None) or "shinka-evolve",
            "entity": getattr(evo_config, "wandb_entity", None),
            "group": getattr(evo_config, "wandb_group", None),
            "name": getattr(evo_config, "wandb_name", None)
            or Path(results_dir).name,
            "mode": getattr(evo_config, "wandb_mode", None),
            "tags": getattr(evo_config, "wandb_tags", None) or None,
            "notes": getattr(evo_config, "wandb_notes", None),
            "dir": getattr(evo_config, "wandb_dir", None) or str(results_dir),
            "id": getattr(evo_config, "wandb_run_id", None),
            "resume": getattr(evo_config, "wandb_resume", "allow"),
            "config": {
                "evo_config": json_safe(evo_config),
                "db_config": json_safe(db_config),
                "job_config": json_safe(job_config),
                "results_dir": str(results_dir),
                **json_safe(getattr(evo_config, "wandb_config", {}) or {}),
                "run_manifest": _load_run_manifest(results_dir),
            },
            "reinit": True,
        }
        init_kwargs = {
            key: value for key, value in init_kwargs.items() if value is not None
        }

        try:
            self._run = self._wandb.init(**init_kwargs)
            if hasattr(self._run, "define_metric"):
                self._run.define_metric("generation")
                self._run.define_metric("*", step_metric="generation")
            logger.info(
                "W&B logging initialized for project '%s'",
                init_kwargs["project"],
            )
        except Exception as e:
            logger.warning("Failed to initialize W&B logging: %s", e)
            self.enabled = False
            self._run = None

    def log_program(
        self,
        *,
        program: Program,
        db: Optional[ProgramDatabase],
        prompt_db: Optional[Any] = None,
    ) -> None:
        if not self.active or program.id in self._logged_program_ids:
            return

        try:
            payload = build_program_log_payload(program, db, prompt_db)
            latest_table = self._make_table([program_table_row(program)])
            if latest_table is not None:
                payload["program/latest"] = latest_table
            self._run.log(payload)
            self._logged_program_ids.add(program.id)
            self._program_rows.append(program_table_row(program))
        except Exception as e:
            logger.warning("Failed to log program %s to W&B: %s", program.id, e)

    def log_final(
        self,
        *,
        db: Optional[ProgramDatabase],
        prompt_db: Optional[Any] = None,
        results_dir: Optional[Path] = None,
        total_proposals_generated: Optional[int] = None,
        total_api_cost: Optional[float] = None,
    ) -> None:
        if not self.active:
            return

        try:
            payload = {
                "final/total_proposals_generated": total_proposals_generated,
                "final/total_api_cost": total_api_cost,
            }
            payload.update(
                flatten_scalars(compute_database_stats(db, prompt_db), "final")
            )

            payload = json_safe(payload, max_string_length=8000)

            if db is not None:
                all_programs = db.get_all_programs()
                all_table = self._make_table(program_table_rows(all_programs))
                if all_table is not None:
                    payload["program/all"] = all_table

            self._run.log(payload)
            self._log_webui_artifact(
                db=db,
                prompt_db=prompt_db,
                results_dir=results_dir,
            )
        except Exception as e:
            logger.warning("Failed to log final W&B summary: %s", e)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as e:
            logger.warning("Failed to finish W&B run cleanly: %s", e)
        finally:
            self._run = None

    def _make_table(self, rows: Sequence[Sequence[Any]]) -> Optional[Any]:
        if self._wandb is None or not hasattr(self._wandb, "Table"):
            return None
        return self._wandb.Table(
            columns=PROGRAM_TABLE_COLUMNS,
            data=json_safe(list(rows)),
        )

    def _log_webui_artifact(
        self,
        *,
        db: Optional[ProgramDatabase],
        prompt_db: Optional[Any],
        results_dir: Optional[Path],
    ) -> None:
        if self._wandb is None or not hasattr(self._wandb, "Artifact"):
            return
        if not hasattr(self._run, "log_artifact"):
            return

        artifact_name = getattr(self._run, "name", None) or "shinka-webui-data"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(artifact_name)).strip("-")
        artifact = self._wandb.Artifact(
            name=f"{safe_name}-webui-data",
            type="shinka-webui",
        )

        added = False
        db_path = getattr(getattr(db, "config", None), "db_path", None)
        if db_path and Path(db_path).exists():
            artifact.add_file(str(db_path), name="programs.sqlite")
            added = True

        prompt_db_path = getattr(
            getattr(prompt_db, "config", None), "db_path", None
        )
        if prompt_db_path and Path(prompt_db_path).exists():
            artifact.add_file(str(prompt_db_path), name="prompts.sqlite")
            added = True

        if results_dir is not None:
            results_path = Path(results_dir)
            meta_dir = results_path / "meta"
            if meta_dir.exists():
                for meta_file in sorted(meta_dir.glob("meta_*.txt")):
                    artifact.add_file(str(meta_file), name=f"meta/{meta_file.name}")
                    added = True
            for file_path in _iter_webui_artifact_files(results_path):
                artifact.add_file(
                    str(file_path),
                    name=str(file_path.relative_to(results_path)),
                )
                added = True

        if added:
            self._run.log_artifact(artifact)


def _aggregate_costs(programs: Sequence[Program]) -> Dict[str, float]:
    totals = {key: 0.0 for key in COST_KEYS}
    for program in programs:
        costs = program_cost_breakdown(program)
        for key in COST_KEYS:
            totals[key] += costs[key]
    totals["total"] = sum(totals.values())
    return totals


def _load_run_manifest(results_dir: Path) -> Dict[str, Any]:
    path = results_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _prompt_stats(prompt_db: Optional[Any]) -> Dict[str, Any]:
    if prompt_db is None:
        return {}
    prompt_count = 0
    prompt_evo_cost = 0.0
    try:
        if hasattr(prompt_db, "_count_prompts_in_db"):
            prompt_count = prompt_db._count_prompts_in_db()
        elif hasattr(prompt_db, "get_all_prompts"):
            prompt_count = len(prompt_db.get_all_prompts())
    except Exception as e:
        logger.debug("Failed to read prompt count for W&B stats: %s", e)
    try:
        if hasattr(prompt_db, "get_total_evolution_costs"):
            prompt_evo_cost = float(prompt_db.get_total_evolution_costs())
    except Exception as e:
        logger.debug("Failed to read prompt evolution cost for W&B stats: %s", e)
    return {
        "prompt_count": prompt_count,
        "prompt_evo_cost": prompt_evo_cost,
        "has_prompt_evo": prompt_count > 0,
    }


def _min_metadata_value(programs: Sequence[Program], key: str) -> Optional[float]:
    values = [
        float(program.metadata[key])
        for program in programs
        if isinstance(program.metadata, dict)
        and isinstance(program.metadata.get(key), (int, float))
    ]
    return min(values) if values else None


def _max_metadata_value(programs: Sequence[Program], key: str) -> Optional[float]:
    values = [
        float(program.metadata[key])
        for program in programs
        if isinstance(program.metadata, dict)
        and isinstance(program.metadata.get(key), (int, float))
    ]
    return max(values) if values else None


def _count_archive(db: ProgramDatabase) -> int:
    try:
        if db.cursor:
            db.cursor.execute("SELECT COUNT(*) FROM archive")
            row = db.cursor.fetchone()
            return int((row or [0])[0])
    except Exception:
        return 0
    return 0


def _count_by_value(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _iter_webui_artifact_files(results_dir: Path) -> Iterable[Path]:
    patterns = [
        "gen_*/failure.json",
        "gen_*/results/plots/*.png",
        "gen_*/results/plots/*.gif",
        "gen_*/results/plots/*.jpg",
        "gen_*/results/plots/*.jpeg",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for file_path in sorted(results_dir.glob(pattern)):
            if file_path in seen or not file_path.is_file():
                continue
            seen.add(file_path)
            yield file_path


def _program_model_name(program: Program) -> Optional[str]:
    metadata = program.metadata or {}
    llm_result = metadata.get("llm_result") or {}
    if not isinstance(llm_result, dict):
        llm_result = {}
    value = (
        metadata.get("model_name")
        or metadata.get("model")
        or llm_result.get("model")
    )
    return str(value) if value is not None else None


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _is_short_scalar_list(value: Sequence[Any]) -> bool:
    return len(value) <= 10 and all(_is_scalar(item) for item in value)


def _metric_segment(value: Any) -> str:
    segment = str(value).strip()
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment)
    return segment.strip("_") or "value"


def _metric_key(prefix: str, parts: Sequence[str]) -> str:
    return "/".join([item for item in [prefix, *parts] if item])
