"""Crash-safe clean checkpoints for deterministic Shinka-owned RNG resume.

Checkpoint files use pickle and must only be loaded from a trusted results
directory. Pickle can execute code while loading untrusted input.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import math
import os
import pickle
import platform
import random
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shinka import __version__

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.pkl"
PREVIOUS_CHECKPOINT_FILENAME = "checkpoint.previous.pkl"
RESUME_METADATA_FILENAME = "checkpoint_resume.json"
_ENVELOPE_FORMAT = "shinka-clean-checkpoint-v1"


class CheckpointError(RuntimeError):
    """Base error for checkpoint persistence or validation failures."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when no checkpoint file exists in the results directory."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint envelope or checksum is invalid."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint is incompatible with the current run."""


class UncleanCheckpointError(CheckpointError):
    """Raised when a checkpoint does not represent a clean drain boundary."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    payload: dict[str, Any]
    path: Path


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync for atomic directory-entry publication."""
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_fsynced_temp(directory: Path, prefix: str, data: bytes) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    temp_path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _copy_fsynced(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    temp_path = _write_fsynced_temp(
        destination.parent,
        f".{destination.name}.",
        data,
    )
    try:
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    payload_bytes = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "format": _ENVELOPE_FORMAT,
        "schema_version": payload.get("schema_version"),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload": payload_bytes,
    }
    return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


def write_checkpoint(results_dir: Path | str, payload: Mapping[str, Any]) -> Path:
    """Atomically publish a checkpoint while retaining the prior generation."""
    directory = Path(results_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / CHECKPOINT_FILENAME
    previous = directory / PREVIOUS_CHECKPOINT_FILENAME
    temp_path = _write_fsynced_temp(
        directory,
        f".{CHECKPOINT_FILENAME}.",
        _checkpoint_bytes(payload),
    )
    try:
        # Copy rather than rename the current checkpoint away. If publication of
        # the replacement fails, the current checkpoint remains valid in place.
        if target.exists():
            _copy_fsynced(target, previous)
        os.replace(temp_path, target)
        _fsync_directory(directory)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def _decode_checkpoint(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"Could not load checkpoint envelope {path.name}: {exc}"
        ) from exc

    if not isinstance(envelope, dict) or envelope.get("format") != _ENVELOPE_FORMAT:
        raise CheckpointIntegrityError(f"Invalid checkpoint envelope in {path.name}")
    payload_bytes = envelope.get("payload")
    expected_digest = envelope.get("payload_sha256")
    if not isinstance(payload_bytes, bytes) or not isinstance(expected_digest, str):
        raise CheckpointIntegrityError(
            f"Checkpoint envelope fields are invalid in {path.name}"
        )
    actual_digest = hashlib.sha256(payload_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise CheckpointIntegrityError(f"Checkpoint checksum mismatch in {path.name}")

    try:
        payload = pickle.loads(payload_bytes)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"Could not decode checkpoint payload in {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError(
            f"Checkpoint payload in {path.name} must be a dictionary"
        )
    if envelope.get("schema_version") != payload.get("schema_version"):
        raise CheckpointIntegrityError(
            f"Checkpoint schema fields disagree in {path.name}"
        )
    return payload


def load_checkpoint(results_dir: Path | str) -> LoadedCheckpoint:
    """Load the newest valid fixed-name checkpoint, falling back to the prior one."""
    directory = Path(results_dir).resolve()
    candidates = [
        directory / CHECKPOINT_FILENAME,
        directory / PREVIOUS_CHECKPOINT_FILENAME,
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise CheckpointNotFoundError(
            f"No {CHECKPOINT_FILENAME} found in results directory {directory}"
        )

    failures: list[str] = []
    for path in existing:
        try:
            return LoadedCheckpoint(payload=_decode_checkpoint(path), path=path)
        except CheckpointIntegrityError as exc:
            failures.append(str(exc))
    raise CheckpointIntegrityError("; ".join(failures))


def write_resume_metadata(results_dir: Path | str, metadata: Mapping[str, Any]) -> Path:
    """Atomically publish a human-readable resume audit record."""
    directory = Path(results_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / RESUME_METADATA_FILENAME
    data = (json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n").encode()
    temp_path = _write_fsynced_temp(directory, f".{target.name}.", data)
    try:
        os.replace(temp_path, target)
        _fsync_directory(directory)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


_PROGRAM_WATERMARK_TABLES = ("archive", "metadata_store", "programs")
_PROMPT_WATERMARK_TABLES = (
    "prompt_archive",
    "prompt_metadata_store",
    "system_prompts",
)


def sqlite_tables_digest(
    connection: sqlite3.Connection,
    tables: Sequence[str],
) -> str:
    """Hash committed rows from named tables, ordered by the first column."""
    cursor = connection.cursor()
    payload: list[Any] = []
    for name in tables:
        # Order by the first column, not rowid: INSERT OR REPLACE rewrites rowids.
        cursor.execute(f'SELECT * FROM "{name}" ORDER BY 1')
        payload.append((name, [tuple(row) for row in cursor.fetchall()]))
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def relative_results_path(results_dir: Path | str, path_value: str) -> str:
    results_root = Path(results_dir).resolve()
    database_path = Path(path_value).resolve()
    try:
        return database_path.relative_to(results_root).as_posix()
    except ValueError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint database must be inside results_dir: {database_path}"
        ) from exc


def program_database_watermark(results_dir: Path | str, db: Any) -> dict[str, Any]:
    if db is None or db.cursor is None or db.conn is None:
        raise CheckpointCompatibilityError("Program database is not initialized")
    db.cursor.execute("SELECT COUNT(*) FROM programs")
    program_count = int(db.cursor.fetchone()[0])
    return {
        "path": relative_results_path(results_dir, db.config.db_path),
        "last_iteration": int(db.last_iteration),
        "program_count": program_count,
        "content_sha256": sqlite_tables_digest(db.conn, _PROGRAM_WATERMARK_TABLES),
    }


def prompt_database_watermark(
    results_dir: Path | str,
    prompt_db: Any,
) -> dict[str, Any] | None:
    if prompt_db is None:
        return None
    if prompt_db.cursor is None or prompt_db.conn is None:
        raise CheckpointCompatibilityError("Prompt database is not initialized")
    prompt_db.cursor.execute("SELECT COUNT(*) FROM system_prompts")
    prompt_count = int(prompt_db.cursor.fetchone()[0])
    return {
        "path": relative_results_path(results_dir, prompt_db.config.db_path),
        "last_generation": int(prompt_db.last_generation),
        "prompt_count": prompt_count,
        "content_sha256": sqlite_tables_digest(
            prompt_db.conn, _PROMPT_WATERMARK_TABLES
        ),
    }


def _qualified_type(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, np.generic):
        return _normalize_config_value(value.item())
    if isinstance(value, np.ndarray):
        return [_normalize_config_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_config_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if is_dataclass(value):
        return {
            field.name: _normalize_config_value(getattr(value, field.name))
            for field in fields(value)
        }
    checkpoint_config = getattr(value, "checkpoint_config", None)
    if callable(checkpoint_config):
        return _normalize_config_value(checkpoint_config())
    return {"type": _qualified_type(value)}


def _file_identity(path_value: Any) -> Any:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "sha256": None}
    return {
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _hash_mapping(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _normalize_config_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def configuration_hashes(
    evo_config: Any,
    db_config: Any,
    job_config: Any,
) -> dict[str, str]:
    """Hash normalized selection, database, and evaluator configuration."""
    evolution = _normalize_config_value(evo_config)
    database = _normalize_config_value(db_config)
    job = _normalize_config_value(job_config)
    if not isinstance(evolution, dict) or not isinstance(database, dict):
        raise TypeError("Evolution and database configurations must be structured")
    if not isinstance(job, dict):
        raise TypeError("Job configuration must be structured")

    # Location, observability, and resume policy do not alter selection state.
    evolution.pop("results_dir", None)
    evolution.pop("checkpoint_resume_mode", None)
    evolution.pop("num_generations", None)
    for key in list(evolution):
        if key.startswith("wandb_") or key == "enable_wandb_logging":
            evolution.pop(key, None)
    evolution["init_program"] = _file_identity(
        getattr(evo_config, "init_program_path", None)
    )
    evolution.pop("init_program_path", None)

    database.pop("db_path", None)
    job["eval_program"] = _file_identity(getattr(job_config, "eval_program_path", None))
    job.pop("eval_program_path", None)
    return {
        "evolution": _hash_mapping(evolution),
        "database": _hash_mapping(database),
        "job": _hash_mapping(job),
    }


def source_tree_identity(package_dir: Path | str | None = None) -> dict[str, str]:
    """Return a stable identifier for the installed Shinka Python source tree."""
    root = (
        Path(package_dir).resolve()
        if package_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return {"shinka_version": __version__, "python_source_sha256": digest.hexdigest()}


def runtime_identity(
    named_generators: Mapping[str, np.random.Generator],
) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sqlite_version": sqlite3.sqlite_version,
        "python_random_implementation": _qualified_type(random._inst),
        "numpy_legacy_implementation": _qualified_type(np.random.mtrand._rand),
        "named_bit_generators": {
            name: _qualified_type(generator.bit_generator)
            for name, generator in sorted(named_generators.items())
        },
    }


def capture_rng_states(
    named_generators: Mapping[str, np.random.Generator],
) -> dict[str, Any]:
    """Capture RNG state without drawing from any stream."""
    return {
        "python_global": copy.deepcopy(random.getstate()),
        "numpy_legacy_global": copy.deepcopy(np.random.get_state()),
        "named_generators": {
            name: {
                "bit_generator": _qualified_type(generator.bit_generator),
                "state": copy.deepcopy(generator.bit_generator.state),
            }
            for name, generator in sorted(named_generators.items())
        },
    }


def _validated_rng_states(
    state: Mapping[str, Any],
    named_generators: Mapping[str, np.random.Generator],
) -> tuple[Any, Any, Mapping[str, Any]]:
    try:
        python_state = copy.deepcopy(state["python_global"])
        numpy_state = copy.deepcopy(state["numpy_legacy_global"])
        saved_named = state["named_generators"]
    except (KeyError, TypeError) as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint RNG state is incomplete"
        ) from exc
    if not isinstance(saved_named, Mapping):
        raise CheckpointCompatibilityError("Named RNG state must be a mapping")
    if set(saved_named) != set(named_generators):
        raise CheckpointCompatibilityError(
            "Named RNG registry mismatch: "
            f"checkpoint={sorted(saved_named)} current={sorted(named_generators)}"
        )

    # Probe every state first so a malformed named state cannot leave globals
    # partially restored.
    try:
        python_probe = random.Random()
        python_probe.setstate(python_state)
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_state)
        for name, generator in named_generators.items():
            saved = saved_named[name]
            expected_type = _qualified_type(generator.bit_generator)
            if saved.get("bit_generator") != expected_type:
                raise CheckpointCompatibilityError(
                    f"BitGenerator mismatch for {name}: "
                    f"checkpoint={saved.get('bit_generator')} current={expected_type}"
                )
            probe = type(generator.bit_generator)()
            probe.state = copy.deepcopy(saved["state"])
    except CheckpointCompatibilityError:
        raise
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint RNG state is not restorable: {exc}"
        ) from exc
    return python_state, numpy_state, saved_named


def validate_rng_states(
    state: Mapping[str, Any],
    named_generators: Mapping[str, np.random.Generator],
) -> None:
    """Validate RNG compatibility without mutating any current stream."""
    _validated_rng_states(state, named_generators)


def restore_rng_states(
    state: Mapping[str, Any],
    named_generators: Mapping[str, np.random.Generator],
) -> None:
    """Validate all RNG states, then restore globals and named generators."""
    python_state, numpy_state, saved_named = _validated_rng_states(
        state,
        named_generators,
    )

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    for name, generator in named_generators.items():
        generator.bit_generator.state = copy.deepcopy(saved_named[name]["state"])


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    expected_config_hashes: Mapping[str, str],
    expected_runtime: Mapping[str, Any],
) -> None:
    """Validate schema, clean marker, source/config, and runtime identity."""
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint schema {schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if payload.get("clean") is not True:
        raise UncleanCheckpointError("Strict resume requires a clean checkpoint")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("Checkpoint identity is missing")
    if identity.get("source") != dict(expected_source):
        raise CheckpointCompatibilityError("Shinka source-tree identity changed")
    if identity.get("configuration_hashes") != dict(expected_config_hashes):
        raise CheckpointCompatibilityError("Selection-relevant configuration changed")
    if identity.get("runtime") != dict(expected_runtime):
        raise CheckpointCompatibilityError(
            "Python, NumPy, or RNG implementation version changed"
        )


class RunnerCheckpointMixin:
    """Runner-facing clean checkpoint publish, validate, and restore helpers."""

    def request_checkpoint_and_exit(self) -> bool:
        """Pause proposal admission, drain active work, checkpoint, and exit."""
        if self.checkpoint_complete.is_set():
            return False

        def request() -> None:
            if not self.checkpoint_requested.is_set():
                logger.info(
                    "Checkpoint requested; pausing new proposals and draining "
                    "active work"
                )
            self.pause_new_proposals.set()
            self.checkpoint_requested.set()
            self.slot_available.set()

        loop = self._run_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(request)
        else:
            request()
        return True

    def _owned_rng_generators(self) -> dict[str, np.random.Generator]:
        generators: dict[str, np.random.Generator] = {}
        bandit_rng = getattr(self.llm_selection, "rng", None)
        if isinstance(bandit_rng, np.random.Generator):
            generators["llm_selection"] = bandit_rng
        return generators

    def _seed_random_streams(
        self, *, force: bool = False, include_named: bool = True
    ) -> None:
        seed = self.evo_config.random_seed
        if seed is None and not force:
            return
        random.seed(seed)
        np.random.seed(seed)
        if include_named and self.llm_selection is not None:
            self.llm_selection.reseed(self._llm_selection_seed)

    def _database_watermark(self) -> dict[str, Any]:
        return {
            "programs": program_database_watermark(self.results_dir, self.db),
            "prompts": prompt_database_watermark(self.results_dir, self.prompt_db),
        }

    def _bandit_type_name(self) -> str | None:
        if self.llm_selection is None:
            return None
        return (
            f"{type(self.llm_selection).__module__}."
            f"{type(self.llm_selection).__qualname__}"
        )

    def _capture_meta_summarizer_state(self) -> dict[str, Any] | None:
        if self.meta_summarizer is None:
            return None
        sync_summarizer = getattr(
            self.meta_summarizer, "sync_summarizer", self.meta_summarizer
        )
        return {
            "unprocessed_programs": [
                program.to_dict()
                for program in sync_summarizer.evaluated_since_last_meta
            ],
            "meta_summary": sync_summarizer.meta_summary,
            "meta_scratch_pad": sync_summarizer.meta_scratch_pad,
            "meta_recommendations": sync_summarizer.meta_recommendations,
            "meta_recommendations_history": list(
                sync_summarizer.meta_recommendations_history
            ),
            "total_programs_processed": int(sync_summarizer.total_programs_processed),
        }

    def _restore_meta_summarizer_state(self, state: dict[str, Any] | None) -> None:
        from shinka.database import Program

        if state is None:
            if self.meta_summarizer is not None:
                raise CheckpointCompatibilityError(
                    "Checkpoint is missing configured meta-summarizer state"
                )
            return
        if self.meta_summarizer is None:
            raise CheckpointCompatibilityError(
                "Checkpoint has meta-summarizer state but it is not configured"
            )
        sync_summarizer = getattr(
            self.meta_summarizer, "sync_summarizer", self.meta_summarizer
        )
        sync_summarizer.evaluated_since_last_meta = [
            Program.from_dict(program)
            for program in state.get("unprocessed_programs", [])
        ]
        sync_summarizer.meta_summary = state.get("meta_summary")
        sync_summarizer.meta_scratch_pad = state.get("meta_scratch_pad")
        sync_summarizer.meta_recommendations = state.get("meta_recommendations")
        sync_summarizer.meta_recommendations_history = list(
            state.get("meta_recommendations_history", [])
        )
        sync_summarizer.total_programs_processed = int(
            state.get("total_programs_processed", 0)
        )

    def _capture_component_state(self) -> dict[str, Any]:
        return {
            "bandit_type": self._bandit_type_name(),
            "bandit_state": (
                self.llm_selection.get_state()
                if self.llm_selection is not None
                else None
            ),
            "meta_summarizer": self._capture_meta_summarizer_state(),
        }

    def _restore_component_state(self, state: dict[str, Any]) -> None:
        if state.get("bandit_type") != self._bandit_type_name():
            raise CheckpointCompatibilityError("LLM selection bandit type changed")
        bandit_state = state.get("bandit_state")
        if self.llm_selection is None:
            if bandit_state is not None:
                raise CheckpointCompatibilityError(
                    "Checkpoint unexpectedly contains bandit state"
                )
        elif not isinstance(bandit_state, dict):
            raise CheckpointCompatibilityError("Checkpoint bandit state is missing")
        else:
            self.llm_selection.set_state(bandit_state)
        self._restore_meta_summarizer_state(state.get("meta_summarizer"))

    def _capture_runner_state(self) -> dict[str, Any]:
        return {
            "completed_generations": int(self.completed_generations),
            "next_generation_to_submit": int(self.next_generation_to_submit),
            "configured_num_generations": int(self.evo_config.num_generations),
            "assigned_generations": sorted(self.assigned_generations),
            "best_program_id": self.best_program_id,
            "prompt_evolution_counter": int(self.prompt_evolution_counter),
            "prompt_percentile_recompute_counter": int(
                self.prompt_percentile_recompute_counter
            ),
            "current_prompt_id": self.current_prompt_id,
            "prompt_api_cost": float(self.prompt_api_cost),
            "committed_cost_total": float(self.total_api_cost),
            "completed_proposal_costs": list(self.completed_proposal_costs),
            "avg_proposal_cost": float(self.avg_proposal_cost),
            "total_proposals_generated": int(self.total_proposals_generated),
            "sampling_seconds_ewma": self._sampling_seconds_ewma,
            "evaluation_seconds_ewma": self._evaluation_seconds_ewma,
            "proposal_timing_samples": int(self._proposal_timing_samples),
            "cost_limit_reached": bool(self.cost_limit_reached),
            "in_flight": {
                "running_jobs": len(self.running_jobs),
                "active_proposals": len(self.active_proposal_tasks),
                "failed_db_jobs": len(self.failed_jobs_for_retry),
                "submitted_jobs": len(self.submitted_jobs),
                "completed_jobs": self._get_completed_job_work_count(),
                "background_side_effects": self._get_background_side_effect_work_count(),
            },
        }

    def _restore_runner_state(self, state: dict[str, Any]) -> None:
        in_flight = state.get("in_flight")
        if not isinstance(in_flight, dict) or any(in_flight.values()):
            raise UncleanCheckpointError(
                "Checkpoint contains in-flight proposal, evaluation, or DB work"
            )
        if state.get("assigned_generations", []):
            raise UncleanCheckpointError(
                "Clean checkpoint contains assigned generation IDs"
            )
        self.completed_generations = int(state["completed_generations"])
        self.next_generation_to_submit = int(state["next_generation_to_submit"])
        self.assigned_generations = set()
        self.best_program_id = state.get("best_program_id")
        self.prompt_evolution_counter = int(state.get("prompt_evolution_counter", 0))
        self.prompt_percentile_recompute_counter = int(
            state.get("prompt_percentile_recompute_counter", 0)
        )
        self.current_prompt_id = state.get("current_prompt_id")
        self.prompt_api_cost = float(state.get("prompt_api_cost", 0.0))
        self.completed_proposal_costs = list(state.get("completed_proposal_costs", []))
        self.avg_proposal_cost = float(state.get("avg_proposal_cost", 0.0))
        self.total_proposals_generated = int(state.get("total_proposals_generated", 0))
        self._sampling_seconds_ewma = state.get("sampling_seconds_ewma")
        self._evaluation_seconds_ewma = state.get("evaluation_seconds_ewma")
        self._proposal_timing_samples = int(state.get("proposal_timing_samples", 0))
        self.cost_limit_reached = bool(state.get("cost_limit_reached", False))

    def _checkpoint_cleanliness_issues(
        self, *, allow_database_maintenance: bool = False
    ) -> list[str]:
        issues: list[str] = []
        if not self.pause_new_proposals.is_set():
            issues.append("proposal admission is not paused")
        for name, value in (
            ("running_jobs", len(self.running_jobs)),
            ("active_proposals", len(self.active_proposal_tasks)),
            ("failed_db_jobs", len(self.failed_jobs_for_retry)),
            ("submitted_jobs", len(self.submitted_jobs)),
            ("assigned_generations", len(self.assigned_generations)),
            ("completed_job_work", self._get_completed_job_work_count()),
            ("background_side_effects", self._get_background_side_effect_work_count()),
            ("sampling_slots", self.sampling_slot_pool.in_use),
            ("evaluation_slots", self.evaluation_slot_pool.in_use),
            ("postprocess_slots", self.postprocess_slot_pool.in_use),
        ):
            if value:
                issues.append(f"{name}={value}")
        if self.processing_lock.locked():
            issues.append("database processing lock is active")
        for lock, name in (
            (self._meta_side_effect_lock, "_meta_side_effect_lock"),
            (self._prompt_side_effect_lock, "_prompt_side_effect_lock"),
            (self._best_solution_lock, "_best_solution_lock"),
        ):
            if lock.locked():
                issues.append(f"{name} is active")
        prompt_task = self._prompt_percentile_recompute_task
        if prompt_task is not None and not prompt_task.done():
            issues.append("prompt percentile recomputation is active")
        if self._prompt_percentile_recompute_pending:
            issues.append("prompt percentile recomputation is pending")
        embedding_task = self.async_db._embedding_recompute_task
        if (
            not allow_database_maintenance
            and embedding_task is not None
            and not embedding_task.done()
        ):
            issues.append("embedding recomputation is active")
        if self.proposal_queue.qsize():
            issues.append("proposal queue is not empty")
        if self.side_effect_event_queue.qsize():
            issues.append("side-effect queue is not empty")
        return issues

    def _assert_clean_checkpoint_boundary(
        self, *, allow_database_maintenance: bool = False
    ) -> None:
        issues = self._checkpoint_cleanliness_issues(
            allow_database_maintenance=allow_database_maintenance
        )
        if issues:
            raise UncleanCheckpointError(
                "Cannot publish a clean checkpoint: " + "; ".join(issues)
            )

    @staticmethod
    def _checkpoint_sqlite_wal(connection: Any, label: str) -> None:
        connection.commit()
        result = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if result is not None and int(result[0]) != 0:
            raise CheckpointError(f"{label} WAL checkpoint remained busy")

    async def _commit_checkpoint_databases(self) -> None:
        await self.async_db.flush_async()
        self.db.save()
        self._checkpoint_sqlite_wal(self.db.conn, "program database")
        if self.prompt_db is not None:
            self.prompt_db.save()
            self._checkpoint_sqlite_wal(self.prompt_db.conn, "prompt database")

    def _checkpoint_identity(self) -> dict[str, Any]:
        named_generators = self._owned_rng_generators()
        return {
            "source": source_tree_identity(),
            "configuration_hashes": configuration_hashes(
                self.evo_config, self.db_config, self.job_config
            ),
            "runtime": runtime_identity(named_generators),
        }

    def _build_checkpoint_payload(self) -> dict[str, Any]:
        self._assert_clean_checkpoint_boundary()
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "clean": True,
            "identity": self._checkpoint_identity(),
            "database_watermark": self._database_watermark(),
            "runner_state": self._capture_runner_state(),
            "random_streams": capture_rng_states(self._owned_rng_generators()),
            "component_state": self._capture_component_state(),
        }

    def _validate_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        identity = self._checkpoint_identity()
        validate_checkpoint_identity(
            payload,
            expected_source=identity["source"],
            expected_config_hashes=identity["configuration_hashes"],
            expected_runtime=identity["runtime"],
        )
        live_watermark = self._database_watermark()
        if payload.get("database_watermark") != live_watermark:
            raise CheckpointCompatibilityError(
                "Database watermark does not match the checkpoint"
            )
        runner_state = payload.get("runner_state")
        if not isinstance(runner_state, dict):
            raise CheckpointCompatibilityError("Checkpoint runner state is missing")
        in_flight = runner_state.get("in_flight")
        if not isinstance(in_flight, dict) or any(in_flight.values()):
            raise UncleanCheckpointError(
                "Checkpoint runner state contains in-flight work"
            )
        next_generation = int(runner_state.get("next_generation_to_submit", -1))
        if self.evo_config.num_generations < next_generation:
            raise CheckpointCompatibilityError(
                "Configured num_generations is below checkpoint progress"
            )
        program_count = live_watermark["programs"]["program_count"]
        island_copies = max(0, getattr(self.db_config, "num_islands", 1) - 1)
        expected_completed = max(0, program_count - island_copies)
        if int(runner_state.get("completed_generations", -1)) != expected_completed:
            raise CheckpointCompatibilityError(
                "Runner progress does not match the database watermark"
            )
        random_streams = payload.get("random_streams")
        if not isinstance(random_streams, dict):
            raise CheckpointCompatibilityError("Checkpoint RNG state is missing")
        validate_rng_states(random_streams, self._owned_rng_generators())
        component_state = payload.get("component_state")
        if not isinstance(component_state, dict):
            raise CheckpointCompatibilityError("Checkpoint component state is missing")

    def _restore_checkpoint_without_rng(self, payload: dict[str, Any]) -> None:
        self._restore_component_state(payload["component_state"])
        self._restore_runner_state(payload["runner_state"])

    def _record_resume_status(
        self,
        *,
        deterministic_resume: bool,
        checkpoint_id: str | None,
        detail: str,
    ) -> None:
        metadata = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "resumed": True,
            "checkpoint_resume_mode": self.evo_config.checkpoint_resume_mode,
            "deterministic_resume": deterministic_resume,
            "checkpoint_id": checkpoint_id,
            "detail": detail,
        }
        self._resume_metadata = metadata
        write_resume_metadata(self.results_dir, metadata)
        logger.info(
            "Resume audit: deterministic_resume=%s checkpoint_id=%s (%s)",
            deterministic_resume,
            checkpoint_id,
            detail,
        )

    async def _publish_clean_checkpoint(self) -> dict[str, Any]:
        self._assert_clean_checkpoint_boundary(allow_database_maintenance=True)
        await self._commit_checkpoint_databases()
        self._assert_clean_checkpoint_boundary()
        payload = self._build_checkpoint_payload()
        path = write_checkpoint(self.results_dir, payload)
        self._checkpoint_published = True
        self.checkpoint_complete.set()
        logger.info(
            "Published clean checkpoint %s at %s (programs=%s, last_iteration=%s)",
            payload["checkpoint_id"],
            path,
            payload["database_watermark"]["programs"]["program_count"],
            payload["database_watermark"]["programs"]["last_iteration"],
        )
        return payload
