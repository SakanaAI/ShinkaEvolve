"""Recover charges for failed work that has no program or prompt row."""

import asyncio
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, Awaitable, Iterable, Iterator

from shinka.database.display import stored_number
from shinka.database.prompt_dbase import SystemPromptConfig, SystemPromptDatabase
from shinka.llm.providers.errors import BilledCancellationError

_PROGRAM_COST_FIELDS = ("api_costs", "embed_cost", "novelty_cost", "meta_cost")


class BillingPersistenceError(RuntimeError):
    """Known charges could not be saved to the attempt journal."""


@dataclass(frozen=True)
class GeneratedSeedHandoff:
    charge_id: str
    code: str
    metadata: dict[str, Any]
    cost: float


def load_generated_seed_handoff(connection: sqlite3.Connection) -> GeneratedSeedHandoff | None:
    """Load the latest generated seed receipt until its original row commits."""
    row = connection.execute(
        """
        WITH latest_seed AS (
            SELECT details FROM attempt_log
            WHERE generation = 0 AND stage = 'evaluation' AND status = 'charged'
              AND CASE WHEN json_valid(details) THEN
                  json_extract(details, '$.node_kind') = 'evaluation_cost'
                  AND json_type(details, '$.seed_handoff') = 'object'
              END
            ORDER BY id DESC LIMIT 1
        )
        SELECT details FROM latest_seed
        WHERE NOT EXISTS (
            SELECT 1 FROM programs
            WHERE generation = 0 AND CASE WHEN json_valid(metadata) THEN
                json_extract(metadata, '$.api_charge_id') =
                    json_extract(latest_seed.details, '$.api_charge_id')
                AND NOT COALESCE(json_extract(metadata, '$._is_island_copy'), 0)
                AND NOT COALESCE(json_extract(metadata, '$._spawned_island'), 0)
            END
        )
        """
    ).fetchone()
    if row is None:
        return None

    details = json.loads(row[0])
    charge_id = details.get("api_charge_id")
    code = details["seed_handoff"].get("code")
    metadata = details["seed_handoff"].get("metadata")
    cost = details.get("api_costs")
    model_cost = metadata.get("api_costs") if isinstance(metadata, dict) else None
    if not (
        isinstance(charge_id, str) and charge_id
        and isinstance(code, str) and code
        and isinstance(cost, (int, float)) and math.isfinite(cost)
        and isinstance(model_cost, (int, float)) and 0.0 < model_cost <= cost
    ):
        raise BillingPersistenceError("Invalid generated seed charge receipt")
    return GeneratedSeedHandoff(charge_id, code, {**metadata, "api_charge_id": charge_id}, cost)


def api_cost_in_metadata(metadata: dict[str, Any]) -> float:
    return sum(stored_number(metadata.get(field)) for field in _PROGRAM_COST_FIELDS)


def cancellation_with_cost(
    cancellation: asyncio.CancelledError, cost: float
) -> BilledCancellationError:
    """Carry completed local attempts alongside an interrupted query's bill."""
    usage = dict(getattr(cancellation, "cost_results", {}))
    usage["cost"] = (usage.get("cost") or 0.0) + cost
    return BilledCancellationError(usage)


async def finish_cost_recording(operation: Awaitable[object]) -> None:
    """Finish the same journal write before propagating cancellation."""
    task = asyncio.ensure_future(operation)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


def failed_proposal_generations(
    entries: Iterable[tuple[int, str | None]],
) -> set[int]:
    """Generations whose proposal failed for good, among journal entries.

    Each entry is the generation and the details of a failed attempt. The
    details are parsed in Python: SQLite's JSON functions reject the NaN
    literals that early releases stored, and would drop those entries.
    """
    failed = set()
    for generation, raw_details in entries:
        try:
            details = json.loads(raw_details or "{}")
        except (ValueError, TypeError):
            continue
        if isinstance(details, dict) and details.get("node_kind") == "failed_proposal":
            failed.add(int(generation))
    return failed


def journaled_api_costs(
    connection: sqlite3.Connection, attached_charge_ids: set[str]
) -> float:
    total = 0.0
    counted = set(attached_charge_ids)
    for (raw_details,) in connection.execute(
        "SELECT details FROM attempt_log WHERE status IN ('failed', 'charged') ORDER BY id"
    ):
        try:
            details = json.loads(raw_details or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(details, dict) or details.get("node_kind") not in {
            "failed_proposal", "unattached_cost", "evaluation_cost"
        }:
            continue
        charge_id = details.get("api_charge_id")
        if isinstance(charge_id, str):
            if charge_id in counted:
                continue
            counted.add(charge_id)
        total += api_cost_in_metadata(details)
    return total


@dataclass(frozen=True)
class RunCosts:
    """Every charge a run has stored, split the way the closing table shows it."""

    proposals: float = 0.0
    embeddings: float = 0.0
    novelty: float = 0.0
    meta: float = 0.0
    # Failed and interrupted work has no program row: its charges are journaled.
    without_program: float = 0.0
    prompt_evolution: float = 0.0

    @property
    def total(self) -> float:
        return math.fsum(astuple(self))


def program_db_costs(connection: sqlite3.Connection) -> RunCosts:
    """Sum the charges of a program database: its rows and its charge journal."""
    on_rows = dict.fromkeys(_PROGRAM_COST_FIELDS, 0.0)
    attached_charge_ids: set[str] = set()
    for (raw_metadata,) in connection.execute(
        "SELECT metadata FROM programs WHERE metadata IS NOT NULL"
    ):
        try:
            metadata = json.loads(raw_metadata or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(metadata, dict):
            continue
        charge_id = metadata.get("api_charge_id")
        if isinstance(charge_id, str) and not (
            metadata.get("_is_island_copy") or metadata.get("_spawned_island")
        ):
            attached_charge_ids.add(charge_id)
        for field in _PROGRAM_COST_FIELDS:
            on_rows[field] += stored_number(metadata.get(field))

    has_journal = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'attempt_log'"
    ).fetchone()
    return RunCosts(
        proposals=on_rows["api_costs"],
        embeddings=on_rows["embed_cost"],
        novelty=on_rows["novelty_cost"],
        meta=on_rows["meta_cost"],
        without_program=(
            journaled_api_costs(connection, attached_charge_ids) if has_journal else 0.0
        ),
    )


def stored_prompt_evolution_cost(program_db_path: str | Path) -> float:
    """Read the prompt evolution charges kept beside a program database."""
    prompt_db_path = Path(program_db_path).with_name("prompts.sqlite")
    if not prompt_db_path.exists():
        return 0.0
    config = SystemPromptConfig(db_path=str(prompt_db_path))
    with closing(SystemPromptDatabase(config, read_only=True)) as stored:
        return stored.get_total_evolution_costs()


def stored_proposal_costs(connection: sqlite3.Connection) -> list[float]:
    """What each stored proposal cost, in the order of their generations.

    A resumed run reserves budget for its proposals from this history: evolved
    programs, proposals that failed for good, and proposals that were paid and
    handed to evaluation but left no program row because the run stopped first.
    Meta calls are left out: they are made after a proposal has been evaluated.
    """

    def records(query: str) -> Iterator[tuple[int, dict[str, Any]]]:
        for generation, raw in connection.execute(query):
            try:
                record = json.loads(raw or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield generation, record

    proposals: list[tuple[int, dict[str, Any]]] = []
    counted_charge_ids: set[str] = set()
    for generation, metadata in records(
        "SELECT generation, metadata FROM programs WHERE generation > 0"
    ):
        if metadata.get("_is_island_copy") or metadata.get("_spawned_island"):
            continue
        charge_id = metadata.get("api_charge_id")
        if isinstance(charge_id, str):
            counted_charge_ids.add(charge_id)
        proposals.append((generation, metadata))

    # A proposal is charged in the journal when it is handed to evaluation, and
    # that entry stays once its program row is stored or its failure recorded.
    for generation, details in records(
        "SELECT generation, details FROM attempt_log "
        "WHERE generation > 0 AND status IN ('failed', 'charged') ORDER BY id"
    ):
        if details.get("node_kind") not in {"failed_proposal", "evaluation_cost"}:
            continue
        charge_id = details.get("api_charge_id")
        if isinstance(charge_id, str):
            if charge_id in counted_charge_ids:
                continue
            counted_charge_ids.add(charge_id)
        proposals.append((generation, details))

    costs = (
        sum(
            stored_number(record.get(field))
            for field in ("api_costs", "embed_cost", "novelty_cost")
        )
        for _, record in sorted(proposals, key=lambda proposal: proposal[0])
    )
    return [cost for cost in costs if cost > 0.0]
