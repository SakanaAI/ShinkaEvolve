"""Recover charges for failed work that has no program or prompt row."""

import asyncio
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Awaitable

from shinka.llm.providers.errors import BilledCancellationError


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
    return sum(
        metadata.get(field, 0.0) or 0.0
        for field in ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
    )


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
