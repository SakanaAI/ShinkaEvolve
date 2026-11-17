"""Recover charges for failed work that has no program or prompt row."""

import asyncio
import json
import sqlite3
from typing import Any, Awaitable

from shinka.llm.providers.errors import BilledCancellationError


class BillingPersistenceError(RuntimeError):
    """Known charges could not be saved to the attempt journal."""


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
