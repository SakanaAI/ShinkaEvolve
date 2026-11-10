"""Recover charges for failed work that has no program or prompt row."""

import asyncio
import json
import sqlite3
from typing import Awaitable

from shinka.llm.providers.errors import BilledCancellationError


def cancellation_with_cost(
    cancellation: asyncio.CancelledError, cost: float
) -> BilledCancellationError:
    """Carry completed local attempts alongside an interrupted query's bill."""
    usage = dict(getattr(cancellation, "cost_results", {}))
    usage["cost"] = (usage.get("cost") or 0.0) + cost
    return BilledCancellationError(usage)


async def finish_cost_recording(operation: Awaitable[None]) -> None:
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


def failed_attempt_costs(connection: sqlite3.Connection) -> float:
    total = 0.0
    for (raw_details,) in connection.execute(
        "SELECT details FROM attempt_log WHERE status = 'failed'"
    ):
        try:
            details = json.loads(raw_details or "{}")
        except json.JSONDecodeError:
            continue
        if details.get("node_kind") not in {"failed_proposal", "unattached_cost"}:
            continue
        total += sum(
            details.get(field, 0.0) or 0.0
            for field in ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
        )
    return total
