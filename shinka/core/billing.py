"""Recover charges for failed work that has no program or prompt row."""

import json
import sqlite3


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
