"""SQLite attempt events shared by the synchronous and asynchronous writers."""

import json
import sqlite3
import time
from typing import Any


def insert_attempt_event(
    conn: sqlite3.Connection,
    generation: int,
    stage: str,
    status: str,
    details: dict[str, Any] | None,
) -> None:
    payload = json.dumps(details, sort_keys=True) if details is not None else None
    charge_id = (details or {}).get("api_charge_id")
    if charge_id is not None:
        # Move the same durable charge to its next state, or retry that state
        # after an ambiguous commit. UPDATE acquires the writer transaction.
        updated = conn.execute(
            """
            UPDATE attempt_log SET stage = ?, status = ?, details = ?
            WHERE generation = ?
              AND CASE WHEN json_valid(details)
                       THEN json_extract(details, '$.api_charge_id') END = ?
            """,
            (stage, status, payload, generation, charge_id),
        ).rowcount
        if updated:
            conn.commit()
            return
    conn.execute(
        """
        INSERT INTO attempt_log (generation, stage, status, details, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (generation, stage, status, payload, time.time()),
    )
    conn.commit()
