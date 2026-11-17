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
    params = (generation, stage, status, payload, time.time())
    charge_id = (details or {}).get("api_charge_id")
    if charge_id is None:
        conn.execute(
            """
            INSERT INTO attempt_log (generation, stage, status, details, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
    else:
        # Retry the same transition safely, including after an ambiguous commit.
        conn.execute(
            """
            INSERT INTO attempt_log (generation, stage, status, details, created_at)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM attempt_log
                WHERE generation = ? AND stage = ? AND status = ?
                  AND CASE WHEN json_valid(details)
                      THEN json_extract(details, '$.api_charge_id') END = ?
            )
            """,
            (*params, generation, stage, status, charge_id),
        )
    conn.commit()
