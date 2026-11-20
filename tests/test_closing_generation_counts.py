"""The closing summary says how many generations gave a program and how many failed.

A proposal that fails for good uses up its generation number: the run ends
below its target and a resume does not propose that generation again. The
reader of the summary must not have to work that out from the journal.
"""

import asyncio
import io
import logging
import re
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase

TARGET = 6
FAILED = [2, 4]


@pytest.fixture
def db(tmp_path):
    """Seed and generations 1 and 3 stored; 2 and 4 failed; 5 never ran."""
    database = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    for generation in (0, 1, 3):
        database.add(
            Program(
                id=f"gen{generation}",
                code=f"x = {generation}\n",
                generation=generation,
                correct=True,
                combined_score=float(generation),
            )
        )
    # Generation 9 failed in an earlier run with a higher target.
    for generation in (*FAILED, 9):
        database.record_attempt_event(
            generation, "proposal", "failed", {"node_kind": "failed_proposal"}
        )
    # A cut-off proposal is not a failed generation: a resume proposes it again.
    database.record_attempt_event(
        5, "cancelled_proposal", "failed",
        {"node_kind": "unattached_cost", "api_costs": 0.01},
    )
    yield database
    database.close()


def _closing_log(database, caplog) -> str:
    runner: Any = ShinkaEvolveRunner.__new__(ShinkaEvolveRunner)
    runner.verbose = True
    runner.db = database
    runner.evo_config = SimpleNamespace(num_generations=TARGET, max_api_costs=None)
    runner.start_time = None
    runner.completed_generations = 3
    runner.total_proposals_generated = 5
    runner.total_api_cost = 0.0
    runner.embedding_client = None
    runner.meta_summarizer = None
    runner.console = Console(file=io.StringIO(), color_system=None, width=200)
    runner._log_timing_bottleneck_summary = lambda: None
    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        asyncio.run(runner._print_final_summary())
    return caplog.text


def test_closing_log_counts_generations_with_a_program_and_failed_ones(db, caplog):
    log = _closing_log(db, caplog)

    assert f"Generations with a program: 3 of {TARGET}" in log
    assert re.search(r"Failed generations \(.*\): 2 \[2, 4\]", log), log


def _closing_table(database) -> str:
    output = io.StringIO()
    database.print_summary(
        console=Console(file=output, color_system=None, width=200),
        total_program_target=TARGET,
    )
    return output.getvalue()


def test_closing_table_counts_failed_generations(db):
    table = _closing_table(db)

    assert re.search(rf"Stored Generations\W+3 / {TARGET}\b", table), table
    assert re.search(r"Failed Generations\W+2\b", table), table


def test_failed_generation_with_a_stored_program_is_not_counted_as_failed(db):
    """A row recovered for a generation outranks an older failure entry."""
    db.record_attempt_event(3, "proposal", "failed", {"node_kind": "failed_proposal"})
    output = io.StringIO()

    db.print_summary(
        console=Console(file=output, color_system=None, width=200),
        total_program_target=TARGET,
    )

    assert re.search(r"Failed Generations\W+2\b", output.getvalue())


def test_log_and_table_agree_on_a_failure_an_early_release_journaled(db, caplog):
    """Early releases wrote NaN literals, which SQLite's JSON functions reject."""
    db.conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) "
        "VALUES (5, 'proposal', 'failed', "
        """'{"node_kind": "failed_proposal", "api_costs": NaN}', 0)"""
    )
    db.conn.commit()

    assert re.search(r"Failed generations \(.*\): 3 \[2, 4, 5\]", _closing_log(db, caplog))
    assert re.search(r"Failed Generations\W+3\b", _closing_table(db))


def test_closing_table_counts_no_failed_generation_without_a_journal(db):
    """A database written before the journal existed still gets its summary."""
    db.cursor.execute("DROP TABLE attempt_log")
    db.conn.commit()

    assert re.search(r"Failed Generations\W+0\b", _closing_table(db))
