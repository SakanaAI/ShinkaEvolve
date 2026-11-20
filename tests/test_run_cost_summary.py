"""The cost a summary shows has to be the cost the run's own accounting counts.

A proposal that fails, or that an interrupt cuts off, leaves no program row:
its charge is kept in the attempt journal. Prompt evolution keeps its charges
in the prompt database. The runner counts all three, so the summaries must too.
"""

import asyncio
import io
import re

import pytest
from rich.console import Console

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPrompt,
    SystemPromptConfig,
    SystemPromptDatabase,
)

ROW_COSTS = {"api_costs": 0.05, "embed_cost": 0.002, "novelty_cost": 0.003}
FAILED_PROPOSAL = 0.0064
INTERRUPTED_PROPOSAL = 0.0139
PROGRAM_ROWS = 3 * sum(ROW_COSTS.values()) + 0.01  # generation 1 also paid a meta call
JOURNAL = FAILED_PROPOSAL + INTERRUPTED_PROPOSAL


def _program(program_id, generation, **metadata):
    return Program(
        id=program_id,
        code=f"x = {generation}\n",
        generation=generation,
        parent_id="seed" if generation else None,
        correct=True,
        combined_score=0.1 + generation / 10,
        metadata=metadata,
    )


@pytest.fixture
def db(tmp_path):
    """Three children, one failed proposal and one proposal cut off before its row."""
    database = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    database.add(_program("seed", 0))
    database.add(_program("gen1", 1, **ROW_COSTS, meta_cost=0.01, api_charge_id="charge-1"))
    database.add(_program("gen2", 2, **ROW_COSTS, api_charge_id="charge-2"))
    database.add(_program("gen4", 4, **ROW_COSTS, api_charge_id="charge-4"))
    for generation, charge_id in ((1, "charge-1"), (2, "charge-2"), (4, "charge-4")):
        # The receipt written before evaluation; the row carries the same charge.
        database.record_attempt_event(
            generation, "evaluation", "charged",
            {
                "node_kind": "evaluation_cost",
                "api_charge_id": charge_id,
                "api_costs": sum(ROW_COSTS.values()),
            },
        )
    database.record_attempt_event(
        3, "proposal", "failed",
        {"node_kind": "failed_proposal", "api_costs": FAILED_PROPOSAL},
    )
    database.record_attempt_event(
        4, "cancelled_proposal", "failed",
        {"node_kind": "unattached_cost", "api_costs": INTERRUPTED_PROPOSAL},
    )
    yield database
    database.close()


def _run_accounting(database) -> float:
    """What a resumed run loads as the cost spent so far."""
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db = database
    runner.prompt_db = None
    return asyncio.run(runner._get_total_api_costs())


def _console(output):
    return Console(file=output, color_system=None, width=200)


def _closing_summary(database) -> str:
    output = io.StringIO()
    database.print_summary(console=_console(output))
    return output.getvalue()


def _amount(output: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\W+\$([0-9.]+)", output)
    assert match, output
    return float(match.group(1))


def _add_prompt_evolution_cost(tmp_path, cost: float) -> None:
    config = SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
    prompt_db = SystemPromptDatabase(config)
    try:
        prompt_db.add(
            SystemPrompt(id="p1", prompt_text="be brief", metadata={"llm": {"cost": cost}})
        )
    finally:
        prompt_db.close()


def test_closing_table_total_is_the_run_accounting(db):
    summary = _closing_summary(db)

    assert _run_accounting(db) == pytest.approx(PROGRAM_ROWS + JOURNAL)
    assert _amount(summary, "Total Combined Cost") == pytest.approx(
        _run_accounting(db), abs=1e-4
    )


def test_closing_table_shows_failed_and_interrupted_work_on_its_own_line(db):
    summary = _closing_summary(db)

    assert _amount(summary, "Failed/Interrupted Cost") == pytest.approx(JOURNAL, abs=1e-4)
    lines = ("Proposal", "Embedding", "Novelty", "Meta")
    shown = [_amount(summary, f"Total {line} Cost") for line in lines]
    assert sum(shown) == pytest.approx(PROGRAM_ROWS, abs=1e-4)


def test_closing_table_counts_prompt_evolution(db, tmp_path):
    _add_prompt_evolution_cost(tmp_path, 0.04)

    summary = _closing_summary(db)

    assert _amount(summary, "Prompt Evolution Cost") == pytest.approx(0.04)
    assert _amount(summary, "Total Combined Cost") == pytest.approx(
        _run_accounting(db), abs=1e-4
    )
    assert _run_accounting(db) == pytest.approx(PROGRAM_ROWS + JOURNAL + 0.04)


def test_running_totals_in_table_titles_are_the_run_accounting(db, tmp_path):
    _add_prompt_evolution_cost(tmp_path, 0.04)
    output = io.StringIO()
    db.set_display_console(_console(output))

    db._print_program_summary(db.get("gen4"))
    db._print_sampling_summary_helper(db.get("gen4"), [], [])

    shown = re.findall(r"Total Cost: \$([0-9.]+)", output.getvalue())
    assert [float(amount) for amount in shown] == [
        pytest.approx(_run_accounting(db), abs=0.005)
    ] * 2


def test_closing_table_reads_a_database_without_a_journal(tmp_path):
    """A database written before the journal existed still gets its summary."""
    database = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    try:
        database.add(_program("seed", 0, api_costs=0.25))
        database.cursor.execute("DROP TABLE attempt_log")
        database.conn.commit()

        summary = _closing_summary(database)

        assert _amount(summary, "Total Combined Cost") == pytest.approx(0.25)
        assert _amount(summary, "Failed/Interrupted Cost") == 0.0
    finally:
        database.close()
