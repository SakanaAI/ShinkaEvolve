"""Island copies must not carry the cost of the program they copy.

A copy only places an existing program on another island; making it calls no
LLM and no embedding API. Cost totals sum every row, so a copy that keeps the
original's costs counts that spend again.
"""

import io
import json
import re

import pytest
from rich.console import Console

from shinka.database import DatabaseConfig, Program, ProgramDatabase

COSTS = {"api_costs": 0.5, "embed_cost": 0.25, "novelty_cost": 0.125, "meta_cost": 1.0}


def _program(program_id, generation, score, parent_id=None):
    return Program(
        id=program_id,
        code=f"x = {generation}\n",
        generation=generation,
        parent_id=parent_id,
        correct=True,
        combined_score=score,
        metadata=dict(COSTS),
    )


def _db(tmp_path, **config):
    return ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), **config),
        embedding_model="",
        read_only=False,
    )


def _rows(db):
    db.cursor.execute("SELECT id, metadata FROM programs")
    return {row["id"]: json.loads(row["metadata"]) for row in db.cursor.fetchall()}


def test_initial_program_copies_carry_no_cost(tmp_path):
    db = _db(tmp_path, num_islands=3)
    try:
        db.add(_program("seed", 0, 0.1))
        rows = _rows(db)

        copies = [md for md in rows.values() if md.get("_is_island_copy")]
        assert len(copies) == 2
        for metadata in copies:
            assert {key: metadata[key] for key in COSTS} == dict.fromkeys(COSTS, 0.0)
        assert {key: rows["seed"][key] for key in COSTS} == COSTS
    finally:
        db.close()


def test_spawned_island_copies_carry_no_cost(tmp_path):
    db = _db(
        tmp_path,
        num_islands=2,
        island_spawn_strategy="best",
        island_spawn_subtree_size=2,
    )
    try:
        db.add(_program("seed", 0, 0.1))
        db.add(_program("best", 1, 0.9, parent_id="seed"))
        db.add(_program("child", 2, 0.5, parent_id="best"))
        assert db.island_manager.spawn_new_island()
        rows = _rows(db)

        spawned = [md for md in rows.values() if md.get("_spawned_island")]
        assert len(spawned) == 2
        for metadata in spawned:
            assert {key: metadata[key] for key in COSTS} == dict.fromkeys(COSTS, 0.0)
        for original in ("seed", "best", "child"):
            assert {key: rows[original][key] for key in COSTS} == COSTS
    finally:
        db.close()


@pytest.mark.parametrize("num_islands", [1, 2, 4])
def test_summary_embedding_total_matches_spend(tmp_path, num_islands):
    db = _db(tmp_path, num_islands=num_islands)
    try:
        db.add(_program("seed", 0, 0.1))
        for generation in range(1, 4):
            db.add(_program(f"gen{generation}", generation, 0.1 + generation / 10))

        output = io.StringIO()
        db.print_summary(console=Console(file=output, color_system=None, width=200))
        match = re.search(r"Total Embedding Cost\W+\$([0-9.]+)", output.getvalue())

        assert match, output.getvalue()
        assert float(match.group(1)) == pytest.approx(4 * COSTS["embed_cost"])
    finally:
        db.close()
