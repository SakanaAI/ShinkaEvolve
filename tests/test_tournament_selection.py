"""Tournament parents: finite island-local elites, distinct draws, uniform ties."""

import asyncio
from collections import Counter
from itertools import combinations
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.parents import CombinedParentSelector, ParentSamplingStrategy
from shinka.database.tournament import TournamentSamplingStrategy


@pytest.fixture
def population():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        "CREATE TABLE programs (id TEXT PRIMARY KEY, combined_score REAL, "
        "correct INTEGER, island_idx INTEGER, generation INTEGER);"
        "CREATE TABLE archive (program_id TEXT PRIMARY KEY);"
    )
    programs = {}

    def add(program_id, score, *, correct=True, island=0, archived=True):
        program = SimpleNamespace(
            id=program_id,
            combined_score=score,
            correct=correct,
            island_idx=island,
            generation=0,
        )
        programs[program_id] = program
        connection.execute(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?)",
            (program_id, score, correct, island, 0),
        )
        if archived:
            connection.execute("INSERT INTO archive VALUES (?)", (program_id,))
        return program

    def strategy(*, top_k=8, draws=3, island=0):
        config = DatabaseConfig(
            parent_selection_strategy="tournament",
            tournament_top_k=top_k,
            tournament_num_draws=draws,
        )
        return TournamentSamplingStrategy(
            connection.cursor(),
            connection,
            config,
            programs.get,
            best_program_id=None,
            island_idx=island,
        )

    def selector():
        config = DatabaseConfig(parent_selection_strategy="tournament")
        return CombinedParentSelector(
            connection.cursor(),
            connection,
            config,
            programs.get,
        )

    yield SimpleNamespace(add=add, strategy=strategy, selector=selector)
    connection.close()


def _control_draw(monkeypatch, indices, *, tie_index=0):
    draws = []
    ties = []

    def choice(values, size=None, replace=True, p=None):
        candidates = np.arange(values) if np.isscalar(values) else np.asarray(values)
        assert p is None, "Tournament draws and ties must be uniform"
        if size is not None:
            assert replace is False, "A tournament cannot draw a candidate twice"
            assert size == len(indices)
            draws.append(candidates.tolist())
            return np.asarray(indices)
        ties.append(candidates.tolist())
        return candidates[tie_index if len(candidates) > 1 else 0]

    monkeypatch.setattr("shinka.database.tournament.np.random.choice", choice)
    return draws, ties


def test_tournament_defaults_and_strategy_contract():
    config = DatabaseConfig(parent_selection_strategy="tournament")
    assert (config.tournament_top_k, config.tournament_num_draws) == (8, 3)
    assert issubclass(TournamentSamplingStrategy, ParentSamplingStrategy)


@pytest.mark.parametrize("field", ["tournament_top_k", "tournament_num_draws"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "3", None])
def test_tournament_rejects_nonpositive_or_noninteger_parameters(field, invalid):
    with pytest.raises(ValueError, match=field):
        DatabaseConfig(parent_selection_strategy="tournament", **{field: invalid})


def test_tournament_rejects_more_draws_than_top_k():
    with pytest.raises(ValueError, match="tournament_num_draws"):
        DatabaseConfig(
            parent_selection_strategy="tournament",
            tournament_top_k=2,
            tournament_num_draws=3,
        )


@pytest.mark.parametrize("strategy", ["weighted", "power_law", "beam_search"])
def test_other_strategies_ignore_unused_tournament_parameters(strategy):
    config = DatabaseConfig(
        parent_selection_strategy=strategy,
        tournament_top_k=0,
        tournament_num_draws=True,
    )
    assert config.parent_selection_strategy == strategy


def test_eight_candidate_three_draw_distribution_is_exact(population, monkeypatch):
    for rank in range(8):
        population.add(f"rank-{rank}", 8 - rank)
    counts = Counter()
    for subset in combinations(range(8), 3):
        draws, _ = _control_draw(monkeypatch, subset)
        counts[population.strategy().sample_parent().id] += 1
        assert draws == [list(range(8))]
    assert [counts[f"rank-{rank}"] for rank in range(8)] == [21, 15, 10, 6, 3, 1, 0, 0]
    assert sum(counts.values()) == 56


@pytest.mark.parametrize("size", [1, 2])
def test_small_pool_clamps_draw_count(population, monkeypatch, size):
    for index in range(size):
        population.add(f"p{index}", size - index)
    draws, _ = _control_draw(monkeypatch, list(range(size)))
    assert population.strategy().sample_parent().id == "p0"
    assert draws == [list(range(size))]


def test_one_draw_can_select_every_rank(population, monkeypatch):
    for rank in range(8):
        population.add(f"p{rank}", 8 - rank)
    selected = []
    for index in range(8):
        _control_draw(monkeypatch, [index])
        selected.append(population.strategy(draws=1).sample_parent().id)
    assert selected == [f"p{index}" for index in range(8)]


def test_drawing_whole_top_k_returns_best(population):
    for rank in range(5):
        population.add(f"p{rank}", 5 - rank)
    for _ in range(5):
        assert population.strategy(top_k=5, draws=5).sample_parent().id == "p0"


@pytest.mark.parametrize("tie_index,expected", [(0, "a"), (1, "b")])
def test_equal_best_draws_are_selected_uniformly(
    population, monkeypatch, tie_index, expected
):
    population.add("b", 10)
    population.add("a", 10)
    population.add("c", 9)
    _, ties = _control_draw(monkeypatch, [0, 1, 2], tie_index=tie_index)
    assert population.strategy().sample_parent().id == expected
    assert len(ties) == 1
    assert len(ties[0]) == 2


def test_top_k_tie_boundary_uses_stable_id_order(population, monkeypatch):
    for program_id in ["d", "b", "c", "a"]:
        population.add(program_id, 1)
    _control_draw(monkeypatch, [1])
    assert population.strategy(top_k=2, draws=1).sample_parent().id == "b"


def test_archive_pool_excludes_better_unarchived_programs(population):
    population.add("archived", 1)
    population.add("unarchived", 100, archived=False)
    assert population.strategy().sample_parent().id == "archived"


def test_archive_pool_is_limited_to_top_k(population, monkeypatch):
    for rank in range(5):
        population.add(f"p{rank}", 5 - rank)
    draws, _ = _control_draw(monkeypatch, [1])
    assert population.strategy(top_k=2, draws=1).sample_parent().id == "p1"
    assert draws == [[0, 1]]


def test_invalid_scores_are_filtered_before_top_k(population):
    for program_id, score in [
        ("infinite", float("inf")),
        ("negative-infinite", float("-inf")),
        ("nan", float("nan")),
        ("null", None),
    ]:
        population.add(program_id, score)
    population.add("incorrect", 999, correct=False)
    population.add("negative", -2)
    population.add("more-negative", -3)
    assert population.strategy(top_k=2, draws=2).sample_parent().id == "negative"


def test_zero_score_beats_negative_scores(population):
    population.add("zero", 0)
    population.add("negative", -1)
    assert population.strategy().sample_parent().id == "zero"


@pytest.mark.parametrize("archived", [True, False])
def test_parent_pool_is_island_local_including_fallback(population, archived):
    population.add("local", -1, island=1, archived=archived)
    population.add("foreign", 100, island=0, archived=True)
    assert population.strategy(island=1).sample_parent().id == "local"


def test_no_island_constraint_allows_all_islands(population):
    population.add("local", 1, island=0)
    population.add("other", 2, island=1)
    assert population.strategy(island=None).sample_parent().id == "other"


def test_ineligible_archive_falls_back_to_correct_finite_programs(population):
    population.add("incorrect", 100, correct=False)
    population.add("infinite", float("inf"))
    population.add("fallback", -1, archived=False)
    population.add("foreign", 1000, island=1)
    assert population.strategy().sample_parent().id == "fallback"


@pytest.mark.parametrize("state", ["empty", "incorrect", "nonfinite", "foreign"])
def test_no_eligible_candidates_raises(population, state):
    if state == "incorrect":
        population.add("bad", 1, correct=False)
    elif state == "nonfinite":
        population.add("bad", float("inf"))
    elif state == "foreign":
        population.add("bad", 1, island=1)
    with pytest.raises(ValueError):
        population.strategy().sample_parent()


def test_combined_selector_dispatches_tournament(population):
    population.add("best", 10)
    population.add("other", 1)
    parent, needs_fix = population.selector().sample_parent_with_fix_mode(island_idx=0)
    assert parent.id == "best"
    assert needs_fix is False


def test_combined_selector_preserves_island_fix_mode(population):
    population.add("broken", -1, correct=False, island=0)
    population.add("foreign-correct", 10, island=1)
    parent, needs_fix = population.selector().sample_parent_with_fix_mode(island_idx=0)
    assert parent.id == "broken"
    assert needs_fix is True


def test_async_database_uses_tournament_in_worker_connections(tmp_path):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=1,
        parent_selection_strategy="tournament",
        tournament_top_k=2,
        tournament_num_draws=2,
        num_archive_inspirations=0,
        num_top_k_inspirations=0,
    )
    database = ProgramDatabase(config, embedding_model="")
    for generation, score in enumerate([1.0, 2.0]):
        database.add(
            Program(
                id=f"p{generation}",
                code="x = 1",
                generation=generation,
                correct=True,
                combined_score=score,
                island_idx=0,
            )
        )

    async def run():
        async_db = AsyncProgramDatabase(database, max_workers=2)
        try:
            results = await asyncio.gather(
                async_db.sample_async(),
                async_db.sample_with_fix_mode_async(),
            )
            assert [result[0].id for result in results] == ["p1", "p1"]
            assert results[1][3] is False
        finally:
            await async_db.close_async()

    try:
        asyncio.run(run())
    finally:
        database.close()
