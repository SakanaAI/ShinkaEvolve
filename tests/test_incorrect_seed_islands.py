"""Parent sampling after a seed that failed evaluation."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase


@pytest.fixture
def config(tmp_path: Path) -> DatabaseConfig:
    # The defaults under test: two islands and no migration between them.
    return DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"))


@pytest.fixture
def db(config: DatabaseConfig) -> Iterator[ProgramDatabase]:
    with closing(ProgramDatabase(config, embedding_model="")) as database:
        yield database


def _seed(correct: bool = False, **metadata) -> Program:
    return Program(
        id="seed",
        code="seed",
        correct=correct,
        combined_score=0.0,
        generation=0,
        timestamp=1.0,
        metadata=metadata,
    )


def _child(generation: int, *, correct: bool = True, **metadata) -> Program:
    return Program(
        id=f"child-{generation}",
        code=f"child {generation}",
        parent_id="seed",
        correct=correct,
        combined_score=float(generation),
        generation=generation,
        timestamp=1.0 + generation,
        metadata=metadata,
    )


def _islands_of(db: ProgramDatabase, *, correct: bool) -> dict[int, list[str]]:
    islands: dict[int, list[str]] = {}
    for program in db.get_all_programs():
        if program.correct == correct and program.island_idx is not None:
            islands.setdefault(program.island_idx, []).append(program.id)
    return islands


@pytest.mark.parametrize("sample_method", ["sample", "sample_with_fix_mode"])
def test_first_correct_child_ends_seed_fixing(db, sample_method):
    db.add(_seed())
    assert db.sample_with_fix_mode()[3] is True

    db.add(_child(1))

    for _ in range(40):
        sampled = getattr(db, sample_method)()
        assert sampled[0].correct
        if sample_method == "sample_with_fix_mode":
            assert sampled[3] is False


def test_first_correct_child_initializes_every_island(db):
    db.add(_seed())
    db.add(_child(1, correct=False))
    assert db.island_manager.get_initialized_islands() == []

    db.add(_child(2, api_costs=0.5))

    assert db.island_manager.are_all_islands_initialized()
    correct = _islands_of(db, correct=True)
    assert correct[0] == ["child-2"]
    (copy_id,) = correct[1]
    copy = db.get(copy_id)
    assert copy is not None
    assert copy.code == "child 2"
    assert copy.generation == 2
    assert copy.metadata["_is_island_copy"] is True
    assert copy.metadata["_original_program_id"] == "child-2"
    assert copy.metadata["api_costs"] == 0.0

    # Later correct programs find every island initialized and add no copies.
    db.add(_child(3))
    assert sum(len(ids) for ids in _islands_of(db, correct=True).values()) == 3


def test_both_islands_are_sampled_after_an_incorrect_seed(db):
    db.add(_seed())
    db.add(_child(1))

    sampled_islands = {db.sample_with_fix_mode()[0].island_idx for _ in range(60)}

    assert sampled_islands == {0, 1}


def test_correct_seed_is_copied_once_per_island(db):
    db.add(_seed(correct=True))
    db.add(_child(1))

    assert {k: len(v) for k, v in _islands_of(db, correct=True).items()} == {
        0: 2,
        1: 1,
    }


def test_database_stuck_on_an_incorrect_seed_recovers(config):
    """A run written before the fix has correct children on one island only."""
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.add(_seed())
        for generation in range(1, 4):
            db.add(_child(generation), defer_maintenance=True)
        assert db.island_manager.get_initialized_islands() == [0]

    # The runner samples through read-only connections.
    with closing(ProgramDatabase(config, read_only=True)) as reader:
        for _ in range(40):
            parent, _, _, needs_fix = reader.sample_with_fix_mode()
            assert (parent.correct, parent.island_idx, needs_fix) == (True, 0, False)

    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.add(_child(4))
        assert db.island_manager.are_all_islands_initialized()


def test_repaired_failed_seed_does_not_trap_the_run(db):
    db.add(_seed(correct=True, evaluation_failed=True))
    db.repair_legacy_failed_seeds()
    assert db.island_manager.get_initialized_islands() == []
    assert db.sample_with_fix_mode()[3] is True

    db.add(_child(1))

    assert db.island_manager.are_all_islands_initialized()
    assert all(db.sample_with_fix_mode()[3] is False for _ in range(20))
