import sqlite3
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from shinka.database.dbase import DatabaseConfig, Program, ProgramDatabase
from shinka.database.parents import CombinedParentSelector


@pytest.fixture
def beam_db() -> Iterator[ProgramDatabase]:
    config = DatabaseConfig(
        num_islands=1,
        archive_size=0,
        parent_selection_strategy="beam_search",
        num_beams=1,
        num_archive_inspirations=0,
        num_top_k_inspirations=0,
    )
    db = ProgramDatabase(config, embedding_model="")
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("sample_method", ["sample", "sample_with_fix_mode"])
def test_beam_search_prefers_zero_score_to_negative_score(beam_db, sample_method):
    for program_id, score in [("negative", -1.0), ("zero", 0.0)]:
        beam_db.add(
            Program(
                id=program_id,
                code="pass",
                correct=True,
                combined_score=score,
                generation=0,
            )
        )

    parent = getattr(beam_db, sample_method)()[0]

    assert parent.id == "zero"


@pytest.mark.parametrize("sample_method", ["sample", "sample_with_fix_mode"])
@pytest.mark.parametrize(
    "parent_metrics, child_metrics, child_timestamp, expected_id",
    [
        pytest.param({}, {}, 2.0, "child", id="newer-child"),
        pytest.param(
            {"quality": 1.0}, {"quality": 2.0}, 0.5, "child", id="better-metrics"
        ),
        pytest.param(
            {"quality": 2.0}, {"quality": 1.0}, 2.0, "parent", id="worse-metrics"
        ),
    ],
)
def test_completed_beam_uses_existing_tie_breaks(
    beam_db, sample_method, parent_metrics, child_metrics, child_timestamp, expected_id
):
    beam_db.add(
        Program(
            id="parent",
            code="pass",
            correct=True,
            combined_score=1.0,
            public_metrics=parent_metrics,
            timestamp=1.0,
            generation=0,
        )
    )
    assert getattr(beam_db, sample_method)()[0].id == "parent"

    beam_db.add(
        Program(
            id="child",
            code="pass",
            parent_id="parent",
            correct=True,
            combined_score=1.0,
            public_metrics=child_metrics,
            timestamp=child_timestamp,
            generation=1,
        )
    )

    parent = getattr(beam_db, sample_method)()[0]

    assert parent.id == expected_id


@pytest.mark.parametrize("sample_method", ["sample", "sample_with_fix_mode"])
def test_completed_beam_preserves_stored_program_fields(beam_db, sample_method):
    beam_db.add(
        Program(
            id="parent",
            code="pass",
            correct=True,
            combined_score=1.0,
        )
    )
    assert getattr(beam_db, sample_method)()[0].id == "parent"

    beam_db.add(
        Program(
            id="child",
            code="pass",
            parent_id="parent",
            generation=1,
            correct=True,
            combined_score=2.0,
            embedding=[1.0, 2.0],
            embedding_pca_2d=[0.1, 0.2],
            embedding_pca_3d=[0.1, 0.2, 0.3],
            archive_inspiration_ids=["parent"],
            top_k_inspiration_ids=["parent"],
            migration_history=[{"from": 1, "to": 0}],
        )
    )

    parent = getattr(beam_db, sample_method)()[0]

    assert parent.id == "child"
    assert parent == beam_db.get("child")


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            correct INTEGER NOT NULL,
            island_idx INTEGER,
            parent_id TEXT
        )
        """)

    cursor.executemany(
        """
        INSERT INTO programs (id, correct, island_idx, parent_id)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("best_island_0", 1, 0, None),
            ("program_island_1", 1, 1, None),
        ],
    )
    conn.commit()

    return conn, cursor


def _make_programs():
    return {
        "best_island_0": SimpleNamespace(
            id="best_island_0",
            correct=True,
            island_idx=0,
            generation=1,
            combined_score=100.0,
        ),
        "program_island_1": SimpleNamespace(
            id="program_island_1",
            correct=True,
            island_idx=1,
            generation=1,
            combined_score=10.0,
        ),
    }


def _make_selector(beam_search_parent_id=None):
    conn, cursor = _make_db()
    programs = _make_programs()

    def get_program(program_id):
        return programs.get(program_id)

    def get_best_program(island_idx=None):
        if island_idx == 1:
            return programs["program_island_1"]
        return programs["best_island_0"]

    config = SimpleNamespace(
        parent_selection_strategy="beam_search",
        num_beams=5,
    )

    selector = CombinedParentSelector(
        cursor=cursor,
        conn=conn,
        config=config,
        get_program_func=get_program,
        best_program_id="best_island_0",
        beam_search_parent_id=beam_search_parent_id,
        get_best_program_func=get_best_program,
    )

    return selector, conn


def test_beam_search_initial_parent_respects_requested_island():
    selector, conn = _make_selector()

    try:
        parent = selector.sample_parent(island_idx=1)

        assert parent.id == "program_island_1"
        assert parent.island_idx == 1
    finally:
        conn.close()


def test_beam_search_active_parent_does_not_cross_islands():
    selector, conn = _make_selector(
        beam_search_parent_id="best_island_0",
    )

    try:
        parent = selector.sample_parent(island_idx=1)

        assert parent.id == "program_island_1"
        assert parent.island_idx == 1
    finally:
        conn.close()


def test_program_database_beam_parent_persists_across_samples():
    config = DatabaseConfig(
        num_islands=1,
        archive_size=0,
        parent_selection_strategy="beam_search",
        num_beams=3,
        num_archive_inspirations=0,
        num_top_k_inspirations=0,
    )

    db = ProgramDatabase(config, embedding_model="")

    try:
        original_parent = Program(
            id="parent",
            code="pass",
            correct=True,
            combined_score=1.0,
            generation=0,
        )
        db.add(original_parent, defer_maintenance=True)
        db._update_best_program(original_parent)

        parent_1, _, _ = db.sample()

        assert parent_1.id == "parent"

        # Add a better child. The current beam should nevertheless remain
        # locked on "parent" until it reaches num_beams children.
        better_child = Program(
            id="child",
            code="pass",
            parent_id="parent",
            correct=True,
            combined_score=2.0,
            generation=1,
        )
        db.add(better_child, defer_maintenance=True)
        db._update_best_program(better_child)

        parent_2, _, _ = db.sample()

        assert parent_2.id == "parent", (
            "Beam search switched to the new best program before the active "
            "parent reached num_beams children"
        )

    finally:
        db.close()
