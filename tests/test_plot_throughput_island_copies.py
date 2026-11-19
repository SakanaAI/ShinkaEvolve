"""Runtime plots recognise island copies in data loaded from a run database."""

from contextlib import closing

import pandas as pd

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.plots.plot_throughput import _island_copy_mask, _prepare_pool_runtime_data
from shinka.utils import load_programs_to_df

_TIMING = {
    "timeline_lane_mode": "pool_slots",
    "pipeline_started_at": 0.0,
    "sampling_started_at": 0.0,
    "sampling_finished_at": 2.0,
    "evaluation_started_at": 2.0,
    "evaluation_finished_at": 6.0,
    "postprocess_started_at": 6.0,
    "postprocess_finished_at": 8.0,
    "sampling_worker_id": 1,
    "evaluation_worker_id": 1,
    "postprocess_worker_id": 1,
}


def _load_run(tmp_path, config: DatabaseConfig):
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        db.add(
            Program(
                id="seed",
                code="pass",
                generation=0,
                correct=True,
                combined_score=1.0,
                metadata={"source_job_id": "job-seed", **_TIMING},
            )
        )
        if config.enable_dynamic_islands:
            assert db.island_manager.spawn_new_island()
    df = load_programs_to_df(str(tmp_path), verbose=False)
    assert df is not None and len(df) == 2
    return df


def _kept_ids(df) -> list[str]:
    prepared = _prepare_pool_runtime_data(df)
    assert prepared is not None
    return list(prepared.rows["id"])


def test_island_copy_never_replaces_its_original(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=2)
    df = _load_run(tmp_path, config)
    assert df["_is_island_copy"].eq(True).sum() == 1

    # Original and copy tie on score, correctness and timestamp.
    assert _kept_ids(df) == ["seed"]
    assert _kept_ids(df.iloc[::-1]) == ["seed"]


def test_spawned_island_copy_never_replaces_its_original(tmp_path):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=1,
        enable_dynamic_islands=True,
    )
    df = _load_run(tmp_path, config)
    assert df["_spawned_island"].eq(True).sum() == 1

    assert _kept_ids(df) == ["seed"]
    assert _kept_ids(df.iloc[::-1]) == ["seed"]


def test_copy_mask_reads_every_flag_column():
    """The metadata flags of a run database, and the plain column of older frames."""
    df = pd.DataFrame(
        {
            "_is_island_copy": [True, None, None, None],
            "_spawned_island": [None, True, None, None],
            "is_island_copy": [False, False, True, False],
        }
    )

    assert list(_island_copy_mask(df)) == [True, True, True, False]
    assert list(_island_copy_mask(df[["_spawned_island"]])) == [
        False,
        True,
        False,
        False,
    ]
    assert not _island_copy_mask(df[[]]).any()
