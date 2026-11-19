"""Files `store_best_path` writes for the code evolution animation."""

from contextlib import closing

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.edit.apply_diff import write_git_diff
from shinka.utils import get_path_to_best_node, load_programs_to_df, store_best_path


def _write_run(db_path, tmp_path, num_children: int) -> None:
    config = DatabaseConfig(db_path=str(db_path), num_islands=1)
    with closing(ProgramDatabase(config, embedding_model="")) as db:
        code = "value = 0\n"
        db.add(
            Program(
                id="g0",
                code=code,
                generation=0,
                correct=True,
                combined_score=0.0,
                metadata={"patch_name": "seed"},
            )
        )
        for generation in range(1, num_children + 1):
            new_code = f"value = {generation}\n"
            diff_path = write_git_diff(
                code, new_code, "original.py", tmp_path / "edit.diff"
            )
            db.add(
                Program(
                    id=f"g{generation}",
                    code=new_code,
                    parent_id=f"g{generation - 1}",
                    generation=generation,
                    correct=True,
                    combined_score=float(generation),
                    code_diff=diff_path.read_text(),
                    metadata={"patch_name": f"step-{generation}"},
                )
            )
            code = new_code


def test_best_path_has_the_seed_code_and_one_patch_per_step(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run(run_dir / "programs.sqlite", tmp_path, num_children=11)

    # The steps of shinka/plots/code_path_anim.py before it renders.
    df = load_programs_to_df(str(run_dir), verbose=False)
    best_path = get_path_to_best_node(df)
    store_best_path(best_path, str(run_dir))

    best_path_dir = run_dir / "best_path"
    assert (best_path_dir / "code" / "main_0.py").read_text() == "value = 0\n"
    assert (best_path_dir / "code" / "main_11.py").read_text() == "value = 11\n"
    # The seed has no diff, so patch i is the step that produced program i.
    patches = {path.name: path for path in (best_path_dir / "patches").iterdir()}
    assert sorted(patches) == sorted(f"patch_{i}.patch" for i in range(1, 12))
    assert "+value = 10" in patches["patch_10.patch"].read_text()
    assert best_path.patch_name.iloc[1:].tolist() == [
        f"step-{i}" for i in range(1, 12)
    ]


def test_best_path_is_read_from_a_legacy_database_file(tmp_path):
    _write_run(tmp_path / "evolution_db.sqlite", tmp_path, num_children=1)

    df = load_programs_to_df(str(tmp_path), verbose=False)

    assert df is not None
    assert list(df["id"]) == ["g0", "g1"]
