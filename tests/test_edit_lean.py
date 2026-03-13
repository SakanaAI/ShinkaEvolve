from shinka.core import run_shinka_eval
from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
)


def test_apply_diff_patch_supports_lean_markers(tmp_path):
    original_content = """-- EVOLVE-BLOCK-START
theorem demo : True := by
  trivial
-- EVOLVE-BLOCK-END"""

    patch_content = """-- EVOLVE-BLOCK-START
<<<<<<< SEARCH
trivial
=======
exact True.intro
>>>>>>> REPLACE
-- EVOLVE-BLOCK-END"""

    patch_dir = tmp_path / "lean_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="lean",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "exact True.intro" in updated_content
    assert output_path == patch_dir / "main.lean"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.lean").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "exact True.intro" in patch_txt

    search_replace_txt = (patch_dir / "search_replace.txt").read_text("utf-8")
    assert "EVOLVE-BLOCK-START" not in search_replace_txt
    assert "EVOLVE-BLOCK-END" not in search_replace_txt


def test_apply_full_patch_supports_lean4_fence(tmp_path):
    original_content = """-- EVOLVE-BLOCK-START
theorem demo : True := by
  trivial
-- EVOLVE-BLOCK-END
"""

    patch_content = """```lean4
-- EVOLVE-BLOCK-START
theorem demo : True := by
  exact True.intro
-- EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "lean_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="lean",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "exact True.intro" in updated_content
    assert output_path == patch_dir / "main.lean"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.lean").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "exact True.intro" in patch_txt


def test_language_helpers_support_lean_aliases():
    assert get_language_extension("lean") == "lean"
    assert get_language_extension("lean4") == "lean"
    assert get_evolve_comment_prefix("lean") == "--"
    assert get_evolve_comment_prefix("lean4") == "--"

    fences = get_code_fence_languages("lean4")
    assert fences[0] == "lean4"
    assert "lean" in fences


def test_run_shinka_eval_supports_callable_for_lean_tasks(tmp_path):
    program_path = tmp_path / "initial.lean"
    program_path.write_text("-- EVOLVE-BLOCK-START\n()\n-- EVOLVE-BLOCK-END\n", "utf-8")

    def fake_experiment(file_path: str, **_: object) -> str:
        return file_path

    metrics, correct, error = run_shinka_eval(
        program_path=str(program_path),
        results_dir=str(tmp_path / "results"),
        experiment_fn_name=fake_experiment,
        num_runs=1,
        aggregate_metrics_fn=lambda results: {
            "combined_score": 1.0,
            "returned_path": results[0],
        },
        validate_fn=lambda result: (
            result == str(program_path),
            None if result == str(program_path) else "wrong file path",
        ),
    )

    assert correct is True
    assert error is None
    assert metrics["returned_path"] == str(program_path)
