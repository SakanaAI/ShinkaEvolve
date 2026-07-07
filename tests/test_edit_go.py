from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
    normalize_language,
)


def test_apply_diff_patch_supports_go(tmp_path):
    original_content = """// EVOLVE-BLOCK-START
func score(x int) int {
    return x + 1
}
// EVOLVE-BLOCK-END"""

    patch_content = """// EVOLVE-BLOCK-START
<<<<<<< SEARCH
    return x + 1
=======
    return x + 2
>>>>>>> REPLACE
// EVOLVE-BLOCK-END"""

    patch_dir = tmp_path / "go_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="go",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "return x + 2" in updated_content
    assert output_path == patch_dir / "main.go"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.go").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "return x + 2" in patch_txt

    search_replace_txt = (patch_dir / "search_replace.txt").read_text("utf-8")
    assert "EVOLVE-BLOCK-START" not in search_replace_txt
    assert "EVOLVE-BLOCK-END" not in search_replace_txt


def test_apply_full_patch_supports_go_and_golang_fence(tmp_path):
    original_content = """// EVOLVE-BLOCK-START
func score(x int) int {
    return x + 1
}
// EVOLVE-BLOCK-END
"""

    patch_content = """```golang
// EVOLVE-BLOCK-START
func score(x int) int {
    return x + 10
}
// EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "go_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="go",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "return x + 10" in updated_content
    assert output_path == patch_dir / "main.go"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.go").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "return x + 10" in patch_txt


def test_language_helpers_support_go_aliases():
    assert normalize_language("go") == "go"
    assert normalize_language("golang") == "go"
    assert normalize_language("Go") == "go"
    assert get_language_extension("go") == "go"
    assert get_language_extension("golang") == "go"
    assert get_evolve_comment_prefix("go") == "//"
    assert get_evolve_comment_prefix("golang") == "//"

    fences = get_code_fence_languages("golang")
    assert fences[0] == "golang"
    assert "go" in fences
