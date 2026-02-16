from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
)


def test_apply_diff_patch_supports_julia(tmp_path):
    original_content = """# EVOLVE-BLOCK-START
function score(x)
    return x + 1
end
# EVOLVE-BLOCK-END"""

    patch_content = """# EVOLVE-BLOCK-START
<<<<<<< SEARCH
return x + 1
=======
return x + 2
>>>>>>> REPLACE
# EVOLVE-BLOCK-END"""

    patch_dir = tmp_path / "julia_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="julia",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "return x + 2" in updated_content
    assert output_path == patch_dir / "main.jl"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.jl").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "return x + 2" in patch_txt

    # Markers should be stripped before SEARCH/REPLACE parsing.
    search_replace_txt = (patch_dir / "search_replace.txt").read_text("utf-8")
    assert "EVOLVE-BLOCK-START" not in search_replace_txt
    assert "EVOLVE-BLOCK-END" not in search_replace_txt


def test_apply_full_patch_supports_julia_and_jl_fence(tmp_path):
    original_content = """# EVOLVE-BLOCK-START
function score(x)
    return x + 1
end
# EVOLVE-BLOCK-END
"""

    patch_content = """```jl
# EVOLVE-BLOCK-START
function score(x)
    return x + 10
end
# EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "julia_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="julia",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "return x + 10" in updated_content
    assert output_path == patch_dir / "main.jl"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.jl").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "return x + 10" in patch_txt


def test_language_helpers_support_julia_aliases():
    assert get_language_extension("julia") == "jl"
    assert get_language_extension("jl") == "jl"
    assert get_evolve_comment_prefix("julia") == "#"
    assert get_evolve_comment_prefix("jl") == "#"

    fences = get_code_fence_languages("jl")
    assert fences[0] == "jl"
    assert "julia" in fences
