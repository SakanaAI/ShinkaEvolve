from __future__ import annotations

from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
    normalize_language,
)


def test_apply_diff_patch_supports_wolfram(tmp_path):
    original_content = """(* EVOLVE-BLOCK-START *)
f[n_Integer] := n
(* EVOLVE-BLOCK-END *)"""

    patch_content = """(* EVOLVE-BLOCK-START *)
<<<<<<< SEARCH
f[n_Integer] := n
=======
f[n_Integer] := n (n + 1) / 2
>>>>>>> REPLACE
(* EVOLVE-BLOCK-END *)"""

    patch_dir = tmp_path / "wolfram_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="wolfram",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "n (n + 1) / 2" in updated_content
    assert output_path == patch_dir / "main.wl"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.wl").exists()
    assert diff_path is not None and diff_path.exists()


def test_apply_full_patch_supports_wolfram_and_wl_fence(tmp_path):
    original_content = """(* EVOLVE-BLOCK-START *)
f[n_Integer] := n
(* EVOLVE-BLOCK-END *)
"""

    patch_content = """```wl
(* EVOLVE-BLOCK-START *)
f[n_Integer] := Fibonacci[n]
(* EVOLVE-BLOCK-END *)
```"""

    patch_dir = tmp_path / "wolfram_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="wolfram",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "Fibonacci[n]" in updated_content
    assert output_path == patch_dir / "main.wl"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()


def test_apply_full_patch_supports_wolfram_without_markers(tmp_path):
    original_content = """(* EVOLVE-BLOCK-START *)
f[n_Integer] := n
(* EVOLVE-BLOCK-END *)
"""

    patch_content = """```wl
f[n_Integer] := n + 1
```"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=tmp_path / "wolfram_full_without_markers",
        language="wolfram",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, _, _ = result

    assert error is None
    assert num_applied == 1
    assert updated_content == """(* EVOLVE-BLOCK-START *)
f[n_Integer] := n + 1
(* EVOLVE-BLOCK-END *)
"""
    assert output_path == tmp_path / "wolfram_full_without_markers" / "main.wl"


def test_apply_full_patch_supports_mathematica_fence(tmp_path):
    original_content = """(* EVOLVE-BLOCK-START *)
f[n_Integer] := n
(* EVOLVE-BLOCK-END *)
"""

    patch_content = """```mathematica
(* EVOLVE-BLOCK-START *)
f[n_Integer] := 2^n - 1
(* EVOLVE-BLOCK-END *)
```"""

    patch_dir = tmp_path / "wolfram_full_mathematica"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="mathematica",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, _, _ = result

    assert error is None
    assert num_applied == 1
    assert "2^n - 1" in updated_content
    assert output_path == patch_dir / "main.wl"


def test_language_helpers_support_wolfram_aliases():
    assert normalize_language("wolfram") == "wolfram"
    assert normalize_language("wl") == "wolfram"
    assert normalize_language("wls") == "wolfram"
    assert normalize_language("mathematica") == "wolfram"
    assert normalize_language("Mathematica") == "wolfram"

    assert get_language_extension("wolfram") == "wl"
    assert get_language_extension("wl") == "wl"
    assert get_language_extension("mathematica") == "wl"

    assert get_evolve_comment_prefix("wolfram") == "(*"
    assert get_evolve_comment_prefix("wl") == "(*"

    fences = get_code_fence_languages("wl")
    assert fences[0] == "wl"
    for tag in ("wolfram", "mathematica", "wls"):
        assert tag in fences

    fences_mma = get_code_fence_languages("mathematica")
    assert fences_mma[0] == "mathematica"
    assert "wolfram" in fences_mma
