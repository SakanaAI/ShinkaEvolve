from __future__ import annotations

import pytest

from shinka.edit.marker_validation import validate_evolve_markers
from shinka.edit import apply_full_patch, apply_diff_patch

# ---------------------------------------------------------------------------
# Well-formed cases — should validate cleanly
# ---------------------------------------------------------------------------

WELL_FORMED = {
    "python": "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END",
    "julia": "# EVOLVE-BLOCK-START\nfunction f(x) end\n# EVOLVE-BLOCK-END",
    "cpp": "// EVOLVE-BLOCK-START\nint x;\n// EVOLVE-BLOCK-END",
    "cuda": "// EVOLVE-BLOCK-START\n__device__ int f();\n// EVOLVE-BLOCK-END",
    "rust": "// EVOLVE-BLOCK-START\nfn f() {}\n// EVOLVE-BLOCK-END",
    "swift": "// EVOLVE-BLOCK-START\nfunc f() {}\n// EVOLVE-BLOCK-END",
    "json": '// EVOLVE-BLOCK-START\n{"a": 1}\n// EVOLVE-BLOCK-END',
    "json5": "// EVOLVE-BLOCK-START\n{a: 1}\n// EVOLVE-BLOCK-END",
    "go": "// EVOLVE-BLOCK-START\nfunc f() {}\n// EVOLVE-BLOCK-END",
    "fortran": "! EVOLVE-BLOCK-START\ninteger :: x\n! EVOLVE-BLOCK-END",
    "markdown": "<!-- EVOLVE-BLOCK-START -->\nhello\n<!-- EVOLVE-BLOCK-END -->",
    "wolfram": "(* EVOLVE-BLOCK-START *)\nf[n_] := n\n(* EVOLVE-BLOCK-END *)",
}


@pytest.mark.parametrize("language,code", list(WELL_FORMED.items()))
def test_well_formed_passes(language, code):
    assert validate_evolve_markers(code, language) is None


# Aliases should normalise to canonical names.
@pytest.mark.parametrize("alias", ["py", "python3"])
def test_python_aliases(alias):
    assert validate_evolve_markers(WELL_FORMED["python"], alias) is None


@pytest.mark.parametrize("alias", ["wl", "wls", "mathematica", "Mathematica"])
def test_wolfram_aliases(alias):
    assert validate_evolve_markers(WELL_FORMED["wolfram"], alias) is None


@pytest.mark.parametrize("alias", ["c++", "cxx", "cc"])
def test_cpp_aliases(alias):
    assert validate_evolve_markers(WELL_FORMED["cpp"], alias) is None


def test_go_aliases():
    assert validate_evolve_markers(WELL_FORMED["go"], "golang") is None


# ---------------------------------------------------------------------------
# Missing-marker cases (should fail for any language)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", list(WELL_FORMED.keys()))
def test_missing_start_fails(language):
    code = WELL_FORMED[language].replace("EVOLVE-BLOCK-START", "REMOVED")
    err = validate_evolve_markers(code, language)
    assert err is not None
    assert "EVOLVE-BLOCK-START" in err


@pytest.mark.parametrize("language", list(WELL_FORMED.keys()))
def test_missing_end_fails(language):
    code = WELL_FORMED[language].replace("EVOLVE-BLOCK-END", "REMOVED")
    err = validate_evolve_markers(code, language)
    assert err is not None
    assert "EVOLVE-BLOCK-END" in err


# ---------------------------------------------------------------------------
# Wolfram-specific bugs — the actual reason this validator exists
# ---------------------------------------------------------------------------


def test_wolfram_markers_inside_single_comment_block():
    """The exact LLM-produced bug: both markers wrapped in one (* ... *) comment,
    so all the candidate code is silently commented out."""
    code = (
        "(* EVOLVE-BLOCK-START\n"
        "solve[] := With[{r = Range[n]}, EulerPhi[r] . Quotient[n, r]^2]\n"
        " EVOLVE-BLOCK-END *)\n"
    )
    err = validate_evolve_markers(code, "wolfram")
    assert err is not None
    # Neither marker line matches its strict canonical form: the START line
    # never closes its comment, the END line never opens one.
    assert "EVOLVE-BLOCK-START" in err
    assert "EVOLVE-BLOCK-END" in err


def test_wolfram_missing_close_on_start_marker():
    """LLM forgets the closing *) on the START marker."""
    code = "(* EVOLVE-BLOCK-START\n" "f[n_] := n\n" "(* EVOLVE-BLOCK-END *)\n"
    err = validate_evolve_markers(code, "wolfram")
    assert err is not None
    assert "malformed" in err.lower()


def test_wolfram_missing_open_on_end_marker():
    """LLM forgets the opening (* on the END marker."""
    code = "(* EVOLVE-BLOCK-START *)\n" "f[n_] := n\n" "EVOLVE-BLOCK-END *)\n"
    err = validate_evolve_markers(code, "wolfram")
    assert err is not None
    assert "malformed" in err.lower()


def test_wolfram_extra_text_on_marker_line():
    """Markers must occupy their own line; extra Wolfram code on the same
    line breaks the strict shape."""
    code = "(* EVOLVE-BLOCK-START *) f[n_] := n;\n" "(* EVOLVE-BLOCK-END *)\n"
    err = validate_evolve_markers(code, "wolfram")
    assert err is not None


def test_wolfram_inner_comments_not_inspected():
    """Validation is scoped to the marker lines: ordinary Wolfram comments
    in the body — including nested ones — are never inspected, so a
    well-formed candidate passes regardless of what comments it contains."""
    code = (
        "(* EVOLVE-BLOCK-START *)\n"
        "(* this is a normal Wolfram comment *)\n"
        "f[n_] := n\n"
        "(* another (* nested *) comment *)\n"
        "(* EVOLVE-BLOCK-END *)\n"
    )
    assert validate_evolve_markers(code, "wolfram") is None


def test_wolfram_block_comment_delim_in_string_not_rejected():
    """A candidate whose body contains a block-comment delimiter inside an
    unrelated string literal must NOT be rejected. The validator only
    inspects the marker lines, never balances comments across the file."""
    code = (
        "(* EVOLVE-BLOCK-START *)\n"
        'f[] := "a literal containing (* which is not a comment";\n'
        "(* EVOLVE-BLOCK-END *)\n"
    )
    assert validate_evolve_markers(code, "wolfram") is None


# ---------------------------------------------------------------------------
# Line-comment languages should remain lenient — they're not affected by
# the block-comment-smuggling failure mode, so trailing markers on a code
# line should NOT be rejected.
# ---------------------------------------------------------------------------


def test_python_trailing_marker_is_ok():
    """In Python, '   y = 2# EVOLVE-BLOCK-END' is still a valid line comment;
    must NOT be rejected (would over-reject the diff applier's empty-search
    insertion path)."""
    code = (
        "# EVOLVE-BLOCK-START\n"
        "def f():\n"
        "    x = 1\n"
        "    return x\n"
        "    y = 2# EVOLVE-BLOCK-END\n"
    )
    assert validate_evolve_markers(code, "python") is None


def test_cpp_trailing_marker_is_ok():
    code = "// EVOLVE-BLOCK-START\n" "int x = 1;// EVOLVE-BLOCK-END\n"
    assert validate_evolve_markers(code, "cpp") is None


def test_go_trailing_marker_is_ok():
    code = "// EVOLVE-BLOCK-START\n" "var x = 1 // EVOLVE-BLOCK-END\n"
    assert validate_evolve_markers(code, "go") is None


def test_fortran_trailing_marker_is_ok():
    """Fortran uses ``!`` line comments, so a trailing marker stays lenient
    like Python/C++ — not subject to the block-comment-balance check."""
    code = "! EVOLVE-BLOCK-START\n" "integer :: x  ! EVOLVE-BLOCK-END\n"
    assert validate_evolve_markers(code, "fortran") is None


# ---------------------------------------------------------------------------
# Unknown language — defer (no validation)
# ---------------------------------------------------------------------------


def test_unknown_language_defers():
    assert (
        validate_evolve_markers("EVOLVE-BLOCK-START\nx\nEVOLVE-BLOCK-END", "klingon")
        is None
    )


# ---------------------------------------------------------------------------
# Applier integration — bad Wolfram patches are rejected, original preserved
# ---------------------------------------------------------------------------


def test_apply_full_patch_rejects_wolfram_smuggled_markers(tmp_path):
    original = (
        "(* EVOLVE-BLOCK-START *)\n" "f[n_Integer] := n\n" "(* EVOLVE-BLOCK-END *)\n"
    )
    # Patch where the LLM has emitted markers inside a single comment block.
    bad_patch = (
        "```wolfram\n"
        "(* EVOLVE-BLOCK-START\n"
        "f[n_Integer] := n (n + 1) / 2\n"
        " EVOLVE-BLOCK-END *)\n"
        "```\n"
    )
    result = apply_full_patch(
        patch_str=bad_patch,
        original_str=original,
        patch_dir=tmp_path / "wolfram_bad",
        language="wolfram",
        verbose=False,
    )
    updated, num_applied, output_path, error, _, _ = result
    assert error is not None
    assert "EVOLVE-BLOCK" in error
    # The original content is preserved on rejection.
    assert updated == original
    assert num_applied == 0
    # No corrupted output file is written.
    assert output_path is None


def test_apply_full_patch_accepts_well_formed_wolfram(tmp_path):
    original = (
        "(* EVOLVE-BLOCK-START *)\n" "f[n_Integer] := n\n" "(* EVOLVE-BLOCK-END *)\n"
    )
    good_patch = (
        "```wolfram\n"
        "(* EVOLVE-BLOCK-START *)\n"
        "f[n_Integer] := n (n + 1) / 2\n"
        "(* EVOLVE-BLOCK-END *)\n"
        "```\n"
    )
    result = apply_full_patch(
        patch_str=good_patch,
        original_str=original,
        patch_dir=tmp_path / "wolfram_good",
        language="wolfram",
        verbose=False,
    )
    updated, num_applied, output_path, error, _, _ = result
    assert error is None
    assert num_applied == 1
    assert "n (n + 1) / 2" in updated


def test_apply_diff_patch_still_works_after_validation_hook(tmp_path):
    """Diff patches that operate inside well-formed markers should be
    completely unaffected by the new validation."""
    original = (
        "# EVOLVE-BLOCK-START\n"
        "def f(x):\n"
        "    return x + 1\n"
        "# EVOLVE-BLOCK-END\n"
    )
    diff_patch = (
        "<<<<<<< SEARCH\n"
        "    return x + 1\n"
        "=======\n"
        "    return x + 2\n"
        ">>>>>>> REPLACE\n"
    )
    result = apply_diff_patch(
        patch_str=diff_patch,
        original_str=original,
        patch_dir=tmp_path / "py_diff",
        language="python",
        verbose=False,
    )
    updated, num_applied, output_path, error, _, _ = result
    assert error is None
    assert num_applied == 1
    assert "return x + 2" in updated
