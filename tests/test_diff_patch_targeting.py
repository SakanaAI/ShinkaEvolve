"""SEARCH/REPLACE patches change the lines they name and nothing else."""

import pytest

from shinka.edit import apply_diff_patch
from shinka.edit.apply_diff import PatchError, apply_search_replace


def _patch(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def _apply(original: str, patch: str, language: str = "python"):
    updated, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch, original_str=original, language=language, verbose=False
    )
    return updated, num_applied, error


# -- whole-line matching ------------------------------------------------------


def test_search_does_not_match_inside_a_longer_line():
    original = "# EVOLVE-BLOCK-START\nidx = 10\nx = 1\n# EVOLVE-BLOCK-END\n"

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 2"))

    assert error is None
    assert num_applied == 1
    assert updated == "# EVOLVE-BLOCK-START\nidx = 10\nx = 2\n# EVOLVE-BLOCK-END\n"


def test_search_that_is_only_part_of_a_line_is_not_found():
    original = "# EVOLVE-BLOCK-START\nidx = 10\n# EVOLVE-BLOCK-END\n"

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 2"))

    assert num_applied == 0
    assert error is not None and "SEARCH text not found" in error
    assert updated == original


def test_multiline_search_must_end_at_a_line_end():
    original = "# EVOLVE-BLOCK-START\na = 1\nb = 22\na = 1\nb = 2\n# EVOLVE-BLOCK-END\n"

    updated, _, error = _apply(original, _patch("a = 1\nb = 2", "c = 3"))

    assert error is None
    assert updated == "# EVOLVE-BLOCK-START\na = 1\nb = 22\nc = 3\n# EVOLVE-BLOCK-END\n"


def test_reindented_search_replaces_the_line_it_matched():
    original = (
        "def helper():\n"
        "    value = 1\n"
        "# EVOLVE-BLOCK-START\n"
        "def solve():\n"
        "    if True:\n"
        "        value = 1\n"
        "    return value\n"
        "# EVOLVE-BLOCK-END\n"
    )

    updated, num_applied, error = _apply(original, _patch("value = 1", "value = 2"))

    assert error is None
    assert num_applied == 1
    assert updated == original.replace("        value = 1", "        value = 2")


def test_block_whose_first_line_lost_its_indentation_still_applies():
    original = (
        "# EVOLVE-BLOCK-START\n"
        "def solve(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    return total\n"
        "# EVOLVE-BLOCK-END\n"
    )
    patch = _patch(
        "for i in range(n):\n        total += i",
        "for i in range(n):\n        total += 2 * i",
    )

    updated, num_applied, error = _apply(original, patch)

    assert error is None
    assert num_applied == 1
    assert updated == original.replace("total += i", "total += 2 * i")


def test_first_line_without_indentation_does_not_match_mid_line():
    original = (
        "# EVOLVE-BLOCK-START\n"
        "    total = idx = 1\n"
        "    done = True\n"
        "# EVOLVE-BLOCK-END\n"
    )

    updated, num_applied, error = _apply(
        original, _patch("idx = 1\n    done = True", "idx = 2\n    done = False")
    )

    assert num_applied == 0
    assert error is not None and "SEARCH text not found" in error
    assert updated == original


# -- matches outside the editable region ---------------------------------------


def test_match_inside_the_block_wins_over_an_earlier_one_outside():
    original = "x = 1\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 2"))

    assert error is None
    assert num_applied == 1
    assert updated == "x = 1\n# EVOLVE-BLOCK-START\nx = 2\n# EVOLVE-BLOCK-END\n"


def test_match_only_outside_the_block_is_rejected():
    original = "x = 1\n# EVOLVE-BLOCK-START\ny = 1\n# EVOLVE-BLOCK-END\n"

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 2"))

    assert num_applied == 0
    assert error is not None
    assert "Attempted to edit outside EVOLVE-BLOCK regions" in error
    assert updated == original


def test_verbatim_immutable_code_is_not_swapped_for_an_indented_look_alike():
    original = (
        "def helper():\n"
        "    return 1\n"
        "# EVOLVE-BLOCK-START\n"
        "class Solver:\n"
        "    def helper():\n"
        "        return 1\n"
        "# EVOLVE-BLOCK-END\n"
    )
    patch = _patch("def helper():\n    return 1", "def helper():\n    return 2")

    updated, num_applied, error = _apply(original, patch)

    assert num_applied == 0
    assert error is not None
    assert "Attempted to edit outside EVOLVE-BLOCK regions" in error
    assert updated == original


# -- code on the same line as a marker -------------------------------------------


def test_code_before_an_end_marker_can_be_replaced():
    original = "# EVOLVE-BLOCK-START\ndef f(x):\n    return x  # EVOLVE-BLOCK-END\n"

    updated, num_applied, error = _apply(
        original, _patch("    return x", "    return x + 1")
    )

    assert error is None
    assert num_applied == 1
    assert updated == original.replace("return x ", "return x + 1 ")


def test_code_after_a_start_marker_can_be_replaced():
    original = "<!-- EVOLVE-BLOCK-START --><p>hi</p>\n<!-- EVOLVE-BLOCK-END -->\n"

    # Marker validation in apply_diff_patch rejects this layout for markdown,
    # so it is only reachable through the matcher itself.
    updated, num_applied = apply_search_replace(
        _patch("<p>hi</p>", "<p>bye</p>"), original
    )

    assert num_applied == 1
    assert updated == original.replace("hi", "bye")


def test_marker_line_itself_stays_out_of_reach():
    original = "# EVOLVE-BLOCK-START\nx = 1\ny = 2  # EVOLVE-BLOCK-END\ntail = 0\n"

    updated, num_applied, error = _apply(
        original, _patch("y = 2  # EVOLVE-BLOCK-END\ntail = 0", "y = 3\ntail = 1")
    )

    assert num_applied == 0
    assert error is not None
    assert updated == original


# -- trailing comments --------------------------------------------------------


def _block(*lines: str, comment: str = "#") -> str:
    body = "\n".join(lines)
    return f"{comment} EVOLVE-BLOCK-START\n{body}\n{comment} EVOLVE-BLOCK-END\n"


def test_search_without_the_trailing_comment_of_its_line_applies():
    original = _block("lr = 0.01  # tuned", "steps = 10")

    updated, num_applied, error = _apply(original, _patch("lr = 0.01", "lr = 0.02"))

    assert error is None
    assert num_applied == 1
    assert updated == _block("lr = 0.02  # tuned", "steps = 10")


def test_commented_line_is_preferred_over_a_look_alike_at_another_indentation():
    original = _block(
        "def f():",
        "    best = 0.0  # running best",
        "    return best",
        "class G:",
        "    def g(self):",
        "        best = 0.0",
        "        return best",
    )

    updated, _, error = _apply(original, _patch("    best = 0.0", "    best = 1.0"))

    assert error is None
    assert updated == original.replace("best = 0.0  # running", "best = 1.0  # running")


def test_whole_line_match_is_preferred_over_a_commented_line():
    original = _block("x = 1  # first", "x = 1")

    updated, _, error = _apply(original, _patch("x = 1", "x = 2"))

    assert error is None
    assert updated == _block("x = 1  # first", "x = 2")


def test_comments_inside_a_block_are_replaced_with_their_lines():
    original = _block("a = 1  # one", "b = 2", "c = 3  # three")

    updated, _, error = _apply(
        original, _patch("a = 1\nb = 2\nc = 3", "a = 10\nc = 30")
    )

    assert error is None
    assert updated == _block("a = 10", "c = 30  # three")


@pytest.mark.parametrize(
    ("line", "search"),
    [
        # Code, not a comment: `//` divides in Python and `#` sits in a string.
        ("x = a // 2", "x = a"),
        ("x = a + b", "x = a"),
        ("# x = 1", ""),
        ("    # only a comment", "    "),
    ],
)
def test_search_line_that_ends_before_code_is_not_found(line, search):
    original = _block("y = 0", line, "z = 9")
    patch = _patch(f"y = 0\n{search}\nz = 9", "y = 1")

    updated, num_applied, error = _apply(original, patch)

    assert num_applied == 0
    assert error is not None and "SEARCH text not found" in error
    assert updated == original


def test_trailing_comment_follows_the_comment_style_of_the_file():
    original = _block("int lr = 1;  // tuned", "int q = a # b;", comment="//")

    updated, _, error = _apply(original, _patch("int lr = 1;", "int lr = 2;"), "cpp")
    assert error is None
    assert updated == original.replace("int lr = 1;", "int lr = 2;")

    _, num_applied, error = _apply(original, _patch("int q = a", "int q = c"), "cpp")
    assert num_applied == 0
    assert error is not None and "SEARCH text not found" in error


# -- marker-only SEARCH -------------------------------------------------------

_TWO_BLOCKS = (
    "# EVOLVE-BLOCK-START\n"
    "a = 1\n"
    "# EVOLVE-BLOCK-END\n"
    "mid = 0\n"
    "# EVOLVE-BLOCK-START\n"
    "b = 2\n"
    "# EVOLVE-BLOCK-END\n"
)


@pytest.mark.parametrize(
    "search",
    [
        "# EVOLVE-BLOCK-START",
        "# EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END",
        "    # EVOLVE-BLOCK-START   ",
    ],
)
def test_marker_only_search_is_rejected(search):
    patch = _patch(search, "# EVOLVE-BLOCK-START\nimport os")

    updated, num_applied, error = _apply(_TWO_BLOCKS, patch)

    assert num_applied == 0
    assert error is not None
    assert "SEARCH block contains only EVOLVE-BLOCK markers" in error
    assert updated == _TWO_BLOCKS


def test_marker_only_search_is_rejected_for_other_comment_styles():
    original = "// EVOLVE-BLOCK-START\nint a = 1;\n// EVOLVE-BLOCK-END\n"
    patch = _patch("// EVOLVE-BLOCK-START", "// EVOLVE-BLOCK-START\nint b = 2;")

    updated, num_applied, error = _apply(original, patch, language="cpp")

    assert num_applied == 0
    assert error is not None and "only EVOLVE-BLOCK markers" in error
    assert updated == original


def test_apply_search_replace_rejects_marker_only_search():
    with pytest.raises(PatchError, match="only EVOLVE-BLOCK markers"):
        apply_search_replace(_patch("# EVOLVE-BLOCK-START", "import os"), _TWO_BLOCKS)


_ONE_BLOCK = "import os\n# EVOLVE-BLOCK-START\na = 1\n# EVOLVE-BLOCK-END\nprint(a)\n"


@pytest.mark.parametrize("replace", ["b = 2\n# EVOLVE-BLOCK-END", "b = 2"])
def test_end_marker_search_appends_to_the_only_block(replace):
    """With one editable region its END marker names the place an append goes."""
    patch = _patch("# EVOLVE-BLOCK-END", replace)

    updated, num_applied, error = _apply(_ONE_BLOCK, patch)

    assert error is None
    assert num_applied == 1
    appended = replace.split("\n")[0]
    assert updated == _ONE_BLOCK.replace("a = 1\n", f"a = 1\n{appended}\n")


@pytest.mark.parametrize(
    "search", ["# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END"]
)
def test_start_marker_search_is_rejected_in_a_single_block_file(search):
    updated, num_applied, error = _apply(_ONE_BLOCK, _patch(search, "b = 2"))

    assert num_applied == 0
    assert error is not None and "only EVOLVE-BLOCK markers" in error
    assert updated == _ONE_BLOCK


def test_marker_used_as_context_still_matches_the_code_below_it():
    patch = _patch("# EVOLVE-BLOCK-START\nb = 2", "# EVOLVE-BLOCK-START\nb = 3")

    updated, num_applied, error = _apply(_TWO_BLOCKS, patch)

    assert error is None
    assert num_applied == 1
    assert updated == _TWO_BLOCKS.replace("b = 2", "b = 3")


def test_empty_search_still_appends_to_the_last_block():
    updated, num_applied, error = _apply(_TWO_BLOCKS, _patch("", "c = 3"))

    assert error is None
    assert num_applied == 1
    assert updated == _TWO_BLOCKS.replace("b = 2\n", "b = 2\nc = 3\n")


# -- immutable regions stay byte for byte --------------------------------------

_IMMUTABLE_HEAD = 'DOC = """keep  \ntrailing\x0cform feed\x1cand more"""\t\n\n'
_IMMUTABLE_TAIL = 'TAIL = "also  "  \n\n\n'


def _with_immutable(block: str) -> str:
    return (
        f"{_IMMUTABLE_HEAD}# EVOLVE-BLOCK-START\n{block}"
        f"# EVOLVE-BLOCK-END\n{_IMMUTABLE_TAIL}"
    )


def test_immutable_text_is_not_rewritten():
    original = _with_immutable("x = 1\n")

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 2"))

    assert error is None
    assert num_applied == 1
    assert updated == _with_immutable("x = 2\n")


def test_rejected_patch_returns_the_original_text():
    original = _with_immutable("x = 1\n")

    updated, num_applied, error = _apply(original, _patch("y = 1", "y = 2"))

    assert num_applied == 0
    assert error is not None
    assert updated == original


def test_patch_without_changes_keeps_the_file_identical():
    original = _with_immutable("x = 1\n")

    updated, num_applied, error = _apply(original, _patch("x = 1", "x = 1"))

    assert error is None
    assert num_applied == 1
    assert updated == original


def test_trailing_whitespace_is_ignored_inside_the_block():
    original = _with_immutable("x = 1   \ny = 2\t\n")

    updated, num_applied, error = _apply(
        original, _patch("x = 1  \ny = 2", "x = 3  \ny = 4")
    )

    assert error is None
    assert num_applied == 1
    assert updated == _with_immutable("x = 3\ny = 4\n")


def test_form_feed_in_editable_string_is_not_a_line_break():
    original = _with_immutable('a = "one\x0ctwo"\nb = 1\n')

    updated, _, error = _apply(original, _patch("b = 1", 'b = "three\x0cfour"'))

    assert error is None
    assert updated == _with_immutable('a = "one\x0ctwo"\nb = "three\x0cfour"\n')


def test_indented_end_marker_keeps_its_indentation():
    original = (
        "class Solver:\n"
        "    # EVOLVE-BLOCK-START\n"
        "    x = 1   \n"
        "    # EVOLVE-BLOCK-END\n"
        "    y = 2\n"
    )

    updated, num_applied, error = _apply(original, _patch("    x = 1", "    x = 2"))

    assert error is None
    assert num_applied == 1
    assert updated == original.replace("    x = 1   \n", "    x = 2\n")


def test_written_file_keeps_the_final_newline(tmp_path):
    original = _with_immutable("x = 1\n")

    result = apply_diff_patch(
        patch_str=_patch("x = 1", "x = 2"),
        original_str=original,
        patch_dir=tmp_path,
        verbose=False,
    )

    assert result[2] is not None
    assert result[2].read_bytes() == _with_immutable("x = 2\n").encode()
    assert (tmp_path / "original.py").read_bytes() == original.encode()
