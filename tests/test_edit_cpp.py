import pytest

from shinka.edit import apply_full_patch
from shinka.llm.llm import extract_between
from shinka.utils.languages import get_code_fence_languages

CPP_ORIGINAL = """#include <cstdio>
// EVOLVE-BLOCK-START
int solve() { return 1; }
// EVOLVE-BLOCK-END
int main() { return solve(); }
"""

CPP_REWRITE = """#include <cstdio>
// EVOLVE-BLOCK-START
int solve() { return 2; }
// EVOLVE-BLOCK-END
int main() { return solve(); }
"""


@pytest.mark.parametrize("fence", ["c++", "cpp", "cxx", "cc"])
def test_extract_between_matches_cpp_fence_tags_literally(fence):
    """A fence tag is literal text; ```c++ must not compile to a possessive
    quantifier that matches only the ``c`` and leaves ``++`` in the code."""
    content = (
        f"Here is the program:\n```{fence}\nint solve() {{ return 2; }}\n```\nDone."
    )
    assert (
        extract_between(content, f"```{fence}", "```", False)
        == "int solve() { return 2; }"
    )


def test_cpp_aliases_are_all_accepted_fence_tags():
    assert set(get_code_fence_languages("c++")) == {"c++", "cpp", "cc", "cxx"}


@pytest.mark.parametrize(
    ("language", "fence"),
    [
        ("c++", "cpp"),  # the canonical fence under the alias spelling of the language
        ("c++", "c++"),
        ("cpp", "c++"),
        ("cpp", "cxx"),
        ("c++", "cxx"),
    ],
)
def test_apply_full_patch_cpp_rewrite_is_not_corrupted(language, fence):
    """A model that returns just the block payload (no EVOLVE markers) is the
    shape where a mis-parsed fence corrupts the candidate: whatever the fence
    regex left behind lands inside the EVOLVE block. Every (language, fence)
    pair the language table accepts must yield the payload verbatim."""
    patch_content = f"```{fence}\nint solve() {{ return 2; }}\n```"

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=CPP_ORIGINAL,
        language=language,
        verbose=False,
    )
    updated_content, num_applied, _output_path, error, _patch_txt, _diff_path = result

    assert error is None
    assert num_applied == 1
    assert updated_content == CPP_REWRITE
