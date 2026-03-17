from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
    normalize_language,
)


def test_language_helpers_support_markdown():
    """Markdown is registered as a language with HTML-style comment markers."""
    assert get_language_extension("markdown") == "md"
    assert get_evolve_comment_prefix("markdown") == "<!--"

    fences = get_code_fence_languages("markdown")
    assert fences[0] == "markdown"
    assert "md" in fences


def test_language_helpers_support_md_alias():
    """The 'md' alias resolves to the canonical 'markdown' language."""
    assert normalize_language("md") == "markdown"
    assert get_language_extension("md") == "md"
    assert get_evolve_comment_prefix("md") == "<!--"

    fences = get_code_fence_languages("md")
    assert fences[0] == "md"
    assert "markdown" in fences


def test_apply_diff_patch_supports_markdown(tmp_path):
    """Diff patching works with HTML-comment EVOLVE-BLOCK markers."""
    original_content = """<!-- EVOLVE-BLOCK-START -->
## Step 1

Run the old command.
<!-- EVOLVE-BLOCK-END -->"""

    patch_content = """<!-- EVOLVE-BLOCK-START -->
<<<<<<< SEARCH
Run the old command.
=======
Run the new command.
>>>>>>> REPLACE
<!-- EVOLVE-BLOCK-END -->"""

    patch_dir = tmp_path / "markdown_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="markdown",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "Run the new command." in updated_content
    assert output_path == patch_dir / "main.md"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.md").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "Run the new command." in patch_txt


def test_apply_full_patch_supports_markdown(tmp_path):
    """Full patching works with markdown code fences."""
    original_content = """<!-- EVOLVE-BLOCK-START -->
## Step 1

Run the old command.
<!-- EVOLVE-BLOCK-END -->
"""

    patch_content = """```markdown
<!-- EVOLVE-BLOCK-START -->
## Step 1

Run the new command.
<!-- EVOLVE-BLOCK-END -->
```"""

    patch_dir = tmp_path / "markdown_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="markdown",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "Run the new command." in updated_content
    assert output_path == patch_dir / "main.md"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.md").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "Run the new command." in patch_txt


def test_apply_full_patch_supports_md_fence(tmp_path):
    """Full patching accepts the short 'md' fence tag."""
    original_content = """<!-- EVOLVE-BLOCK-START -->
# Title
Some content.
<!-- EVOLVE-BLOCK-END -->
"""

    patch_content = """```md
<!-- EVOLVE-BLOCK-START -->
# Title
Updated content.
<!-- EVOLVE-BLOCK-END -->
```"""

    patch_dir = tmp_path / "markdown_md_fence"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="markdown",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "Updated content." in updated_content


def test_clean_evolve_markers_strips_html_comments():
    """The marker stripper handles <!-- EVOLVE-BLOCK-START/END --> syntax."""
    from shinka.edit.apply_diff import _clean_evolve_markers

    text = """<!-- EVOLVE-BLOCK-START -->
<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE
<!-- EVOLVE-BLOCK-END -->"""

    cleaned = _clean_evolve_markers(text)
    assert "EVOLVE-BLOCK-START" not in cleaned
    assert "EVOLVE-BLOCK-END" not in cleaned
    assert "new text" in cleaned
