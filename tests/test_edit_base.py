from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.edit.apply_diff import (
    _find_indented_match,
    _apply_indentation_to_replace,
    _strip_trailing_whitespace,
)


patch_str = """
<<<<<<< SEARCH
def run_experiment(train_dataset, device):
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
    return model
=======
THIS IS A TEST
>>>>>>> REPLACE

<<<<<<< SEARCH
THIS IS A TEST
=======
THIS IS A TEST PART 2
>>>>>>> REPLACE
"""

# Trailing newline is preserved from the source file (tests/file.py ends with
# one); _strip_trailing_whitespace no longer drops it.
new_str = """# EVOLVE-BLOCK-START
THIS IS A TEST PART 2


# EVOLVE-BLOCK-END
"""


def test_edit():
    result = apply_diff_patch(
        original_path="tests/file.py",
        patch_str=patch_str,
        patch_dir=None,
    )
    updated_str, num_applied, output_path, error, patch_txt, diff_path = result
    assert updated_str == new_str
    assert num_applied == 2
    assert output_path is None
    assert error is None


def test_apply_full_patch_single_evolve_block():
    """Test apply_full_patch with single EVOLVE-BLOCK region."""
    original_content = """# Immutable header
import os

# EVOLVE-BLOCK-START
def old_function():
    return "old"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == "__main__":
    pass
"""

    patch_content = """```python
# Immutable header
import os

# EVOLVE-BLOCK-START
def new_function():
    return "new"
    
def another_function():
    return "another"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == "__main__":
    pass
```"""

    expected_result = """# Immutable header
import os

# EVOLVE-BLOCK-START
def new_function():
    return "new"
    
def another_function():
    return "another"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == "__main__":
    pass
"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert output_path is None
    assert error is None
    # Now we can directly check the updated content
    assert updated_content.strip() == expected_result.strip()


def test_apply_full_patch_with_evolve_blocks_in_patch():
    """Test apply_full_patch when patch contains EVOLVE-BLOCK markers."""
    original_content = """# Header
# EVOLVE-BLOCK-START
def old_func1():
    pass
# EVOLVE-BLOCK-END

# Middle section
# EVOLVE-BLOCK-START
def old_func2():
    pass
# EVOLVE-BLOCK-END
# Footer
"""

    patch_content = """```python
# Header
# EVOLVE-BLOCK-START
def new_func1():
    return 1
# EVOLVE-BLOCK-END

# Middle section
# EVOLVE-BLOCK-START
def new_func2():
    return 2
# EVOLVE-BLOCK-END
# Footer
```"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    # Should have replaced both evolve blocks with new content


def test_apply_full_patch_full_file_without_markers_extracts_block_only():
    """Full-file patch without EVOLVE markers should not copy immutable code
    into the evolve block; only the block payload is replaced."""
    original_content = """# Header line\n# EVOLVE-BLOCK-START\nold_line()\n# EVOLVE-BLOCK-END\n# Footer line\n"""

    # Patch is the entire file content but with the EVOLVE markers omitted.
    patch_content = """```python
new_line()
another_new_line()
```"""

    expected = """# Header line
# EVOLVE-BLOCK-START
new_line()
another_new_line()
# EVOLVE-BLOCK-END
# Footer line
"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert updated_content == expected

def test_apply_full_patch_patch_with_start_marker_only():
    """Patch has only START marker; original has both markers."""
    original_content = """# Header line
# EVOLVE-BLOCK-START
old_line()
# EVOLVE-BLOCK-END
# Footer line
"""

    patch_content = """```python
# Header line
# EVOLVE-BLOCK-START
new_line()
# Footer line
```"""

    expected = """# Header line
# EVOLVE-BLOCK-START
new_line()
# EVOLVE-BLOCK-END
# Footer line
"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert updated_content == expected


def test_apply_full_patch_patch_with_end_marker_only():
    """Patch has only END marker; original has both markers."""
    original_content = """# Header line
# EVOLVE-BLOCK-START
old_line()
# EVOLVE-BLOCK-END
# Footer line
"""

    patch_content = """```python
# Header line
new_line()
# EVOLVE-BLOCK-END
# Footer line
```"""

    expected = """# Header line
# EVOLVE-BLOCK-START
new_line()
# EVOLVE-BLOCK-END
# Footer line
"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert updated_content == expected


def test_apply_full_patch_no_evolve_blocks():
    """Test apply_full_patch with no EVOLVE-BLOCK regions - should error."""
    original_content = """# Just regular code
def function():
    return "no evolve blocks"
"""

    patch_content = """```python
def new_function():
    return "new"
```"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error == "No EVOLVE-BLOCK regions found in original content"
    assert output_path is None
    assert updated_content == original_content  # Should return original content


def test_apply_full_patch_multiple_evolve_blocks_ambiguous():
    """Test apply_full_patch with multiple EVOLVE-BLOCK regions."""
    original_content = """# EVOLVE-BLOCK-START
def func1():
    pass
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def func2():
    pass
# EVOLVE-BLOCK-END
"""

    patch_content = """```python
def new_function():
    return "ambiguous which block to replace"
```"""

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "Multiple EVOLVE-BLOCK regions found" in error
    assert "doesn't specify which to replace" in error
    assert output_path is None
    assert updated_content == original_content  # Should return original content


def test_apply_full_patch_patch_with_single_marker_ambiguous_multiple_regions():
    """Single marker in patch is ambiguous when original has multiple regions."""
    original_content = """# Header
# EVOLVE-BLOCK-START
func1()
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
func2()
# EVOLVE-BLOCK-END
# Footer
"""

    # Patch includes only START marker
    patch_content = """```python
# Header
# EVOLVE-BLOCK-START
new_code()
# Footer
```"""

    updated_content, num_applied, output_path, error, patch_txt, diff_path = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert num_applied == 0
    assert error is not None
    assert "only one EVOLVE-BLOCK marker" in error


def test_apply_full_patch_invalid_extraction():
    """Test apply_full_patch with invalid code extraction."""
    original_content = """# EVOLVE-BLOCK-START
def old_func():
    pass
# EVOLVE-BLOCK-END
"""

    # No proper language fences - extract_between will return "none"
    patch_content = "def new_function(): return 'no fences'"

    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    # extract_between returns "none" when it can't find the pattern
    # After our fix, this should be treated as an error
    assert num_applied == 0
    assert error == "Could not extract code from patch string"
    assert output_path is None
    assert updated_content == original_content  # Should return original content


def test_apply_full_patch_with_patch_dir():
    """Test apply_full_patch with patch directory specified."""
    import tempfile
    from pathlib import Path

    original_content = """# EVOLVE-BLOCK-START
def old_function():
    return "old"
# EVOLVE-BLOCK-END
"""

    patch_content = """```python
def new_function():
    return "new"
```"""

    with tempfile.TemporaryDirectory() as temp_dir:
        patch_dir = Path(temp_dir) / "test_patch"

        result = apply_full_patch(
            patch_str=patch_content,
            original_str=original_content,
            patch_dir=str(patch_dir),
            language="python",
            verbose=False,
        )
        updated_content, num_applied, output_path, error, patch_txt, diff_path = result

        assert num_applied == 1
        assert error is None
        assert output_path is not None
        assert output_path.exists()
        assert diff_path is not None
        assert diff_path.exists()

        # Check that files were created
        assert (patch_dir / "rewrite.txt").exists()
        assert (patch_dir / "original.py").exists()
        assert (patch_dir / "main.py").exists()
        assert (patch_dir / "edit.diff").exists()

        # Verify the updated content matches what's in the file
        file_content = output_path.read_text("utf-8")
        assert file_content == updated_content


# ============================================================================
# Tests for Indentation Correction Functionality
# ============================================================================


def test_find_indented_match_exact_match():
    """Test _find_indented_match when exact match is found."""
    original = """def function():
    x = 1
    y = 2
    return x + y"""
    search = "x = 1"
    matched, pos = _find_indented_match(search, original)

    assert matched == search
    assert pos != -1
    assert original[pos : pos + len(matched)] == matched


def test_find_indented_match_needs_indentation():
    """Test _find_indented_match when indentation correction is needed."""
    original = """def function():
    x = 1
    y = 2
    return x + y"""

    # Search text without proper indentation
    search = "x = 1\ny = 2"
    matched, pos = _find_indented_match(search, original)

    expected = "    x = 1\n    y = 2"
    assert matched == expected
    assert pos != -1
    assert original[pos : pos + len(matched)] == matched


def test_find_indented_match_multiline_with_relative_indentation():
    """Test _find_indented_match with multiline blocks having relative indentation."""
    original = """def function():
    if True:
        x = 1
        if nested:
            y = 2
    return x + y"""

    # Search text without proper base indentation but with relative indentation
    search = """if True:
    x = 1
    if nested:
        y = 2"""

    matched, pos = _find_indented_match(search, original)

    expected = """    if True:
        x = 1
        if nested:
            y = 2"""
    assert matched == expected
    assert pos != -1


def test_find_indented_match_not_found():
    """Test _find_indented_match when text is not found."""
    original = """def function():
    x = 1
    return x"""

    search = "z = 3"
    matched, pos = _find_indented_match(search, original)

    assert matched == ""
    assert pos == -1


def test_find_indented_match_empty_search():
    """Test _find_indented_match with empty search text."""
    original = "def function():\n    pass"
    search = ""

    matched, pos = _find_indented_match(search, original)
    assert matched == ""
    assert pos == -1


def test_apply_indentation_to_replace():
    """Test _apply_indentation_to_replace function."""
    replace_text = """x = 10
if x > 5:
    print("big")
else:
    print("small")"""

    indent_str = "    "  # 4 spaces
    result = _apply_indentation_to_replace(replace_text, indent_str)

    expected = """    x = 10
    if x > 5:
        print("big")
    else:
        print("small")"""

    assert result == expected


def test_apply_indentation_to_replace_empty_lines():
    """Test _apply_indentation_to_replace with empty lines."""
    replace_text = """x = 1

y = 2"""

    indent_str = "    "
    result = _apply_indentation_to_replace(replace_text, indent_str)

    expected = """    x = 1

    y = 2"""

    assert result == expected


def test_strip_trailing_whitespace():
    """Test _strip_trailing_whitespace function."""
    # Create text with trailing whitespace programmatically to avoid linting issues
    text_with_trailing = "line1   \nline2\t\nline3\nline4 \t "

    result = _strip_trailing_whitespace(text_with_trailing)
    expected = "line1\nline2\nline3\nline4"

    assert result == expected


# ============================================================================
# Integration Tests for Indentation Correction in apply_diff_patch
# ============================================================================


def test_indentation_correction_in_patch():
    """Test that apply_diff_patch correctly handles indentation mismatches."""
    original_content = """# EVOLVE-BLOCK-START
def calculate():
    centers = compute_centers()
    radius = get_radius()
    area = math.pi * radius ** 2
    return area
# EVOLVE-BLOCK-END"""

    # Patch with incorrect indentation
    patch_str = """<<<<<<< SEARCH
centers = compute_centers()
radius = get_radius()
=======
centers = compute_new_centers()
radius = get_new_radius()
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "compute_new_centers()" in updated_content
    assert "get_new_radius()" in updated_content
    # Verify indentation is preserved
    assert "    centers = compute_new_centers()" in updated_content


def test_indentation_correction_multiline_patch():
    """Test indentation correction with multiline search/replace blocks."""
    original_content = """# EVOLVE-BLOCK-START
def process_data():
    if condition:
        data = load_data()
        result = process(data)
        return result
    return None
# EVOLVE-BLOCK-END"""

    # Patch with no indentation
    patch_str = """<<<<<<< SEARCH
if condition:
    data = load_data()
    result = process(data)
    return result
=======
if new_condition:
    data = load_new_data()
    result = new_process(data)
    return enhanced_result
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "new_condition" in updated_content
    assert "load_new_data()" in updated_content
    # Verify proper indentation is applied
    assert "    if new_condition:" in updated_content
    assert "        data = load_new_data()" in updated_content


def test_indentation_correction_with_trailing_whitespace():
    """Test that indentation correction works with trailing whitespace."""
    # Create content with trailing whitespace programmatically
    original_content = """# EVOLVE-BLOCK-START
def func():
    x = 1
    y = 2
    return x + y
# EVOLVE-BLOCK-END"""

    # Patch with trailing whitespace and incorrect indentation
    patch_str = """<<<<<<< SEARCH
x = 1
y = 2
=======
x = 10
y = 20
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "x = 10" in updated_content
    assert "y = 20" in updated_content
    # Verify trailing whitespace is stripped
    lines = updated_content.split("\n")
    for line in lines:
        assert line == line.rstrip(), f"Line has trailing whitespace: {repr(line)}"


def test_indentation_correction_fails_gracefully():
    """Test that indentation correction fails gracefully when match cannot be found."""
    original_content = """# EVOLVE-BLOCK-START
def func():
    x = 1
    y = 2
    return x + y
# EVOLVE-BLOCK-END"""

    # Patch with text that doesn't exist
    patch_str = """<<<<<<< SEARCH
z = 3
w = 4
=======
z = 30
w = 40
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "SEARCH text not found" in error
    assert updated_content == original_content  # Should remain unchanged


def test_mixed_indentation_styles():
    """Test handling of mixed indentation styles (spaces and tabs)."""
    original_content = """# EVOLVE-BLOCK-START
def func():
\tx = 1  # Tab indented
\ty = 2  # Tab indented
\treturn x + y
# EVOLVE-BLOCK-END"""

    # Search with space indentation (should match tab indented lines)
    patch_str = """<<<<<<< SEARCH
x = 1  # Tab indented
y = 2  # Tab indented
=======
x = 10
y = 20
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "x = 10" in updated_content
    # Verify original tab indentation is preserved
    assert "\tx = 10" in updated_content
    assert "\ty = 20" in updated_content


def test_indentation_with_empty_lines_in_search():
    """Test indentation correction with empty lines in search block."""
    original_content = """# EVOLVE-BLOCK-START
def func():
    x = 1
    
    y = 2
    return x + y
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
x = 1

y = 2
=======
x = 10

y = 20
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "    x = 10" in updated_content
    assert "    y = 20" in updated_content


def test_indentation_correction_preserves_mutable_regions():
    """Test that indentation correction respects EVOLVE-BLOCK boundaries."""
    original_content = """# Immutable section
def immutable_func():
    x = 1
    return x

# EVOLVE-BLOCK-START
def mutable_func():
    y = 2
    return y
# EVOLVE-BLOCK-END

# Another immutable section
def another_immutable():
    z = 3
    return z"""

    # Try to patch something in immutable region (should fail)
    patch_str = """<<<<<<< SEARCH
x = 1
=======
x = 100
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "outside EVOLVE-BLOCK" in error


def test_insertion_with_indentation():
    """Test insertion (empty search) with proper indentation context."""
    original_content = """# EVOLVE-BLOCK-START
def func():
    x = 1
    return x
# EVOLVE-BLOCK-END"""

    # Empty search = insertion at end of mutable region
    patch_str = """<<<<<<< SEARCH

=======
    # New comment
    y = 2
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 1
    assert error is None
    assert "# New comment" in updated_content
    assert "y = 2" in updated_content


# ============================================================================
# Tests for Enhanced Error Messages
# ============================================================================


def test_enhanced_search_not_found_error():
    """Test that search not found errors provide helpful suggestions."""
    original_content = """# EVOLVE-BLOCK-START
def calculate():
    centers = compute_centers()
    radius = get_radius()
    area = math.pi * radius ** 2
    return area
# EVOLVE-BLOCK-END"""

    # Search for similar but not exact text
    patch_str = """<<<<<<< SEARCH
centers = compute_center()
=======
centers = compute_new_centers()
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "SEARCH text not found" in error


def test_enhanced_evolve_block_violation_error():
    """Test that EVOLVE-BLOCK violation errors show context and suggestions."""
    original_content = """# Immutable header
import os
import sys

# EVOLVE-BLOCK-START
def mutable_function():
    return "editable"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == "__main__":
    main()"""

    # Try to edit immutable code
    patch_str = """<<<<<<< SEARCH
import os
=======
import os
import json
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "Attempted to edit outside EVOLVE-BLOCK regions" in error
    assert "Context around found text:" in error
    assert "Available editable regions" in error
    assert "Line" in error  # Should show line numbers in context
    assert "Suggestions:" in error


def test_enhanced_no_evolve_block_error():
    """Test error message when no EVOLVE-BLOCK regions exist."""
    original_content = """def regular_function():
    return "no evolve blocks here"

if __name__ == "__main__":
    print("Hello world")"""

    # Try to insert into file with no EVOLVE-BLOCK
    patch_str = """<<<<<<< SEARCH

=======
# New comment
new_var = 42
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None
    assert "Cannot perform insertion: No EVOLVE-BLOCK regions found" in error
    assert "Current file structure:" in error
    assert "Expected format:" in error
    assert "EVOLVE-BLOCK-START" in error
    assert "Suggestions:" in error


def test_enhanced_error_with_multiline_search():
    """Test enhanced error messages with multiline search blocks."""
    original_content = """# EVOLVE-BLOCK-START
def process():
    data = load_data()
    result = transform(data)
    return result
# EVOLVE-BLOCK-END"""

    # Search for multiline block with typo
    patch_str = """<<<<<<< SEARCH
data = load_data()
result = transform_data(data)
return result
=======
data = load_new_data()
result = new_transform(data)
return result
>>>>>>> REPLACE"""

    result = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert num_applied == 0
    assert error is not None


# ============================================================================
# Regression tests: silent-corruption bugs in the diff engine
# ============================================================================


def test_search_does_not_match_midline_substring():
    """A SEARCH block must not rewrite a mid-token substring of another line.

    Regression for the substring-match corruption: SEARCH ``xval = 10`` used to
    match inside ``maxval = 100`` via ``str.find``, silently producing
    ``maxval = 9990`` and reporting success.
    """
    original_content = """# EVOLVE-BLOCK-START
maxval = 100
result = 5
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
xval = 10
=======
xval = 999
>>>>>>> REPLACE"""

    updated_content, num_applied, output_path, error, patch_txt, diff_path = (
        apply_diff_patch(
            patch_str=patch_str,
            original_str=original_content,
            language="python",
            verbose=False,
        )
    )

    # The corrupting edit must be rejected, not silently applied.
    assert num_applied == 0
    assert error is not None
    assert "maxval = 9990" not in updated_content
    assert "maxval = 100" in updated_content


def test_line_aligned_match_still_allows_indentation_offset():
    """A search that differs only by leading indentation still matches."""
    original_content = """# EVOLVE-BLOCK-START
def f():
    x = 1
    return x
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert error is None
    assert num_applied == 1
    assert "x = 2" in updated_content


def test_deletion_hunk_is_applied_not_silently_dropped():
    """An empty-REPLACE (deletion) hunk must actually delete, not vanish.

    Regression for the parser dropping deletion hunks (empty REPLACE body)
    while the surrounding patch still reported success.
    """
    original_content = """# EVOLVE-BLOCK-START
a = 1
b = 2
c = 3
# EVOLVE-BLOCK-END"""

    # Second hunk deletes ``b = 2`` (empty REPLACE body, no blank line).
    patch_str = """<<<<<<< SEARCH
a = 1
=======
a = 111
>>>>>>> REPLACE

<<<<<<< SEARCH
b = 2
=======
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert error is None
    assert num_applied == 2
    assert "a = 111" in updated_content
    assert "b = 2" not in updated_content
    assert "c = 3" in updated_content


def test_unparseable_hunk_is_reported_not_dropped():
    """A SEARCH header with no valid body must raise, not report partial success."""
    original_content = """# EVOLVE-BLOCK-START
a = 1
b = 2
# EVOLVE-BLOCK-END"""

    # Second header is malformed (missing '=======' divider).
    patch_str = """<<<<<<< SEARCH
a = 1
=======
a = 111
>>>>>>> REPLACE

<<<<<<< SEARCH
b = 2
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert num_applied == 0
    assert error is not None
    assert updated_content == _strip_trailing_whitespace(original_content)


def test_ambiguous_search_in_editable_region_is_rejected():
    """Non-unique SEARCH inside editable regions must be rejected, not guessed."""
    original_content = """# EVOLVE-BLOCK-START
x = 1
y = 1
x = 1
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert num_applied == 0
    assert error is not None
    assert "mbiguous" in error


def test_match_prefers_editable_over_earlier_immutable_occurrence():
    """An identical line in an earlier immutable region must not block a valid edit."""
    original_content = """target = 1
# EVOLVE-BLOCK-START
target = 1
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
target = 1
=======
target = 2
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert error is None
    assert num_applied == 1
    # The immutable occurrence is untouched; the editable one is changed.
    assert updated_content.splitlines()[0] == "target = 1"
    assert "target = 2" in updated_content


def test_search_marker_literal_in_body_does_not_false_reject():
    """A valid hunk whose body contains the literal '<<<<<<< SEARCH' must apply.

    Regression: the dropped-hunk guard counted SEARCH markers across the whole
    patch (including inside bodies), so a single valid hunk whose REPLACE body
    mentioned the marker literally was falsely rejected as malformed.
    """
    original_content = """# EVOLVE-BLOCK-START
    pass
# EVOLVE-BLOCK-END"""

    patch_str = """<<<<<<< SEARCH
    pass
=======
    # Example header looks like: <<<<<<< SEARCH (do not use)
    return 1
>>>>>>> REPLACE"""

    updated_content, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch_str,
        original_str=original_content,
        language="python",
        verbose=False,
    )

    assert error is None
    assert num_applied == 1
    assert "return 1" in updated_content
    assert "pass" not in updated_content.replace("# Example", "")
