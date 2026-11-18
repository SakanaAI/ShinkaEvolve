import re
from pathlib import Path
import difflib
import logging
from typing import Iterator, NamedTuple, Union, Optional, List, Tuple

from shinka.utils.languages import get_language_extension
from .marker_validation import validate_evolve_markers

logger = logging.getLogger(__name__)

PATCH_PATTERN = re.compile(
    r"<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*REPLACE\s*",
    re.DOTALL,
)


EVOLVE_START = re.compile(
    r"(?:#|//|!|<!--|\(\*)?[^\S\r\n]*EVOLVE-BLOCK-START[^\S\r\n]*(?:-->|\*\))?"
)
EVOLVE_END = re.compile(
    r"(?:#|//|!|<!--|\(\*)?[^\S\r\n]*EVOLVE-BLOCK-END[^\S\r\n]*(?:-->|\*\))?"
)


def _mutable_ranges(text: str) -> list[tuple[int, int]]:
    """Return index ranges that are legal to edit.

    Handles both sequential and nested EVOLVE-BLOCK markers correctly by
    processing all markers in position order.
    """
    # Collect all markers with their positions and types
    markers = []
    for m in EVOLVE_START.finditer(text):
        markers.append((m.end(), "start"))  # mutable starts *after* the START line
    for m in EVOLVE_END.finditer(text):
        markers.append((m.start(), "end"))  # mutable ends *before* END line

    # Sort by position to handle both sequential and nested blocks
    markers.sort(key=lambda x: x[0])

    # Process in order using a stack for nesting support
    spans = []
    stack = []
    for pos, marker_type in markers:
        if marker_type == "start":
            stack.append(pos)
        elif marker_type == "end" and stack:
            start = stack.pop()
            spans.append((start, pos))

    return spans


def _inside(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    """True if span is fully contained in one of the ranges."""
    return any(span[0] >= a and span[1] <= b for a, b in ranges)


def _strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line in the text.

    Lines end at a newline only. ``str.splitlines`` also breaks on form feeds
    and other separators, which are ordinary characters inside a string
    literal and must survive.
    """
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _normalize_mutable_whitespace(text: str) -> str:
    """Strip trailing whitespace from the lines inside EVOLVE regions.

    SEARCH text reaches the matcher with its trailing whitespace stripped, so
    editable lines get the same treatment. Everything outside the regions is
    immutable and returned byte for byte.
    """
    parts = []
    cursor = 0
    for start, end in sorted(_mutable_ranges(text)):
        if start < cursor:
            continue  # nested region, already covered by the enclosing one
        lines = text[start:end].split("\n")
        # The last piece is the indentation of the END marker's own line.
        stripped = [line.rstrip() for line in lines[:-1]] + lines[-1:]
        parts += [text[cursor:start], "\n".join(stripped)]
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _shift_indentation(text: str, offset: int, indent_chars: str = "") -> str:
    """Shift every non-empty line of text by a signed indentation offset.

    Relative indentation between lines is preserved exactly, including lines
    that dedent below the first line; only the final indentation is clamped at
    column zero. The leading whitespace is drawn from ``indent_chars`` (the
    whitespace of the line the block was matched at, so tabs stay tabs) and
    padded with spaces beyond it. Empty lines are emitted empty.
    """
    shifted = []
    for line in text.split("\n"):
        if line.strip():
            indent = max(_indent_of(line) + offset, 0)
            if indent <= len(indent_chars):
                whitespace = indent_chars[:indent]
            else:
                whitespace = indent_chars + " " * (indent - len(indent_chars))
            shifted.append(whitespace + line.strip())
        else:
            shifted.append("")
    return "\n".join(shifted)


class _LineMatch(NamedTuple):
    """Whole lines of the original that a SEARCH block matches."""

    text: str
    pos: int
    end: int
    # Set when the block only matched after re-indenting it: the signed shift
    # and the leading whitespace of the line it was matched at.
    indent_offset: int = 0
    indent_chars: Optional[str] = None


def _iter_line_matches(search_text: str, original_text: str) -> Iterator[_LineMatch]:
    """Yield every place where the search block matches whole lines.

    A block matches a run of consecutive lines, never a fragment of one, so
    ``x = 1`` cannot hit ``idx = 10``. Trailing whitespace is ignored. Matches
    at the block's own indentation come first, in file order, followed by
    matches that need the block shifted to another indentation.
    """
    if not search_text.strip():
        return

    search_lines = [line.rstrip() for line in search_text.split("\n")]
    lines = original_text.split("\n")
    stripped = [line.rstrip() for line in lines]
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1

    count = len(search_lines)
    candidates = range(len(lines) - count + 1)

    def match_at(index: int, **indent) -> _LineMatch:
        end = starts[index + count - 1] + len(lines[index + count - 1])
        pos = starts[index]
        return _LineMatch(original_text[pos:end], pos, end, **indent)

    for index in candidates:
        if stripped[index : index + count] == search_lines:
            yield match_at(index)

    first_search_line = search_lines[0].strip()
    if not first_search_line:
        return

    # A search block copied out of a nested scope carries its own base
    # indentation. Measure the signed offset from that base to each candidate
    # line in the original and shift the whole block by it, so lines that
    # dedent below the first line keep dedenting and cannot be flattened onto
    # a same-indent block that lives in a different scope.
    base_indent = _indent_of(search_lines[0])
    for index in candidates:
        line = lines[index]
        if line.strip() != first_search_line:
            continue
        indent_offset = _indent_of(line) - base_indent
        indent_chars = line[: _indent_of(line)]
        shifted = _shift_indentation(search_text, indent_offset, indent_chars)
        shifted_lines = shifted.split("\n")
        if shifted_lines == search_lines:
            continue  # same as the unshifted block, handled above
        if stripped[index : index + count] == shifted_lines:
            yield match_at(
                index, indent_offset=indent_offset, indent_chars=indent_chars
            )


def _find_indented_match(search_text: str, original_text: str) -> tuple[str, int]:
    """
    Find the lines search_text matches in original_text, first as given and
    then with corrected indentation. Returns (matched_text, position) or
    ("", -1).
    """
    match = next(_iter_line_matches(search_text, original_text), None)
    if match is None:
        return "", -1
    return match.text, match.pos


def _apply_indentation_to_replace(
    replace_text: str, indent_offset: int, indent_chars: str = ""
) -> str:
    """Shift the replace text by the offset the search block was matched at.

    Using the search-derived offset, rather than re-basing the replace block on
    its own first line, keeps any intentional difference between the search
    and replace first-line indentation.
    """
    if not replace_text.strip():
        return replace_text
    return _shift_indentation(replace_text, indent_offset, indent_chars)


def _clean_evolve_markers(text: str) -> str:
    """Remove EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers from text
    if present."""
    # Remove various forms of EVOLVE markers that might appear in patches
    patterns_to_remove = [
        r"^\s*#\s*EVOLVE-BLOCK-START\s*$",  # Python style
        r"^\s*//\s*EVOLVE-BLOCK-START\s*$",  # C/C++/CUDA style
        r"^\s*!\s*EVOLVE-BLOCK-START\s*$",  # Fortran style
        r"^\s*<!--\s*EVOLVE-BLOCK-START\s*-->\s*$",  # HTML/Markdown style
        r"^\s*\(\*\s*EVOLVE-BLOCK-START\s*\*\)\s*$",  # Wolfram style
        r"^\s*EVOLVE-BLOCK-START\s*$",  # Plain text
        r"^\s*#\s*EVOLVE-BLOCK-END\s*$",  # Python style
        r"^\s*//\s*EVOLVE-BLOCK-END\s*$",  # C/C++/CUDA
        r"^\s*!\s*EVOLVE-BLOCK-END\s*$",  # Fortran style
        r"^\s*<!--\s*EVOLVE-BLOCK-END\s*-->\s*$",  # HTML/Markdown style
        r"^\s*\(\*\s*EVOLVE-BLOCK-END\s*\*\)\s*$",  # Wolfram style
        r"^\s*EVOLVE-BLOCK-END\s*$",  # Plain text
    ]

    cleaned_text = text
    markers_found = False

    for pattern in patterns_to_remove:
        if re.search(pattern, cleaned_text, flags=re.MULTILINE):
            markers_found = True
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE)

    if markers_found:
        logger.debug("Removed EVOLVE-BLOCK markers from patch text")

    return cleaned_text


def redact_immutable(text: str, no_state: bool = False) -> str:
    out = []
    for a, b in _mutable_ranges(text):
        # keep immutable gap as a 1-liner placeholder
        if not no_state:
            out.append("<… non-evolvable code omitted …>")
        out.append(text[a:b])
    if not no_state:
        out.append("<… non-evolvable tail omitted …>")
    return "".join(out)


class PatchError(RuntimeError):
    pass


_MARKER_ONLY_SEARCH_ERROR = """SEARCH block contains only EVOLVE-BLOCK markers

The EVOLVE-BLOCK-START and EVOLVE-BLOCK-END lines are immutable and cannot be
used as an anchor.

Suggestions:
1. To change code, copy the lines you want to replace into the SEARCH block
2. To add code at the end of the editable region, leave the SEARCH block empty
3. Do not repeat the EVOLVE-BLOCK markers in SEARCH/REPLACE blocks"""


def _reject_marker_only_search(patch_text: str) -> None:
    """Raise when a SEARCH block names nothing but EVOLVE-BLOCK markers.

    Removing the markers would leave an empty SEARCH, which means "insert at
    the end of the last editable region": not where the block pointed.
    """
    for block in PATCH_PATTERN.finditer(patch_text):
        search = block.group(1)
        if search.strip() and not _clean_evolve_markers(search).strip():
            raise PatchError(_MARKER_ONLY_SEARCH_ERROR)


def _find_similar_lines(
    search_line: str, original_text: str, max_suggestions: int = 3
) -> List[Tuple[str, int]]:
    """Find similar lines in the original text for suggestions."""
    import difflib

    search_line_clean = search_line.strip()
    if not search_line_clean:
        return []

    original_lines = original_text.splitlines()
    similarities = []

    for i, line in enumerate(original_lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, search_line_clean, line_clean).ratio()
        if ratio > 0.6:  # Only suggest lines with >60% similarity
            similarities.append((line, i + 1, ratio))

    # Sort by similarity and return top suggestions
    similarities.sort(key=lambda x: x[2], reverse=True)
    return [(line, line_num) for line, line_num, _ in similarities[:max_suggestions]]


def _find_best_match_with_diff(
    search_text: str, original_text: str
) -> Optional[Tuple[str, int, List[str]]]:
    """Find the best matching block and return a diff comparison."""
    import difflib

    search_lines = search_text.strip().splitlines()
    if not search_lines:
        return None

    original_lines = original_text.splitlines()
    search_len = len(search_lines)

    best_match = None
    best_ratio = 0.0
    best_start_line = 0

    # Look for the best matching block of the same length
    for i in range(len(original_lines) - search_len + 1):
        candidate_lines = original_lines[i : i + search_len]

        # Calculate similarity for the entire block
        candidate_text = "\n".join(candidate_lines)
        search_block = "\n".join(search_lines)

        ratio = difflib.SequenceMatcher(None, search_block, candidate_text).ratio()

        if ratio > best_ratio and ratio > 0.7:  # Require >70% similarity
            best_ratio = ratio
            best_match = candidate_lines
            best_start_line = i + 1

    if best_match is None:
        return None

    # Generate unified diff
    search_lines_labeled = [f"  {line}" for line in search_lines]
    match_lines_labeled = [f"  {line}" for line in best_match]

    diff_lines = list(
        difflib.unified_diff(
            search_lines_labeled,
            match_lines_labeled,
            fromfile="Search Pattern",
            tofile=f"Actual Code (line {best_start_line})",
            lineterm="",
            n=0,  # No context lines needed since we're showing the full blocks
        )
    )

    # Remove the file headers and @@ lines for cleaner output
    clean_diff = []
    for line in diff_lines:
        if (
            not line.startswith("---")
            and not line.startswith("+++")
            and not line.startswith("@@")
        ):
            clean_diff.append(line)

    return best_match, best_start_line, clean_diff


def _get_context_lines(
    text: str, position: int, context_lines: int = 2
) -> Tuple[List[str], int]:
    """Get context lines around a position in text."""
    lines = text.splitlines()
    if not lines:
        return [], 0

    # Find which line the position is on
    char_count = 0
    target_line = 0
    for i, line in enumerate(lines):
        if char_count + len(line) + 1 > position:  # +1 for newline
            target_line = i
            break
        char_count += len(line) + 1

    start_line = max(0, target_line - context_lines)
    end_line = min(len(lines), target_line + context_lines + 1)

    context = lines[start_line:end_line]
    return context, start_line + 1


def _get_line_position(text: str, line_num: int) -> int:
    """Get character position of the start of a specific line number (1-based)."""
    lines = text.splitlines(keepends=True)
    if line_num < 1 or line_num > len(lines):
        return 0

    char_pos = 0
    for i in range(line_num - 1):
        char_pos += len(lines[i])
    return char_pos


def _char_to_line_num(text: str, char_pos: int) -> int:
    """Convert character position to line number (1-based)."""
    if char_pos < 0:
        return 1

    lines = text.splitlines(keepends=True)
    current_pos = 0
    for i, line in enumerate(lines):
        if current_pos + len(line) > char_pos:
            return i + 1
        current_pos += len(line)

    return len(lines) if lines else 1


def _create_search_not_found_error(
    search_text: str, original_text: str, mutable_ranges: List[Tuple[int, int]]
) -> str:
    """Create a detailed error message when search text is not found."""
    search_lines = search_text.strip().splitlines()
    if not search_lines:
        return "Empty search text provided"

    first_line = search_lines[0].strip()

    # Find similar lines for suggestions
    similar_lines = _find_similar_lines(first_line, original_text)

    error_parts = [
        "SEARCH text not found in editable regions",
        "",
    ]

    # Show the search text in a more compact way
    if len(search_lines) == 1:
        error_parts.extend(
            [
                f"Looking for: {first_line!r}",
                "",
            ]
        )
    else:
        line_count = len(search_lines)
        error_parts.extend(
            [
                f"Looking for {line_count}-line block starting with: {first_line!r}",
                "",
                "Full search pattern:",
                "```",
                search_text.strip(),
                "```",
                "",
            ]
        )

    # Try to find the best matching block and show a diff
    best_match_result = _find_best_match_with_diff(search_text, original_text)

    if best_match_result:
        best_match, start_line, diff_lines = best_match_result

        # Check if the match is in an editable region
        match_start_pos = _get_line_position(original_text, start_line)
        match_text = "\n".join(best_match)
        match_span = (match_start_pos, match_start_pos + len(match_text))
        in_editable = _inside(match_span, mutable_ranges)
        region_status = "✓ editable" if in_editable else "✗ immutable"

        error_parts.extend(
            [
                f"Found similar code block at line {start_line} ({region_status}):",
                "",
                "Differences between search pattern and actual code:",
                "```diff",
            ]
        )

        error_parts.extend(diff_lines)
        error_parts.extend(
            [
                "```",
                "",
            ]
        )

        if not in_editable:
            error_parts.extend(
                [
                    "⚠️  Note: The similar code is in an immutable region.",
                    "   Look for similar code in the editable regions below.",
                    "",
                ]
            )

    elif similar_lines:
        # Fallback to the old similar lines approach for single-line searches
        error_parts.extend(
            [
                "Found similar text (but not exact match):",
            ]
        )
        for line, line_num in similar_lines:
            # Show if it's in an editable region or not
            line_pos = _get_line_position(original_text, line_num)
            span = (line_pos, line_pos + len(line))
            in_editable = _inside(span, mutable_ranges)
            region_status = "✓ editable" if in_editable else "✗ immutable"
            line_content = line.strip()
            error_parts.append(f"  Line {line_num}: {line_content} ({region_status})")
        error_parts.append("")

    # Show a more focused view of editable regions
    if mutable_ranges:
        error_parts.extend(
            [
                "Editable regions where you can make changes:",
            ]
        )
        for i, (start, end) in enumerate(mutable_ranges[:2]):  # Show max 2 regions
            # Convert char positions to line numbers for display
            start_line = _char_to_line_num(original_text, start)
            end_line = _char_to_line_num(original_text, end)

            error_parts.append(f"  Region {i + 1} (lines {start_line}-{end_line}):")

            # Show a few key lines from this region
            region_text = original_text[start:end].strip()
            region_lines = region_text.splitlines()
            if region_lines:
                # Show first few lines and last few lines if it's long
                if len(region_lines) <= 6:
                    for line in region_lines:
                        error_parts.append(f"    {line}")
                else:
                    for line in region_lines[:3]:
                        error_parts.append(f"    {line}")
                    line_count = len(region_lines) - 6
                    error_parts.append(f"    ... ({line_count} more lines)")
                    for line in region_lines[-3:]:
                        error_parts.append(f"    {line}")
                error_parts.append("")

        if len(mutable_ranges) > 2:
            remaining = len(mutable_ranges) - 2
            error_parts.append(f"  ... and {remaining} more regions")
            error_parts.append("")

    # More actionable suggestions
    if similar_lines:
        error_parts.extend(
            [
                "Quick fixes:",
                "• Check indentation - search text must match exactly "
                "including spaces/tabs",
                "• Look for typos in the search text",
                "• Try searching for just the first line instead of the full block",
            ]
        )
    else:
        error_parts.extend(
            [
                "Quick fixes:",
                "• Verify the text exists in the file",
                "• Check that you're searching within EVOLVE-BLOCK regions",
                "• Try a smaller, more specific search pattern",
            ]
        )

    return "\n".join(error_parts)


def _create_evolve_block_error(
    matched_text: str,
    position: int,
    original_text: str,
    mutable_ranges: List[Tuple[int, int]],
) -> str:
    """Create a detailed error message for EVOLVE-BLOCK violations."""
    first_line = matched_text.splitlines()[0] if matched_text.splitlines() else ""

    # Get context around the found position
    context_lines, start_line_num = _get_context_lines(original_text, position, 3)

    error_parts = [
        "Attempted to edit outside EVOLVE-BLOCK regions",
        "",
        f"Found text: {first_line!r}",
        f"At position: {position}",
        "",
        "Context around found text:",
    ]

    for i, line in enumerate(context_lines):
        line_num = start_line_num + i
        marker = " >>> " if i == len(context_lines) // 2 else "     "
        error_parts.append(f"{marker}Line {line_num:3}: {line}")

    error_parts.extend(
        [
            "",
            "This text was found in an immutable (non-editable) region.",
            "",
            "Available editable regions (EVOLVE-BLOCK content):",
        ]
    )

    if mutable_ranges:
        for i, (start, end) in enumerate(mutable_ranges[:3]):
            region_text = original_text[start:end].strip()
            region_lines = region_text.splitlines()
            if region_lines:
                error_parts.append(f"  Region {i + 1} (chars {start}-{end}):")
                for line in region_lines[:3]:
                    error_parts.append(f"    {line}")
                if len(region_lines) > 3:
                    error_parts.append(f"    ... ({len(region_lines) - 3} more lines)")
                error_parts.append("")
    else:
        error_parts.append("  No EVOLVE-BLOCK regions found in the code!")

    error_parts.extend(
        [
            "Suggestions:",
            "1. Move your edit to within an EVOLVE-BLOCK region",
            "2. Check if similar code exists in the editable regions above",
            "3. Ensure your search text targets code within # EVOLVE-BLOCK-START/END markers",
        ]
    )

    return "\n".join(error_parts)


def _create_no_evolve_block_error(original_text: str, operation: str) -> str:
    """Create an error message when no EVOLVE-BLOCK regions are found."""
    lines = original_text.splitlines()

    error_parts = [
        f"Cannot perform {operation}: No EVOLVE-BLOCK regions found",
        "",
        "The code must contain EVOLVE-BLOCK regions to be editable.",
        "",
        "Current file structure:",
    ]

    # Show first few lines of the file
    for i, line in enumerate(lines[:10]):
        error_parts.append(f"  Line {i + 1:2}: {line}")

    if len(lines) > 10:
        error_parts.append(f"  ... ({len(lines) - 10} more lines)")

    error_parts.extend(
        [
            "",
            "Expected format:",
            "```",
            "# Your immutable code here",
            "",
            "# EVOLVE-BLOCK-START",
            "# Your editable code goes here",
            "def function():",
            "    pass",
            "# EVOLVE-BLOCK-END",
            "",
            "# More immutable code here",
            "```",
            "",
            "Suggestions:",
            "1. Add EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers around editable code",
            "2. Ensure the markers are properly formatted (with # for Python, // for C/C++)",
            "3. Check that there's at least one EVOLVE-BLOCK region in the file",
        ]
    )

    return "\n".join(error_parts)


def apply_search_replace(
    patch_text: str,
    original: str,
    strict: bool = True,
) -> tuple[str, int]:
    """
    Apply SEARCH/REPLACE blocks but **only** inside EVOLVE regions.
    Mutable ranges are recalculated after each replacement to account for
    text changes.
    """
    _reject_marker_only_search(patch_text)
    new_text = original
    num_applied = 0
    for block in PATCH_PATTERN.finditer(patch_text):
        search, replace = block.group(1), block.group(2)
        # Clean EVOLVE markers from search and replace text if present
        search = _clean_evolve_markers(search)
        replace = _clean_evolve_markers(replace)

        # Strip trailing whitespace from search and replace blocks
        search = _strip_trailing_whitespace(search)
        replace = _strip_trailing_whitespace(replace)

        # Recalculate mutable ranges based on current text state
        mutable = _mutable_ranges(new_text)

        # ── insertions ───────────────────────────────────────────────────────
        if not search.strip():  # empty SEARCH  → insertion
            # Safe strategy: append inside the final mutable span.
            if not mutable:
                msg = _create_no_evolve_block_error(new_text, "insertion")
                raise PatchError(msg)
            if not replace:
                # Empty payload: keep the historical content no-op.
                num_applied += 1
                continue
            a, b = mutable[-1]
            # `b` points at the END marker itself and `replace` has no
            # trailing newline (stripped above), so splicing at `b` would
            # glue the marker onto the payload's last line. When the marker
            # sits on its own (possibly indented) line, insert at the start
            # of that line; when code precedes the marker on the same line,
            # append after that code and move the marker to its own line.
            line_start = new_text.rfind("\n", 0, b) + 1
            marker_line_prefix = new_text[line_start:b]
            if marker_line_prefix.strip():
                leading_whitespace = marker_line_prefix[
                    : len(marker_line_prefix) - len(marker_line_prefix.lstrip())
                ]
                new_text = (
                    new_text[:line_start]
                    + marker_line_prefix.rstrip()
                    + "\n"
                    + replace
                    + "\n"
                    + leading_whitespace
                    + new_text[b:]
                )
            else:
                new_text = (
                    new_text[:line_start] + replace + "\n" + new_text[line_start:]
                )
            num_applied += 1
            continue

        # ── replacements ────────────────────────────────────────────────────
        # Find the lines the search text matches, with indentation correction
        # if needed, and take the first match that lies in an editable region.
        matches = list(_iter_line_matches(search, new_text))
        match = next((m for m in matches if _inside((m.pos, m.end), mutable)), None)

        if match is None:
            if matches:
                msg = _create_evolve_block_error(
                    matches[0].text, matches[0].pos, new_text, mutable
                )
                raise PatchError(msg)
            if strict:
                msg = _create_search_not_found_error(search, new_text, mutable)
                raise PatchError(msg)
            continue

        # If we found an indented match, shift the replace text by the same
        # signed offset the search block was shifted by
        if match.indent_chars is not None:
            replace = _apply_indentation_to_replace(
                replace, match.indent_offset, match.indent_chars
            )
            logger.debug("Applied indentation correction to search/replace block")

        new_text = new_text[: match.pos] + replace + new_text[match.end :]
        num_applied += 1
    return new_text, num_applied


def write_git_diff(
    original: str,
    updated: str,
    filename: str,
    out_path: Union[str, Path],
    context: int = 9999,
) -> Path:
    """
    Save a unified-diff (Git patch) of *filename* to *out_path*.

    Parameters
    ----------
    original : str
        Pre-patch file contents.
    updated : str
        Post-patch file contents.
    filename : str
        Path shown inside the diff headers.
    out_path : Union[str, Path]
        Where to write the `.patch` file.
    context : int, default 3
        Number of unchanged context lines to include (-U).
    """
    patch_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context,
    )
    out_path = Path(out_path)
    out_path.write_text("".join(patch_lines), encoding="utf-8")
    return out_path


def apply_diff_patch(
    patch_str: str,
    original_str: Optional[str] = None,
    patch_dir: Optional[Union[str, Path]] = None,
    original_path: Optional[Union[str, Path]] = None,
    language: str = "python",
    verbose: bool = True,
) -> tuple[str, int, Optional[Path], Optional[str], Optional[str], Optional[Path]]:
    """
    Apply SEARCH/REPLACE blocks to old string and optionally emit a `.patch`.
    Returns the updated string, number of patches applied, path to the new
    file (if patch_dir is specified), and an error message string if an
    error occurred (otherwise None).
    """
    if original_str is None and original_path is None:
        raise ValueError("Either original_str or original_path must be provided")
    if original_str is None:
        og_path = Path(str(original_path))
        original = og_path.read_text("utf-8")
    else:
        original = original_str

    # Normalize the editable regions only; immutable code is left untouched
    original = _normalize_mutable_whitespace(original)

    error_message: Optional[str] = None
    # Init with original content and 0 applied patches in case of error
    updated_content: str = original
    num_applied: int = 0
    output_path: Optional[Path] = None

    # Strip trailing whitespace from patch text
    patch_str = _strip_trailing_whitespace(patch_str)

    # Remove EVOLVE-BLOCK markers before parsing SEARCH/REPLACE sections.
    raw_patch_str = patch_str
    patch_str = _clean_evolve_markers(patch_str)

    if patch_dir is not None:
        patch_dir = Path(patch_dir)
        patch_dir.mkdir(parents=True, exist_ok=True)
        # Store the raw search/replace blocks
        patch_path = patch_dir / "search_replace.txt"
        patch_path.write_text(patch_str, "utf-8")

    try:
        # Checked on the text as written: with the markers removed, a
        # marker-only SEARCH can no longer be told from an empty one.
        _reject_marker_only_search(raw_patch_str)
        # Apply the patch
        applied_content, patches_count = apply_search_replace(patch_str, original)
        updated_content = applied_content
        num_applied = patches_count
    except PatchError as e:
        error_message = str(e)
        # Return original content, 0 applied, no output path, error msg
        return updated_content, 0, None, error_message, None, None

    # Validate the assembled file's EVOLVE-BLOCK markers. This is normally a
    # no-op for diff patches because the markers come from the original seed
    # and the SEARCH/REPLACE blocks operate strictly inside them — but a
    # malformed patch that injects bad comment delimiters around the markers
    # would otherwise reach the evaluator silently. See marker_validation.py.
    marker_err = validate_evolve_markers(updated_content, language)
    if marker_err is not None:
        if verbose:
            logger.info(f"Rejecting diff patch: {marker_err}")
        return original, 0, None, marker_err, None, None

    suffix = f".{get_language_extension(language)}"

    # If successful, proceed to write files if patch_dir is specified
    if patch_dir is not None:
        # Store the original string as a backup file
        backup_path = patch_dir / f"original{suffix}"
        backup_path.write_text(original, "utf-8")

        # Write the updated file
        output_path = patch_dir / f"main{suffix}"
        output_path.write_text(updated_content, "utf-8")

        # Write the git diff if requested
        write_git_diff(
            original,
            updated_content,
            filename=backup_path.name,
            out_path=patch_dir / "edit.diff",
        )
        patch_txt = (patch_dir / "edit.diff").read_text("utf-8")
        # Print the patch file
        if verbose:
            logger.debug(f"Patch file written to: {patch_dir / 'edit.diff'}")
            logger.debug(f"Patch file content:\n{patch_txt}")
        return (
            updated_content,
            num_applied,
            output_path,
            error_message,
            patch_txt,
            patch_dir / "edit.diff",
        )
    else:
        return (updated_content, num_applied, None, error_message, None, None)
