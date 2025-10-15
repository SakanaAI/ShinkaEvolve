from pathlib import Path
from typing import Optional, Union
from .apply_diff import write_git_diff, _mutable_ranges
from shinka.llm import extract_between
import logging

logger = logging.getLogger(__name__)


def apply_full_patch(
    patch_str: str,
    original_str: Optional[str] = None,
    patch_dir: Optional[Union[str, Path]] = None,
    original_path: Optional[Union[str, Path]] = None,
    language: str = "python",
    verbose: bool = True,
) -> tuple[str, int, Optional[Path], Optional[str], Optional[str], Optional[Path]]:
    if original_str is None and original_path is None:
        raise ValueError("Either original_str or original_path must be provided")
    if original_str is None:
        if original_path is None:
            raise ValueError("original_path cannot be None")
        og_path = Path(original_path)
        original = og_path.read_text("utf-8")
    else:
        original = original_str

    error_message: Optional[str] = None
    # Init with original content and 0 applied patches in case of error
    updated_content: str = original
    num_applied: int = 0
    output_path: Optional[Path] = None

    # Extract code from language fences
    extracted_code = extract_between(
        patch_str,
        f"```{language}",
        "```",
        False,
    )

    # Handle the case where extract_between returns None, dict, or "none"
    if (
        extracted_code is None
        or isinstance(extracted_code, dict)
        or extracted_code == "none"
    ):
        error_message = "Could not extract code from patch string"
        return original, 0, None, error_message, None, None

    patch_code = str(extracted_code)

    if patch_dir is not None:
        patch_dir = Path(patch_dir)
        patch_dir.mkdir(parents=True, exist_ok=True)
        # Store the raw patch content
        patch_path = patch_dir / "rewrite.txt"
        patch_path.write_text(patch_code, "utf-8")

    try:
        # Get mutable ranges from original content
        mutable_ranges = _mutable_ranges(original)

        if not mutable_ranges:
            # No EVOLVE-BLOCK regions found, treat as error for full patch
            msg = "No EVOLVE-BLOCK regions found in original content"
            error_message = msg
            return original, 0, None, error_message, None, None

        # Build updated content by preserving immutable parts
        # and replacing mutable parts
        updated_content = ""
        last_end = 0

        # Check if patch_code contains EVOLVE-BLOCK markers
        patch_mutable_ranges = _mutable_ranges(patch_code)

        if patch_mutable_ranges:
            # Patch contains EVOLVE-BLOCK markers, extract from them
            for i, (start, end) in enumerate(mutable_ranges):
                # Add immutable part before this mutable range
                updated_content += original[last_end:start]

                # Get corresponding mutable content from patch
                if i < len(patch_mutable_ranges):
                    patch_start, patch_end = patch_mutable_ranges[i]
                    replacement_content = patch_code[patch_start:patch_end]
                else:
                    # Not enough mutable regions in patch, keep original
                    replacement_content = original[start:end]

                updated_content += replacement_content
                last_end = end
        else:
            # Patch doesn't contain EVOLVE-BLOCK markers
            # Assume entire patch content should replace all mutable regions
            if len(mutable_ranges) == 1:
                # Single mutable region, replace with entire patch content
                start, end = mutable_ranges[0]

                # The mutable range ends before "EVOLVE-BLOCK-END" text
                # We need to find the actual start of the comment line
                if language == "python":
                    end_marker = "# EVOLVE-BLOCK-END"
                elif language in ["cuda", "cpp", "rust"]:
                    end_marker = "// EVOLVE-BLOCK-END"
                else:
                    end_marker = "# EVOLVE-BLOCK-END"  # Default fallback

                end_marker_pos = original.find(end_marker, end - 5)
                if end_marker_pos == -1:
                    # Fallback: use the original end position
                    end_marker_pos = end

                # Ensure proper newline handling around the patch content
                if patch_code and not patch_code.startswith("\n"):
                    patch_code = "\n" + patch_code

                if patch_code and not patch_code.endswith("\n"):
                    patch_code = patch_code + "\n"

                updated_content = (
                    original[:start] + patch_code + original[end_marker_pos:]
                )
            else:
                # Multiple mutable regions, this is ambiguous
                error_message = (
                    "Multiple EVOLVE-BLOCK regions found but patch "
                    "doesn't specify which to replace"
                )
                return original, 0, None, error_message, None, None

        # Add remaining immutable content after last mutable range
        if patch_mutable_ranges and mutable_ranges:
            updated_content += original[mutable_ranges[-1][1] :]

        num_applied = 1

    except Exception as e:
        error_message = f"Error applying full patch: {str(e)}"
        return original, 0, None, error_message, None, None

    if language == "python":
        suffix = ".py"
    elif language == "cpp":
        suffix = ".cpp"
    elif language == "cuda":
        suffix = ".cu"
    elif language == "rust":
        suffix = ".rs"
    else:
        raise ValueError(f"Language {language} not supported")

    # If successful, proceed to write files if patch_dir is specified
    if patch_dir is not None:
        # Store the original string as a backup file
        backup_path = patch_dir / f"original{suffix}"
        backup_path.write_text(original, "utf-8")

        # Write the updated file
        output_path = patch_dir / f"main{suffix}"
        output_path.write_text(updated_content, "utf-8")

        # Write the git diff if requested
        diff_path = patch_dir / "edit.diff"
        write_git_diff(
            original,
            updated_content,
            filename=backup_path.name,
            out_path=diff_path,
        )
        patch_txt = diff_path.read_text("utf-8")
        # Print the patch file
        if verbose:
            logger.info(f"Patch file written to: {diff_path}")
            logger.info(f"Patch file content:\n{patch_txt}")
        return (
            updated_content,
            num_applied,
            output_path,
            error_message,
            patch_txt,
            diff_path,
        )
    else:
        return updated_content, num_applied, None, error_message, None, None
