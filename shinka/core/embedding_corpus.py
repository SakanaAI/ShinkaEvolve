"""Extract file content from multi-file corpus text format."""

import re
from typing import Optional


def extract_file_content(corpus_text: str, filename: str) -> Optional[str]:
    """
    Extract the content of a specific file from a corpus text dump.
    Returns None if the file is not found or the corpus format is invalid.
    """
    if not corpus_text:
        return None

    # Regex to find the file header and capture content until the next header or end of string
    # Header format: === FILE: {filename} ({size} bytes)[TRUNCATED?] ===
    escaped_filename = re.escape(filename)
    # Look for header at start of string or after a newline
    pattern = rf"(?:^|\n)=== FILE: {escaped_filename} \(\d+ bytes\)(?: \[TRUNCATED\])? ===\n(.*?)(?=\n=== FILE: |$)"

    match = re.search(pattern, corpus_text, re.DOTALL)
    if match:
        return match.group(1)

    return None
