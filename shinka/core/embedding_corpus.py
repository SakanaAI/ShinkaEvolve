import fnmatch
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


import re

@dataclass
class EmbeddingCorpus:
    """Result of building an embedding corpus for a generation directory."""

    text: str
    included_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    binary_files: List[str] = field(default_factory=list)
    truncated: bool = False
    total_bytes: int = 0


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



def _is_text_bytes(buf: bytes) -> bool:
    """Heuristic: treat content as binary if it contains null bytes."""
    if not buf:
        return True
    return b"\x00" not in buf


def _sha256_prefix(buf: bytes, length: int = 8) -> str:
    return hashlib.sha256(buf).hexdigest()[:length]


def _matches_any(patterns: Sequence[str], path: str) -> bool:
    if not patterns:
        return False
    p_obj = Path(path)
    for pat in patterns:
        if pat in ("**", "**/*"):
            return True
        if fnmatch.fnmatch(path, pat):
            return True
        try:
            if p_obj.match(pat):
                return True
        except Exception:
            continue
    return False


def build_embedding_corpus(
    root: Path,
    *,
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
    max_files: int,
    max_total_bytes: int,
    max_bytes_per_file: int,
    changed_first: Optional[Iterable[Path]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    exclude_suffixes: Optional[Set[str]] = None,
    exclude_files: Optional[Set[str]] = None,
) -> EmbeddingCorpus:
    """
    Build a deterministic, artifact-agnostic corpus from a generation directory.

    Text files contribute their (possibly truncated) content. Binary files and
    over-limit files contribute small placeholders (path, size, hash) so changes
    are still visible to novelty checks without embedding raw bytes.
    """

    root = root.resolve()
    exclude_dirs = exclude_dirs or set()
    exclude_suffixes = exclude_suffixes or set()
    exclude_files = exclude_files or set()

    def should_skip(rel: Path) -> bool:
        if rel.name in exclude_files:
            return True
        if rel.suffix in exclude_suffixes:
            return True
        if rel.parts and rel.parts[0] in exclude_dirs:
            return True
        rel_posix = rel.as_posix()
        if exclude_globs and _matches_any(exclude_globs, rel_posix):
            return True
        if include_globs and not _matches_any(include_globs, rel_posix):
            return True
        return False

    seen: Set[Path] = set()
    ordered_candidates: List[Path] = []

    # Prioritize explicitly changed files (if provided)
    if changed_first:
        for p in changed_first:
            abs_path = (root / p).resolve() if not p.is_absolute() else p
            if abs_path.is_file() and abs_path.is_relative_to(root):
                rel = abs_path.relative_to(root)
                if rel not in seen and not should_skip(rel):
                    seen.add(rel)
                    ordered_candidates.append(rel)

    # Discover remaining files
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel in seen:
            continue
        if should_skip(rel):
            continue
        seen.add(rel)
        ordered_candidates.append(rel)

    segments: List[str] = []
    included_files: List[str] = []
    skipped_files: List[str] = []
    binary_files: List[str] = []
    truncated = False
    total_bytes = 0

    for rel in ordered_candidates:
        if len(included_files) >= max_files:
            truncated = True
            skipped_files.extend([r.as_posix() for r in ordered_candidates[len(included_files) :]])
            break

        abs_path = root / rel
        try:
            raw = abs_path.read_bytes()
        except Exception:
            skipped_files.append(rel.as_posix())
            continue

        size = len(raw)
        to_embed = raw[: max_bytes_per_file]
        file_truncated = size > max_bytes_per_file

        if total_bytes >= max_total_bytes:
            truncated = True
            skipped_files.append(rel.as_posix())
            continue

        is_text = _is_text_bytes(to_embed)
        rel_posix = rel.as_posix()

        if is_text:
            try:
                text = to_embed.decode("utf-8", errors="replace")
            except Exception:
                is_text = False

        if not is_text:
            placeholder = (
                f"[BINARY FILE] {rel_posix} size={size} sha256={_sha256_prefix(raw)}"
            )
            addition = placeholder + "\n"
            if total_bytes + len(addition) > max_total_bytes:
                truncated = True
                skipped_files.append(rel_posix)
                continue
            segments.append(placeholder)
            included_files.append(rel_posix)
            binary_files.append(rel_posix)
            total_bytes += len(addition)
            continue

        # Text path header for clarity/determinism
        header = f"=== FILE: {rel_posix} ({size} bytes){' [TRUNCATED]' if file_truncated else ''} ===\n"
        addition_len = len(header) + len(text) + 1  # trailing newline
        if total_bytes + addition_len > max_total_bytes:
            # Try to fit partial content
            remaining = max_total_bytes - total_bytes - len(header) - 1
            if remaining <= 0:
                truncated = True
                skipped_files.append(rel_posix)
                continue
            text = text[:remaining]
            addition_len = len(header) + len(text) + 1
            truncated = True

        segments.append(header + text + "\n")
        included_files.append(rel_posix)
        total_bytes += addition_len

    corpus_text = "".join(segments)

    return EmbeddingCorpus(
        text=corpus_text,
        included_files=included_files,
        skipped_files=skipped_files,
        binary_files=binary_files,
        truncated=truncated,
        total_bytes=total_bytes,
    )
