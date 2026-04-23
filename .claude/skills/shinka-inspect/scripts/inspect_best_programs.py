#!/usr/bin/env python3
"""Build a compact Markdown context bundle from best Shinka programs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

try:
    from shinka.utils import load_programs_to_df
except ModuleNotFoundError:
    import sys

    # Allow direct script execution from a source checkout without editable install.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from shinka.utils import load_programs_to_df


DEFAULT_K = 5
DEFAULT_MAX_CODE_CHARS = 4000


@dataclass(frozen=True)
class InspectConfig:
    results_dir: str
    k: int
    out: str | None
    max_code_chars: int
    min_generation: int | None
    include_feedback: bool


def _parse_args() -> InspectConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Load top-performing programs from a Shinka run and export a "
            "Markdown context bundle."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to Shinka results directory or direct SQLite DB path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of programs to include (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output Markdown file path. Default: "
            "<results-dir>/shinka_inspect_context.md"
        ),
    )
    parser.add_argument(
        "--max-code-chars",
        type=int,
        default=DEFAULT_MAX_CODE_CHARS,
        help=(
            "Truncate each code block to this many chars "
            f"(default: {DEFAULT_MAX_CODE_CHARS})."
        ),
    )
    parser.add_argument(
        "--min-generation",
        type=int,
        default=None,
        help="Optional minimum generation filter.",
    )
    parser.add_argument(
        "--include-feedback",
        dest="include_feedback",
        action="store_true",
        default=True,
        help="Include text feedback in output (default: enabled).",
    )
    parser.add_argument(
        "--no-include-feedback",
        dest="include_feedback",
        action="store_false",
        help="Disable text feedback in output.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.max_code_chars <= 0:
        raise ValueError("--max-code-chars must be > 0")

    return InspectConfig(
        results_dir=args.results_dir,
        k=args.k,
        out=args.out,
        max_code_chars=args.max_code_chars,
        min_generation=args.min_generation,
        include_feedback=args.include_feedback,
    )


def _resolve_db_path(results_dir_or_db: str) -> Path:
    provided = Path(results_dir_or_db).expanduser()
    if provided.suffix in {".sqlite", ".db"}:
        return provided

    candidates = (
        provided / "programs.sqlite",
        provided / "evolution_db.sqlite",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    sqlite_files = sorted(provided.glob("*.sqlite"))
    if sqlite_files:
        return sqlite_files[0]

    return provided / "programs.sqlite"


def _resolve_output_path(config: InspectConfig, db_path: Path) -> Path:
    if config.out:
        return Path(config.out).expanduser()
    if db_path.parent.exists():
        return db_path.parent / "shinka_inspect_context.md"
    return Path("shinka_inspect_context.md")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n# ... truncated ...", True


def _required_columns(df: pd.DataFrame) -> None:
    required = {"id", "generation", "combined_score", "code", "correct"}
    missing = [col for col in sorted(required) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected Shinka programs table schema."
        )


def _rank_programs(df: pd.DataFrame, config: InspectConfig) -> tuple[pd.DataFrame, str]:
    working = df.copy()
    if config.min_generation is not None and "generation" in working.columns:
        working = working[pd.to_numeric(working["generation"], errors="coerce") >= config.min_generation]

    if working.empty:
        raise ValueError("No rows available after filtering.")

    working = working.dropna(subset=["combined_score"])
    if working.empty:
        raise ValueError("No rows with non-null combined_score.")

    correct_mask = working["correct"].map(_to_bool)
    correct_df = working[correct_mask]

    mode = "top-k-correct"
    selected_pool = correct_df
    if correct_df.empty:
        mode = "top-k-all-fallback-no-correct"
        selected_pool = working

    selected = selected_pool.sort_values("combined_score", ascending=False).head(config.k)
    if selected.empty:
        raise ValueError("Selection returned zero rows.")
    return selected, mode


def _short_id(value: Any, length: int = 8) -> str:
    text = str(value) if value is not None else "None"
    return text[:length]


def _render_markdown(
    selected_df: pd.DataFrame,
    source_db_path: Path,
    out_path: Path,
    config: InspectConfig,
    mode: str,
    total_rows: int,
    correct_rows: int,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Shinka Inspect Context Bundle")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Generated (UTC): `{now}`")
    lines.append(f"- Source DB: `{source_db_path}`")
    lines.append(f"- Selection mode: `{mode}`")
    lines.append(f"- Total rows loaded: `{total_rows}`")
    lines.append(f"- Correct rows loaded: `{correct_rows}`")
    lines.append(f"- Requested k: `{config.k}`")
    lines.append(f"- Included rows: `{len(selected_df)}`")
    lines.append(f"- Output file: `{out_path}`")
    if config.min_generation is not None:
        lines.append(f"- Min generation filter: `{config.min_generation}`")
    lines.append("")

    if mode == "top-k-all-fallback-no-correct":
        lines.append(
            "> WARNING: No correct programs found. Fallback to top-k by score across all rows."
        )
        lines.append("")

    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | ID | Gen | Score | Correct | Parent | Language |")
    lines.append("|---:|---|---:|---:|:---:|---|---|")
    for rank, (_, row) in enumerate(selected_df.iterrows(), start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{_short_id(row.get('id'))} | "
            f"{int(row.get('generation')) if pd.notna(row.get('generation')) else 'NA'} | "
            f"{float(row.get('combined_score')):.6f} | "
            f"{'Y' if _to_bool(row.get('correct')) else 'N'} | "
            f"{_short_id(row.get('parent_id'))} | "
            f"{row.get('language', 'NA')} |"
        )
    lines.append("")

    lines.append("## Program Details")
    lines.append("")
    for rank, (_, row) in enumerate(selected_df.iterrows(), start=1):
        program_id = row.get("id")
        generation = row.get("generation")
        score = row.get("combined_score")
        parent_id = row.get("parent_id")
        language = row.get("language", "python")
        code = str(row.get("code") or "")
        code_block, was_truncated = _truncate(code, config.max_code_chars)

        lines.append(f"### Rank {rank} - Program `{program_id}`")
        lines.append(f"- Generation: `{generation}`")
        lines.append(f"- Combined score: `{float(score):.6f}`")
        lines.append(f"- Correct: `{'true' if _to_bool(row.get('correct')) else 'false'}`")
        lines.append(f"- Parent: `{parent_id}`")
        lines.append(f"- Language: `{language}`")
        if was_truncated:
            lines.append(
                f"- Code truncated to `{config.max_code_chars}` chars for context fit."
            )

        if config.include_feedback:
            feedback = str(row.get("text_feedback") or "").strip()
            if feedback:
                lines.append("")
                lines.append("Feedback:")
                lines.append("```text")
                lines.append(feedback)
                lines.append("```")

        lines.append("")
        lines.append("Code:")
        lines.append(f"```{language if isinstance(language, str) else ''}")
        lines.append(code_block)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    config = _parse_args()
    db_path = _resolve_db_path(config.results_dir)
    output_path = _resolve_output_path(config, db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_programs_to_df(str(db_path), verbose=False)
    programs_df = loaded[0] if isinstance(loaded, tuple) else loaded
    if programs_df is None:
        raise RuntimeError(f"Failed to load programs DataFrame from: {db_path}")
    if programs_df.empty:
        raise ValueError(f"No program rows found in database: {db_path}")

    _required_columns(programs_df)
    total_rows = len(programs_df)
    correct_rows = int(programs_df["correct"].map(_to_bool).sum())

    selected_df, mode = _rank_programs(programs_df, config)
    markdown = _render_markdown(
        selected_df=selected_df,
        source_db_path=db_path,
        out_path=output_path,
        config=config,
        mode=mode,
        total_rows=total_rows,
        correct_rows=correct_rows,
    )
    output_path.write_text(markdown, encoding="utf-8")

    print(
        "Shinka Inspect complete: "
        f"rows={total_rows}, correct={correct_rows}, selected={len(selected_df)}, "
        f"mode={mode}, out={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
