from __future__ import annotations

import sys
from pathlib import Path


SUMMARY = """# Individual Summary

- Schema-Version: repo-individual-v1
- Individual: fake-inference
- Generation: 1
- Commit: pending

## Parent

Fake agent started from the seed inference pipeline.

## Core Idea

Use the exact affine transform expected by the evaluator.

## Lineage Context

This is a deterministic smoke-test mutation.

## Changed Files

- src/pipeline.py

## Validation Performed

Relies on the Shinka evaluator after mutation.

## Performance Hypothesis

Correct affine outputs should maximize correctness and keep latency low.

## Risks and Followups

- This fake agent is only for tests.

## Minimal Snippets

- return [2.0 * x + 1.0 for x in values]
"""


def _arg_value(name: str) -> str:
    if name not in sys.argv:
        raise SystemExit(f"missing {name}")
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    if "--check" in sys.argv:
        return
    work_dir = Path(_arg_value("--work-dir"))
    prompt_path = Path(_arg_value("--prompt-file"))
    if not prompt_path.exists():
        raise SystemExit(f"missing prompt: {prompt_path}")
    pipeline_path = work_dir / "src" / "pipeline.py"
    pipeline_path.write_text(
        "from __future__ import annotations\n\n"
        "def predict(values: list[float]) -> list[float]:\n"
        "    return [2.0 * x + 1.0 for x in values]\n",
        encoding="utf-8",
    )
    summary_path = work_dir / ".shinka" / "individual.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(SUMMARY, encoding="utf-8")
    print("fake headless mutation complete")


if __name__ == "__main__":
    main()
