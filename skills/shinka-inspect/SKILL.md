---
name: shinka-inspect
description: Load top-performing Shinka programs into agent context using `shinka.utils.load_programs_to_df`, and emit a compact Markdown bundle for iteration planning.
---

# Shinka Inspect Skill
Extract the strongest programs from a Shinka run and package them into a context file that coding agents can load directly.

## When to Use
Use this skill when:
- A run already produced a results directory and SQLite database
- You want to inspect top-performing programs before launching the next batch
- You want a compact context artifact instead of manually browsing the DB

Do not use this skill when:
- You still need to scaffold a task (`shinka-setup`)
- You need to run evolution batches (`shinka-run`)

## What it does
- Uses `shinka.utils.load_programs_to_df` to read program records
- Ranks programs by `combined_score`
- Selects top-`k` correct programs (fallback to top-`k` overall if no correct rows)
- Writes one Markdown bundle with metadata, ranking table, feedback, and code snippets

## Workflow
1. Confirm run artifacts exist
```bash
ls -la <results_dir>
```

2. Generate context bundle
```bash
python skills/shinka-inspect/scripts/inspect_best_programs.py \
  --results-dir <results_dir> \
  --k 5
```

3. Optional tuning knobs
```bash
python skills/shinka-inspect/scripts/inspect_best_programs.py \
  --results-dir <results_dir> \
  --k 8 \
  --max-code-chars 5000 \
  --min-generation 10 \
  --out <results_dir>/inspect/top_programs.md
```

4. Load output into agent context
- Default output path: `<results_dir>/shinka_inspect_context.md`
- Use it as the context artifact for next-step mutation planning

## CLI Arguments
- `--results-dir`: Path to run directory (or direct DB file path)
- `--k`: Number of programs to include (default `5`)
- `--out`: Output markdown path (default under results dir)
- `--max-code-chars`: Per-program code truncation cap (default `4000`)
- `--min-generation`: Optional lower bound on generation
- `--include-feedback` / `--no-include-feedback`: Include `text_feedback` blocks

## Notes
- Ranking metric is `combined_score`.
- If no correct rows exist, script falls back to top-score rows and labels fallback in output.
- Script is read-only for run artifacts (writes only the markdown bundle).
