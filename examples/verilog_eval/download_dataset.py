"""Download VerilogEval v2 spec-to-rtl dataset from HuggingFace.

Downloads the 156-problem benchmark and saves as JSONL for use with
ShinkaEvolve's evaluator.

Usage:
    python download_dataset.py                          # All 156 problems
    python download_dataset.py --output problems/custom.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install the 'requests' library: pip install requests")
    sys.exit(1)


DATASET_URL = (
    "https://huggingface.co/datasets/dakies/nvlabs-verilogeval-v2-spec-to-rtl"
    "/resolve/main/data/test-00000-of-00001.parquet"
)


def download_dataset() -> list[dict]:
    """Download VerilogEval v2 parquet from HuggingFace and parse into dicts."""
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas: pip install pandas pyarrow")
        sys.exit(1)

    print(f"Downloading VerilogEval v2 dataset...")
    response = requests.get(DATASET_URL)
    response.raise_for_status()

    import io
    df = pd.read_parquet(io.BytesIO(response.content))
    problems = df.to_dict(orient="records")
    print(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Download VerilogEval v2 dataset")
    parser.add_argument(
        "--output", type=str, default="problems/verilog_eval.jsonl",
        help="Output JSONL path (default: problems/verilog_eval.jsonl)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all problem IDs without downloading",
    )
    args = parser.parse_args()

    problems = download_dataset()

    if args.list:
        for p in problems:
            print(p["problem_id"])
        return

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            row = {
                "problem_id": p["problem_id"],
                "prompt": p["prompt"],
                "ref": p["ref"],
                "test": p["test"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(problems)} problems to {out_path}")
    print(f"\nExample problem IDs:")
    for p in problems[:10]:
        print(f"  {p['problem_id']}")
    print(f"  ... ({len(problems)} total)")


if __name__ == "__main__":
    main()
