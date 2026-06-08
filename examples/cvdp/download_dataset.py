"""Download CVDP benchmark dataset from HuggingFace.

Downloads the no-commercial, non-agentic code generation split and
optionally filters by category or difficulty.

Usage:
    python download_dataset.py                          # All 302 problems
    python download_dataset.py --difficulty medium      # Medium only (~140)
    python download_dataset.py --category cid003        # Spec-to-RTL only
    python download_dataset.py --difficulty medium --category cid003
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("Install the 'requests' library: pip install requests")
    sys.exit(1)


DATASET_BASE = (
    "https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset"
    "/resolve/main"
)
DATASET_VERSION = "v1.1.0"


def download_dataset(commercial: bool = False) -> List[Dict[str, Any]]:
    """Download CVDP dataset JSONL from HuggingFace."""
    license_tag = "commercial" if commercial else "no_commercial"
    filename = f"cvdp_{DATASET_VERSION}_nonagentic_code_generation_{license_tag}.jsonl"
    url = f"{DATASET_BASE}/{filename}"

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    problems = []
    for line in response.iter_lines(decode_unicode=True):
        line = line.strip()
        if line:
            problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Download CVDP dataset from HuggingFace")
    parser.add_argument(
        "--output", type=str, default="problems/cvdp_full.jsonl",
        help="Output JSONL path (default: problems/cvdp_full.jsonl)",
    )
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by category ID (e.g., cid003, cid016)",
    )
    args = parser.parse_args()
    
    try:
        all_problems = download_dataset()
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)
    
    # Filter problems
    problems = []
    for p in all_problems:
        cats = p["categories"]
        if args.difficulty and cats[1] != args.difficulty:
            continue
        if args.category and cats[0] != args.category:
            continue
        problems.append(p)
    
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(problems)} problems to {out_path}")
    
    from collections import Counter
    cats = Counter(p["categories"][0] for p in problems)
    diffs = Counter(p["categories"][1] for p in problems)
    print(f"\nBy category: {dict(sorted(cats.items()))}")
    print(f"By difficulty: {dict(sorted(diffs.items()))}")


if __name__ == "__main__":
    main()
