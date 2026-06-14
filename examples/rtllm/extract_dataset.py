"""Build the self-contained RTLLM problem JSONL + per-design seeds.

Reads a local clone of RTLLM v2.0 (https://github.com/hkust-zhiyao/RTLLM) and
emits, for each selected design:
  - problems/rtllm_proto.jsonl : {design, spec, testbench, reference, top names}
  - seeds/<design>/initial.sv  : the RTLLM reference renamed to the design's
                                 bare module name, wrapped in EVOLVE-BLOCK markers
                                 (a correct seed that scores exactly 100).

We do not commit RTLLM's files; regenerate locally from your clone.

Usage:
    python extract_dataset.py --rtllm-root /path/to/RTLLM
    python extract_dataset.py --rtllm-root /path/to/RTLLM --designs adder_8bit multi_8bit
"""

import argparse
import json
import re
from pathlib import Path

# Default prototype set: combinational arithmetic with real PPA headroom.
DEFAULT_DESIGNS = {
    "adder_8bit":  "Arithmetic/Adder/adder_8bit",
    "adder_32bit": "Arithmetic/Adder/adder_32bit",
    "multi_8bit":  "Arithmetic/Multiplier/multi_8bit",
}


def _declared_modules(src: str) -> list[str]:
    return re.findall(r"^\s*module\s+([A-Za-z0-9_]+)", src, re.M)


def _root_module(src: str) -> str:
    """The module not instantiated by any other module in the file."""
    mods = _declared_modules(src)
    instantiated = set()
    for m in mods:
        for mt in re.finditer(r"\b" + re.escape(m) + r"\s+[A-Za-z0-9_]+\s*\(", src):
            if not src[max(0, mt.start() - 10):mt.start()].rstrip().endswith("module"):
                instantiated.add(m)
                break
    roots = [m for m in mods if m not in instantiated]
    return roots[0] if roots else mods[0]


def _discover(rtllm_root: Path) -> dict[str, str]:
    found = {}
    for desc in rtllm_root.rglob("design_description.txt"):
        rel = desc.parent.relative_to(rtllm_root).as_posix()
        found[desc.parent.name] = rel
    return found


def main():
    ap = argparse.ArgumentParser(description="Extract RTLLM designs into a Shinka JSONL")
    ap.add_argument("--rtllm-root", required=True, type=Path,
                    help="path to a local RTLLM v2.0 clone")
    ap.add_argument("--designs", nargs="*", default=list(DEFAULT_DESIGNS),
                    help="design names to extract (default: the prototype set)")
    ap.add_argument("--out", default="problems/rtllm_proto.jsonl")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    catalog = _discover(args.rtllm_root)
    catalog.update(DEFAULT_DESIGNS)  # prefer canonical paths for the prototype set

    rows = []
    for name in args.designs:
        rel = catalog.get(name)
        if rel is None:
            print(f"  ! design '{name}' not found under {args.rtllm_root}; skipping")
            continue
        d = args.rtllm_root / rel
        desc = (d / "design_description.txt").read_text(encoding="utf-8", errors="replace")
        tb = (d / "testbench.v").read_text(encoding="utf-8", errors="replace")
        ref_files = list(d.glob("verified_*.v"))
        if not ref_files:
            print(f"  ! no verified_*.v for '{name}'; skipping")
            continue
        ref = ref_files[0].read_text(encoding="utf-8", errors="replace")
        ref_mod = _root_module(ref)
        tb_mod = _declared_modules(tb)[0]
        rows.append({
            "design_name": name, "category": rel.split("/")[0], "top_module": name,
            "ref_module": ref_mod, "tb_module": tb_mod,
            "description": desc, "testbench": tb, "reference": ref,
        })
        # seed: rename the reference's root module to the bare design name
        body = ref.replace(ref_mod, name)
        seed = f"// EVOLVE-BLOCK-START\n{body.rstrip()}\n// EVOLVE-BLOCK-END\n"
        sd = here / "seeds" / name
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "initial.sv").write_text(seed, encoding="utf-8")
        print(f"  {name:14} ref_top={ref_mod:24} tb_top={tb_mod}")

    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # reset the reference-PPA cache so it is recomputed against fresh data
    (here / "problems" / ".ppa_ref_cache.json").unlink(missing_ok=True)
    print(f"\nWrote {len(rows)} designs to {out}")


if __name__ == "__main__":
    main()
