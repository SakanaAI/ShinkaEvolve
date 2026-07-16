"""Build the self-contained RTLLM problem JSONL + per-design seeds.

Reads a local clone of RTLLM v2.0 (https://github.com/hkust-zhiyao/RTLLM) and
emits, for each selected design:
  - problems/rtllm.jsonl      : {design, spec, testbench, reference, top names}
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


def _tb_instantiates(tb: str, name: str) -> bool:
    """True if the testbench instantiates a module named `name` (with/without params)."""
    return re.search(r"\b" + re.escape(name) + r"\s*(?:#\s*\(|\w+\s*\()", tb) is not None


def _seed_top(name: str, ref_mod: str, tb: str) -> str:
    """The module name the testbench expects -- the only reliable source.

    Folder names can be misspelled ("substractor") and reference roots can differ
    from the design name ("verified_adder_64bit" for adder_pipe_64bit), so pick the
    first candidate the testbench actually instantiates.
    """
    strip = ref_mod[len("verified_"):] if ref_mod.startswith("verified_") else ref_mod
    for cand in (name, strip, ref_mod):
        if _tb_instantiates(tb, cand):
            return cand
    return strip


# Auxiliary data files a testbench may $readmemh/$readmemb (kept self-contained).
_AUX_SUFFIXES = (".txt", ".mem", ".hex", ".dat", ".bin", ".vh")


def _aux_files(design_dir: Path) -> dict:
    aux = {}
    for f in design_dir.iterdir():
        if (f.is_file() and f.suffix.lower() in _AUX_SUFFIXES
                and f.name != "design_description.txt"):
            aux[f.name] = f.read_text(encoding="utf-8", errors="replace")
    return aux


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


def _wrap_seed(body: str) -> str:
    """Wrap the reference in EVOLVE-BLOCK markers, FREEZING the module's interface.

    RTLLM pins the module name + all I/O signals (name and width) as part of the spec,
    so the port declaration must stay fixed: evolution may only change the implementation,
    never the interface. We place the top module's ``module NAME( ...ports... );`` header
    OUTSIDE the editable block; only the body evolves. (Without this the model can shrink a
    port to win on a testbench that never exercises the dropped bits -- e.g. narrowing a
    RAM address bus -- which is not a valid optimisation.)
    """
    # Mask comments (keeping length so offsets stay aligned) so a ';' or ')' inside a
    # header comment doesn't fool the matcher.
    def _mask(s):
        s = re.sub(r"/\*.*?\*/", lambda mm: " " * len(mm.group()), s, flags=re.S)
        return re.sub(r"//[^\n]*", lambda mm: " " * len(mm.group()), s)
    masked = _mask(body)
    m = re.search(r"\bmodule\b[^;]*?\)\s*;", masked)
    if not m:  # unparsable header: fall back to whole-module (still synthesises)
        return f"// EVOLVE-BLOCK-START\n{body.rstrip()}\n// EVOLVE-BLOCK-END\n"
    end = m.end()
    # Non-ANSI style declares port directions/widths AFTER the name list
    # (module X(a,b); input [7:0] a; ...). Freeze those too, so widths are immutable.
    lines, mlines = body[end:].splitlines(keepends=True), masked[end:].splitlines(keepends=True)
    i = 0
    while i < len(lines) and (not mlines[i].strip()
                              or re.match(r"(input|output|inout)\b", mlines[i].lstrip())):
        i += 1
    header = body[:end].rstrip() + ("\n" + "".join(lines[:i]).rstrip() if i else "")
    rest = "".join(lines[i:]).strip()
    return f"{header.rstrip()}\n// EVOLVE-BLOCK-START\n{rest}\n// EVOLVE-BLOCK-END\n"


def main():
    ap = argparse.ArgumentParser(description="Extract RTLLM designs into a Shinka JSONL")
    ap.add_argument("--rtllm-root", required=True, type=Path,
                    help="path to a local RTLLM v2.0 clone")
    ap.add_argument("--designs", nargs="*", default=list(DEFAULT_DESIGNS),
                    help="design names to extract (default: the prototype set)")
    ap.add_argument("--out", default="problems/rtllm.jsonl")
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
        # Name the seed exactly what the testbench instantiates (robust to misspelled
        # folders and reference roots that differ from the design name).
        top = _seed_top(name, ref_mod, tb)
        rows.append({
            "design_name": name, "category": rel.split("/")[0], "top_module": top,
            "ref_module": ref_mod, "tb_module": tb_mod,
            # The evaluator computes the actual equivalence verdict at runtime; this is
            # only a hint for downstream tooling.
            "verification": "formal",
            "description": desc, "testbench": tb, "reference": ref,
            "aux_files": _aux_files(d),
        })
        # seed: rename the reference's root module to the bare (tb-instantiated) name
        body = ref.replace(ref_mod, top)
        seed = _wrap_seed(body)
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
