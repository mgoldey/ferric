#!/usr/bin/env python3
"""Aggregate PDEP-truncation trust-map runs (water/ethylene/benzene) into a
cross-system view of the central question: at the production 1e-4 threshold, how
many eigenpotentials does each system DROP, and is it still lossless? Plus how
hard you can compress before any observable moves.

Parses the formatted trust-map TABLE (which carries M_kept/naux) from each run's
stdout, not the [spike] lines.

Usage: aggregate_trustmap.py <out1> <out2> ...
   or: aggregate_trustmap.py            (auto-globs scripts/queue/out/trunc_*FINEGRID*.out)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

NAME = re.compile(r"trust map: (\S+?)/(\S+)")
# table row:  "    1e-4  88/118 (75%) |   4.04e-7   0.0000  0.0000   0.000   0.000"
ROW = re.compile(
    r"^\s*(?P<th>[-+0-9.eE]+)\s+(?P<mk>\d+)/(?P<na>\d+)\s+\(\d+%\)\s+\|"
    r"\s+(?P<de>[-+0-9.eE]+)\s+(?P<dip>[-+0-9.eE]+)\s+(?P<dea>[-+0-9.eE]+)"
    r"\s+(?P<da>[-+0-9.eE]+)\s+(?P<dc6>[-+0-9.eE]+)"
)
GWABS = re.compile(r"absolute GW IP at thresh=0.*evGW=([-+0-9.eE]+)")
NATOM = {"water": 3, "ethylene": 6, "benzene": 12}


def parse(path):
    txt = Path(path).read_text()
    m = NAME.search(txt)
    name = m.group(1) if m else Path(path).stem
    rows = []
    for line in txt.splitlines():
        mm = ROW.match(line)
        if mm:
            d = {k: float(v) for k, v in mm.groupdict().items()}
            d["mk"], d["na"] = int(d["mk"]), int(d["na"])
            rows.append(d)
    if not rows:
        return None
    gw = GWABS.search(txt)
    return name, rows, (float(gw.group(1)) if gw else float("nan"))


def at_thresh(rows, target):
    """Row whose thresh is closest to target."""
    return min(rows, key=lambda r: abs(r["th"] - target))


def main():
    files = sys.argv[1:] or sorted(str(p) for p in OUT.glob("trunc_*FINEGRID*.out"))
    runs = [r for r in (parse(f) for f in files) if r]
    if not runs:
        print("no parseable runs found")
        return
    runs.sort(key=lambda r: NATOM.get(r[0], 99))

    print("# PDEP-truncation: compression vs accuracy across system size")
    print("#")
    print("# At the PRODUCTION 1e-4 default — modes kept and accuracy cost:")
    print(f"{'system':10} {'atoms':>5} {'naux':>5} {'M@1e-4':>7} {'kept%':>6} "
          f"| {'dE(µHa)':>8} {'dIP(meV)':>9} {'da(%)':>7} {'dC6(%)':>7}")
    print("-" * 80)
    for name, rows, _gw in runs:
        r = at_thresh(rows, 1e-4)
        kept = 100.0 * r["mk"] / r["na"]
        print(f"{name:10} {NATOM.get(name,0):>5} {r['na']:>5} {r['mk']:>7} {kept:>5.0f}% "
              f"| {r['de']*1e6:>8.2f} {r['dip']*1e3:>9.3f} {r['da']:>7.3f} {r['dc6']:>7.3f}")

    print("\n# How hard can you compress? (largest swept thresh = aggressive end)")
    print(f"{'system':10} {'M@max':>6} {'kept%':>6} {'thresh':>7} "
          f"| {'dE(mHa)':>8} {'da(%)':>7} {'dC6(%)':>7}")
    print("-" * 66)
    for name, rows, _gw in runs:
        r = max(rows, key=lambda x: x["th"])
        kept = 100.0 * r["mk"] / r["na"]
        print(f"{name:10} {r['mk']:>6} {kept:>5.0f}% {r['th']:>7.0e} "
              f"| {r['de']*1e3:>8.3f} {r['da']:>7.3f} {r['dc6']:>7.3f}")

    print("\n# Reading: at 1e-4, larger systems drop a LARGER mode fraction at the")
    print("# same (near-zero) accuracy cost — which is exactly where PDEP truncation")
    print("# is supposed to pay off. Compression scales with size; error does not.")


if __name__ == "__main__":
    main()
