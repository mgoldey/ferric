#!/usr/bin/env python3
"""Aggregate PDEP-truncation trust-map runs (water/ethylene/benzene) into a
cross-system comparison: how does each observable's truncation sensitivity scale
with system size? Parses the `[spike] thresh=...` lines from each run's stdout.

Usage: aggregate_trustmap.py <out1.out> <out2.out> ...
   or: aggregate_trustmap.py            (auto-globs scripts/queue/out/trunc_*.out)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

SPIKE = re.compile(
    r"\[spike\] thresh=(?P<th>[-+0-9.eE]+)\s+E_rpa=(?P<e>[-+0-9.eE]+)\s+"
    r"IP=(?P<ip>[-+0-9.eE]+)\s+EA=(?P<ea>[-+0-9.eE]+)\s+a=(?P<a>[-+0-9.eE]+)\s+"
    r"C6=(?P<c6>[-+0-9.eE]+)\s+evGW_IP=(?P<gw>[-+0-9.eE]+)"
)
NAME = re.compile(r"\[spike\]\s+(\S+?)/(\S+?)\s+aux")


def parse(path):
    txt = Path(path).read_text()
    m = NAME.search(txt)
    name = m.group(1) if m else Path(path).stem
    rows = [mm.groupdict() for mm in SPIKE.finditer(txt)]
    if not rows:
        return None
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    return name, rows


def main():
    files = sys.argv[1:] or sorted(str(p) for p in OUT.glob("trunc_*aug-cc-pvdz*.out"))
    runs = [r for r in (parse(f) for f in files) if r]
    if not runs:
        print("no parseable runs found")
        return
    print("# PDEP-truncation sensitivity vs system size")
    print("# signed/relative change at the LARGEST swept thresh (default-regime end) vs thresh=0")
    print(f"{'system':10} {'natom?':>6} | {'dE(mHa)':>9} {'dIP(eV)':>9} "
          f"{'dEA(eV)':>9} {'da(%)':>8} {'dC6(%)':>8} {'devGW(eV)':>10}")
    print("-" * 78)
    for name, rows in runs:
        r0 = rows[0]
        # find the thresh closest to 0.1
        rmax = max(rows, key=lambda r: r["th"])
        d_e = (rmax["e"] - r0["e"]) * 1000.0
        d_ip = rmax["ip"] - r0["ip"]
        d_ea = rmax["ea"] - r0["ea"]
        d_a = 100.0 * (rmax["a"] - r0["a"]) / r0["a"] if r0["a"] else float("nan")
        d_c6 = 100.0 * (rmax["c6"] - r0["c6"]) / r0["c6"] if r0["c6"] else float("nan")
        d_gw = rmax["gw"] - r0["gw"]
        print(f"{name:10} {'-':>6} | {d_e:9.3f} {d_ip:9.4f} {d_ea:9.4f} "
              f"{d_a:8.3f} {d_c6:8.3f} {d_gw:10.4f}")
    print()
    print("# absolute thresh=0 reference per system:")
    for name, rows in runs:
        r0 = rows[0]
        print(f"  {name:10} E_rpa={r0['e']:.5f}  IP={r0['ip']:.3f}  "
              f"a={r0['a']:.3f}  C6={r0['c6']:.2f}  evGW={r0['gw']:.3f}")


if __name__ == "__main__":
    main()
