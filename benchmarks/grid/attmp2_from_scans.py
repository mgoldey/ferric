#!/usr/bin/env python3
"""Attenuated-MP2 MAE vs r0, extracted free from the B/T scan intermediates.

The rs-mp2-rpa driver prints `E(SR-MP2, erfc)` at every r0 alongside the T
components. RHF + that term IS attenuated MP2 at the scan's basis and CP
convention, so the whole attMP2 r0 curve comes out of scans already run --
no new compute.

THREE CAVEATS, all of which change how the number may be quoted:

1. **This is terfc, matching the published operator.** The T scans set
   `attenuator = "terf"`, which selects terf/terfc -- so the short-range MP2
   here IS terfc-attenuated, the same operator as published attMP2(terfc).
   The CLI used to print the component as `E(SR-MP2, erfc)`, a hardcoded
   label that was wrong for every terf-split run; fixed 2026-07-26.
   What still differs from the literature: this is aQZ with CP, on this
   scan's r0 grid, whereas the published parameters are r0 = 1.35 A (no-CP)
   / 1.75 A (CP) at aTZ, and 1.50 A (no-CP) at aQZ.

2. **The limits are HF and MP2.** As r0 -> 0 the kernel vanishes (HF); as
   r0 -> inf it becomes 1/r (full MP2). A minimum below BOTH endpoints is the
   published attMP2 result and the structural precedent for T's own dip
   between MP2 and MP2+dRPA.

3. **Coverage.** Only systems with all three CP fragments at a given r0 are
   counted, and `n` is printed per row -- MAEs at different n are not
   comparable.

Usage:  python3 attmp2_from_scans.py [--basis aqz] [--systems 12,15,20,...]
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mae_spline as M  # noqa: E402

FRAGS = ("dimer", "mA_cp", "mB_cp")
TAGS = ("r0tscan", "r0text", "r0tscanB")


def att_totals(sysid, frag, basis):
    """{r0: E_total(attMP2)} across every known scan tag."""
    out = {}
    for tag in TAGS:
        p = ROOT / "out" / f"a24-{sysid}_{frag}_{basis}_{tag}_T.out"
        if not p.exists():
            continue
        parts = re.split(r"RS-MP2-RPA \[terf split\] \(r0 = ([\d.]+)", p.read_text())
        for i in range(1, len(parts), 2):
            r0, blk = float(parts[i]), parts[i + 1]

            def g(pat):
                m = re.search(pat + r"\s*=\s*(-?[\d.]+)", blk)
                return float(m.group(1)) if m else None

            # Accept BOTH spellings. The CLI label was hardcoded "erfc" until
            # 2026-07-26, when it was fixed to follow the attenuator (terf ->
            # terfc). Outputs produced before and after that rebuild coexist in
            # out/ and are the SAME quantity -- the fix changed only the name.
            # Matching one spelling silently drops half the data: it made this
            # tool report n=2 where the T curve had n=3, from identical scans.
            tot, e_t = g(r"Total energy"), g(r"E_corr coupled \(T\)")
            sr = g(r"E\(SR-MP2, terfc\)")
            if sr is None:
                sr = g(r"E\(SR-MP2, erfc\)")
            if None in (tot, e_t, sr):
                continue
            out[round(r0, 4)] = (tot - e_t) + sr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="aqz")
    ap.add_argument("--systems", default="12,15,20,21,22,24")
    a = ap.parse_args()
    systems = [int(x) for x in a.systems.split(",") if x.strip()]
    bind = M.load_bind()

    per_r0 = {}
    for s in systems:
        f = {x: att_totals(s, x, a.basis) for x in FRAGS}
        for r in sorted(set(f["dimer"]) & set(f["mA_cp"]) & set(f["mB_cp"])):
            ie = (f["dimer"][r] - f["mA_cp"][r] - f["mB_cp"][r]) * M.K
            per_r0.setdefault(r, {})[s] = ie - bind[s]

    print(f"attMP2(erfc) / {a.basis} / CP  — from B/T scan intermediates")
    print(f"requested systems: {systems}\n")
    print(f"{'r0 (A)':>8} {'n':>3} {'MAE':>9}")
    full = []
    for r in sorted(per_r0):
        errs = per_r0[r]
        mae = sum(abs(v) for v in errs.values()) / len(errs)
        mark = "" if len(errs) == len(systems) else "   (partial)"
        print(f"{r:>8.2f} {len(errs):>3} {mae:>9.4f}{mark}")
        if len(errs) == len(systems):
            full.append((r, mae))
    if full:
        b = min(full, key=lambda t: t[1])
        print(f"\nbest sampled (full coverage): r0 = {b[0]:.2f} A, MAE {b[1]:.4f}")
        if b[0] == max(r for r, _ in full):
            print("BOUNDARY — still falling at the edge; the minimum is beyond "
                  "the sampled range. Do NOT quote this as the optimum.")
    print("\nNOTE: terfc (same operator as published attMP2), but aQZ/CP on this")
    print("      scan's r0 grid — published aQZ r0 is 1.50 A and is NO-CP.")


if __name__ == "__main__":
    sys.exit(main())
