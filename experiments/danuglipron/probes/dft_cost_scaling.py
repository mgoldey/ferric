#!/usr/bin/env python
"""Separate ferric KS-DFT's one-time setup cost from its per-iteration cost.

Tier 4 is unaffordable at drug scale (RESULTS.md M10) and the cause is still
undiagnosed. The single most useful next measurement is which HALF the cost
lives in, because the two candidates have opposite fixes:

  * one-time setup  -> grid construction, which is O(natoms^3)
    (`becke_weights_all` is O(natoms^2) per point, and points scale with
    atoms). Fix: a coarser grid, or pruning.
  * per-iteration   -> the XC pass / Vxc GEMMs over ~577k points. Fix:
    a smaller grid or a cheaper functional; the basis barely matters.

DO NOT try to get this from `max_iter=1` vs `max_iter=3`. That was the first
attempt and it is INVALID: `run_dft` walks a level-shift *ladder*
(`ferric_scf::ladder::ksdft_ladder`), each rung a full SCF whose iteration
count `max_iter` bounds individually. A low `max_iter` makes every rung fail
early and the ladder try MORE rungs, so t(1) can exceed t(3) -- measured
33.5 s vs 22.6 s on alkane_10, implying a nonsensical negative per-iteration
cost. `max_iter` is not a knob on total work.

What this probe does instead: hold the method fixed and sweep ATOM COUNT
across a homologous series. Grid setup scales as O(natoms^3) while the XC pass
scales ~linearly in points, so fitting log(t) vs log(natoms) separates them --
an exponent near 3 implicates setup, near 1 implicates the per-iteration XC
work. Same molecule family throughout, so electronic structure is not a
confound.

RUN THIS ON AN UNCONTENDED BOX. Timings taken while the load average exceeds
the core count are not evidence -- that mistake was made, and discarded, on
2026-09-02. Check `uptime` and `/proc/pressure/memory` first, and always
launch under `scripts/ferric-limited` so a runaway job cannot take the box
down with it.

Usage (one point of the series; run across alkane_5/10/15/20 and fit):
    scripts/ferric-limited -- uv run --no-sync python \
        experiments/danuglipron/probes/dft_cost_split.py <xyz> <charge> [basis]
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    import ferric

    path, charge = sys.argv[1], int(sys.argv[2])
    basis = sys.argv[3] if len(sys.argv) > 3 else "sto-3g"

    mol = ferric.Molecule.from_xyz(path, charge, 1)
    bs = ferric.BasisSet.bundled(basis)

    t0 = time.time()
    res = ferric.run_dft(mol, bs, functional="PBE")
    dt = time.time() - t0
    print(f"{path.split('/')[-1]:20s} natoms={mol.natoms():3d} "
          f"nelec={mol.nelec():4d} basis={basis:9s} "
          f"{dt:8.1f} s  conv={res.converged}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
