#!/usr/bin/env python3
"""Degeneracy and tightness of the three bounds across the size axis.

Companion to ``scripts/ipb_distance_proto.py``.  This script answers the
pre-registration's INERT question and nothing else, so it can run on the size
axis cheaply: it needs no SCF and no K build, only the bounds and the grid
batching, and every number it prints is a deterministic count.

The pre-registered inert criterion (``scripts/queue/out/ipb_distance_prereg.md``
§4): if IPB-D's degenerate fraction lands in the same 45-80% band that afflicts
ferric's sphere bound (whitepaper §6.2b), the new bound has inherited the same
disease and the lane closes.

"Degenerate" means the bound falls back to its distance-free value for a given
(shell pair, grid batch): for ferric that is the ``R = 0`` total; for IPB-D it is
``V_0``, the batch-independent IPB.  Both are measured on the SAME decisions.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ipb_distance_proto import (          # noqa: E402
    BATCH_POINTS, BoundsFerric, BoundsIPB, batch_sphere, build_grid,
    molecular_diameter, _atom_spec,
)
from pyscf import gto                      # noqa: E402
from pyscf import lib as pyscf_lib         # noqa: E402

pyscf_lib.num_threads(1)


def scan(name, basis, atom_grid):
    mol = gto.M(atom=_atom_spec(name), basis=basis, unit="Angstrom", verbose=0)
    coords, _ = build_grid(mol, atom_grid=atom_grid)
    bfe, bipb = BoundsFerric(mol), BoundsIPB(mol)
    di = df = dboth = tot = 0
    gain_flat = []          # IPB-flat / IPB-D  -- does the distance factor bite?
    gain_fe = []            # ferric / IPB-D    -- is IPB-D the tighter bound?
    for b0 in range(0, coords.shape[0], BATCH_POINTS):
        centre, radius = batch_sphere(coords[b0:b0 + BATCH_POINTS])
        for s1 in range(mol.nbas):
            for s2 in range(s1 + 1):
                tot += 1
                gi = bipb.degenerate(s1, s2, centre, radius)
                gf = bfe.degenerate(s1, s2, centre, radius)
                di += gi
                df += gf
                dboth += (gi and gf)
                d = bipb.distance(s1, s2, centre, radius)
                f = bipb.flat(s1, s2)
                fe = bfe.sphere(s1, s2, centre, radius)
                if d > 0.0:
                    gain_flat.append(f / d)
                    gain_fe.append(fe / d)
    gf_, gfe = np.array(gain_flat), np.array(gain_fe)
    return dict(
        name=name, diam=molecular_diameter(mol), nsh=mol.nbas, nbf=mol.nao,
        nbatch=math.ceil(coords.shape[0] / BATCH_POINTS), tot=tot,
        deg_ipb=di / tot, deg_fe=df / tot, deg_both=dboth / tot,
        gflat50=np.median(gf_), gflat90=np.percentile(gf_, 90), gflatmax=gf_.max(),
        gfe50=np.median(gfe), gfe90=np.percentile(gfe, 90),
        gfe_frac_better=float(np.mean(gfe > 1.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="alkane_1,alkane_2,alkane_4,alkane_8,"
                                         "alkane_12,alkane_16")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--grid", default="25,50")
    args = ap.parse_args()
    grid = tuple(int(x) for x in args.grid.split(","))

    print(f"basis={args.basis}  grid={grid}  batch={BATCH_POINTS}")
    print(f"{'system':<11}{'diam':>7}{'nsh':>5}{'nbat':>6}{'decisions':>11}"
          f"{'deg IPB-D':>11}{'deg ferric':>11}{'deg both':>10}"
          f"{'flat/D p50':>11}{'p90':>8}{'max':>9}"
          f"{'fe/D p50':>10}{'p90':>9}{'fe worse':>10}")
    for name in args.systems.split(","):
        r = scan(name, args.basis, grid)
        print(f"{r['name']:<11}{r['diam']:7.2f}{r['nsh']:5d}{r['nbatch']:6d}"
              f"{r['tot']:11d}{100*r['deg_ipb']:10.2f}%{100*r['deg_fe']:10.2f}%"
              f"{100*r['deg_both']:9.2f}%"
              f"{r['gflat50']:11.3g}{r['gflat90']:8.3g}{r['gflatmax']:9.3g}"
              f"{r['gfe50']:10.3g}{r['gfe90']:9.3g}"
              f"{100*r['gfe_frac_better']:9.1f}%", flush=True)


if __name__ == "__main__":
    main()
