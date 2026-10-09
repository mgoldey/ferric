"""Iteration 8: WHY the Becke partition converges slowly in a dense lattice, and which partition fixes it.

H2/STO-3G a=4 (138 image atoms within D=10 of every point).  Anchors per (scheme, grid):
  sum_g w_g - Omega   (integral of the constant 1: sees ONLY the partition's quadrature error),
  |int rho - N|, max|S_grid - S_latt|, Ex(LDA) - uniform reference (spectrally converged to 1e-12).
Predictions (stated before running):
  physics: the error is angular/radial quadrature of the fuzzy-cell faces, which in a crystal sit where the
    density is NOT small (unlike a molecule) -> it shrinks with grid size and shrinks FASTER for smoother
    partitions (becke1 < becke2 < becke3 face sharpness; 'exp' = Hirshfeld-like e^{-2r} smoothest).
  artifact: a broken partition of unity (image bookkeeping) -> sum w - Omega does NOT shrink with the grid
    and is the same size for every scheme.
Usage: python3 run_dft_partition.py [schemes] [grids] [h2|tri]"""

import sys
import time

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals
import pbc_dft as pd
from test_prototype import H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, SP_BASIS

schemes = (
    sys.argv[1].split(",")
    if len(sys.argv) > 1
    else ["becke", "becke2", "becke1", "ssf", "exp"]
)
grids = (
    [tuple(map(int, g.split("x"))) for g in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [(50, 146), (75, 302), (100, 590)]
)
which = sys.argv[3] if len(sys.argv) > 3 else "h2"
cell = (
    Cell(H2_A, H2_ATOMS, "sto-3g")
    if which == "h2"
    else Cell(TRI_A, TRI_ATOMS, SP_BASIS)
)
N = cell.mol.nelectron
MESH = (48,) if which == "h2" else (48, 64)
S = pd.lattice_overlap(cell)
D = pd.probe_density(cell, S)
ref = pd.grid_anchors(pd.uniform_grid(cell, MESH[-1], deriv=0), S, D, N)[2]
I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
jk = pd.dense_jk(I["I"])
rks = lambda g, xc: pd.rks(
    I["S"], I["h"], jk, I["enn"], N, g, xc, kshift=I["madelung"], conv=1e-12
)[0]
for m in MESH:  # last mesh is the reference; h2 == PySCF pbc RKS UniformGrids 40^3/60^3 (see FINDINGS)
    gu = pd.uniform_grid(cell, m)
    eref = {xc: rks(gu, xc) for xc in ("LDA,VWN", "PBE0")}
    print(
        f"uniform-{m}^3 RKS: LDA {eref['LDA,VWN']:.12f}  PBE0 {eref['PBE0']:.12f}",
        flush=True,
    )
    del gu
print(f"{which}: Omega {cell.vol}  Ex ref (uniform 48^3) {ref:.12f}", flush=True)
print(
    "scheme  grid     npts   sum w - Omega   |dN|     max|dS|   Ex-ref   E_LDA-ref  E_PBE0-ref",
    flush=True,
)
for sc in schemes:
    for nr, na in grids:
        t = time.time()
        g = pd.PeriodicGrid(cell, nr, na, D=10.0, scheme=sc, deriv=1)
        dn, ds, ex = pd.grid_anchors(g, S, D, N)
        el, eh = (rks(g, xc) - eref[xc] for xc in ("LDA,VWN", "PBE0"))
        print(
            f"{sc:6s} {nr:3d}x{na:<4d} {g.size:7d} {g.weights.sum() - cell.vol:+.2e}  {dn:.2e} {ds:.2e} {ex - ref:+.2e} {el:+.2e} {eh:+.2e} ({time.time() - t:.0f}s)",
            flush=True,
        )
