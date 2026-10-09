"""Iteration 8 anchor (a)+(b): periodic Becke grid (A2) integrates the lattice-summed density to N,
reproduces the lattice overlap S_latt, and converges the exchange-only LDA energy to a uniform-grid
reference (B, spectrally convergent for smooth all-H densities).  Also PySCF's domain-cut A1 grid.
Usage: python3 run_dft_anchor.py {h2|tri|lih}"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell
import pbc_dft as pd
from test_prototype import H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, SP_BASIS

which = sys.argv[1] if len(sys.argv) > 1 else "h2"
if which == "h2":
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    meshes = (24, 32, 40, 48, 64)
elif which == "tri":
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    meshes = (32, 48, 64, 80)
else:
    cell = Cell(
        np.eye(3) * 4.0, [("Li", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 3.1))], "sto-3g"
    )
    meshes = (32, 48, 64, 96, 128)
S = pd.lattice_overlap(cell)
N = cell.mol.nelectron
D = pd.probe_density(cell, S)
print(
    f"{which}: nao {cell.mol.nao} N {N} tr(DS) {np.sum(D * S):.15f} vol {cell.vol:.3f}",
    flush=True,
)

print("B uniform:  n  npts  |dN|  max|dS|  Ex", flush=True)
ref = None
for n in meshes:
    t = time.time()
    g = pd.uniform_grid(cell, n, deriv=0)
    dn, ds, ex = pd.grid_anchors(g, S, D, N)
    print(
        f"  {n:4d} {g.size:9d} {dn:.2e} {ds:.2e} {ex:.12f} ({time.time() - t:.0f}s)",
        flush=True,
    )
    ref = ex
del g
print(f"reference Ex (finest uniform) = {ref:.12f}", flush=True)

print(
    "A2 periodic Becke (D=10):  nrad nang  npts  |dN|  max|dS|  Ex-ref  <nb/pt> <img/chunk>",
    flush=True,
)
for nr, na in (
    (20, 50),
    (30, 86),
    (40, 110),
    (50, 146),
    (75, 110),
    (75, 194),
    (100, 302),
    (150, 590),
):
    t = time.time()
    g = pd.PeriodicGrid(cell, nr, na, D=10.0, deriv=0)
    dn, ds, ex = pd.grid_anchors(g, S, D, N)
    print(
        f"  {nr:4d} {na:4d} {g.size:7d} {dn:.2e} {ds:.2e} {ex - ref:+.2e}  {g.n_nb.mean():.0f} {g.n_img.mean():.0f} ({time.time() - t:.0f}s)",
        flush=True,
    )

print(
    "A2 neighbour cutoff D at (75,302):  scheme D  npts  |dN|  max|dS|  Ex-ref <nb/pt>",
    flush=True,
)
for scheme in ("becke", "ssf"):
    for Dc in (4.0, 6.0, 8.0, 10.0, 14.0):
        g = pd.PeriodicGrid(cell, 75, 302, D=Dc, deriv=0, scheme=scheme)
        dn, ds, ex = pd.grid_anchors(g, S, D, N)
        print(
            f"  {scheme:5s} {Dc:4.0f} {g.size:7d} {dn:.2e} {ds:.2e} {ex - ref:+.2e} {g.n_nb.mean():.0f}",
            flush=True,
        )

print(
    "A1 PySCF domain-cut BeckeGrids:  nrad nang  npts  |dN|  max|dS|  Ex-ref",
    flush=True,
)
for nr, na in ((30, 86), (50, 146), (75, 110), (75, 302), (100, 302), (150, 590)):
    t = time.time()
    g = pd.pyscf_a1_grid(cell, nr, na, deriv=0)
    dn, ds, ex = pd.grid_anchors(g, S, D, N)
    print(
        f"  {nr:4d} {na:4d} {g.size:7d} {dn:.2e} {ds:.2e} {ex - ref:+.2e} ({time.time() - t:.0f}s)",
        flush=True,
    )
