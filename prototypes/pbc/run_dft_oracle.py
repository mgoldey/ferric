"""Iteration 8 PySCF oracle: Gamma RKS (LDA, PBE, PBE0 exxdiv=ewald), all-electron, exact J.
Ours: pure-AFT dense I (pbc_gamma) + periodic Becke grid A2 (pbc_dft).  PySCF: pbc.dft.RKS on AFTDF
(mesh 61^3) with its all-electron BeckeGrids (A1: domain-cut, no size adjust, prune None, treutler).
Also: OUR SCF on PySCF's A1 points (isolates grid construction from numint/SCF), and PySCF with a
uniform grid (B).  Usage: python3 run_dft_oracle.py {h2|tri}"""

import sys
import time

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals
import pbc_dft as pd
from test_prototype import H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, SP_BASIS
from pyscf.dft import radi
from pyscf.pbc import gto as pgto, dft as pdft, df as pdf

which = sys.argv[1] if len(sys.argv) > 1 else "h2"
a, atoms, basis = (
    (H2_A, H2_ATOMS, "sto-3g") if which == "h2" else (TRI_A, TRI_ATOMS, SP_BASIS)
)
grids = [(30, 86), (50, 146), (75, 110), (75, 302), (100, 302), (150, 590)]
if len(sys.argv) > 2:
    grids = [tuple(map(int, g.split("x"))) for g in sys.argv[2].split(",")]
cell = Cell(a, atoms, basis)
N = cell.mol.nelectron
t = time.time()
I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
print(
    f"{which}: pure-AFT I built in {time.time() - t:.0f}s, v_M {I['madelung']:.10f}",
    flush=True,
)
jk = pd.dense_jk(I["I"])

pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.build()


def pyscf_rks(xc, grid):
    mf = pdft.RKS(pc, xc=xc)
    mf.with_df = pdf.AFTDF(pc)
    mf.with_df.mesh = [61] * 3
    mf.exxdiv = "ewald"
    if grid[0] == "uniform":
        mf.grids = pdft.gen_grid.UniformGrids(pc)
        mf.grids.mesh = [grid[1]] * 3
    else:
        mf.grids = pdft.gen_grid.BeckeGrids(pc)
        mf.grids.atom_grid = grid
        mf.grids.prune = None
        mf.grids.radi_method = radi.treutler
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    e = mf.kernel()
    assert mf.converged
    return e, mf.mo_energy, mf.grids.weights.size


XCS = ("LDA,VWN", "PBE", "PBE0")
print(
    "columns: xc | grid | ours A2 E | PySCF A1 E | ours-on-A1 - PySCF (d eps) | ours A2 - PySCF | npts A2/A1",
    flush=True,
)
for g in grids:  # grids built once per size, reused for every functional
    t = time.time()
    gA2 = pd.PeriodicGrid(cell, *g, D=10.0)
    gA1 = pd.pyscf_a1_grid(cell, *g)
    print(f"  [grids {g}: {time.time() - t:.0f}s]", flush=True)
    for xc in XCS:
        rA2 = pd.rks(
            I["S"],
            I["h"],
            jk,
            I["enn"],
            N,
            gA2,
            xc,
            kshift=I["madelung"],
            conv=1e-12,
            return_all=True,
        )
        eP, epsP, nP = pyscf_rks(xc, g)
        rA1 = pd.rks(
            I["S"],
            I["h"],
            jk,
            I["enn"],
            N,
            gA1,
            xc,
            kshift=I["madelung"],
            conv=1e-12,
            return_all=True,
        )
        print(
            f"  {xc:8s} {g[0]:3d}x{g[1]:<3d} {rA2['e']:.12f} {eP:.12f} {rA1['e'] - eP:+.1e} ({abs(rA1['eps'] - epsP).max():.0e}) "
            f"{rA2['e'] - eP:+.2e} {gA2.size}/{nP}",
            flush=True,
        )
for xc in XCS:
    for n in (40, 60) if which == "h2" else (48,):
        eU = pyscf_rks(xc, ("uniform", n))[0]
        print(f"  {xc:8s} PySCF uniform mesh {n}^3: {eU:.12f}", flush=True)
