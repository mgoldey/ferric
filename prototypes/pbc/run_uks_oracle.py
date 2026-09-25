"""Iteration 10 PySCF oracle: our Gamma UKS/ROKS (pure-AFT dense I + our numint on PySCF's own pbc BeckeGrids
points, A1 50x146) vs PySCF 2.13 pbc.dft.UKS / ROKS (AFTDF mesh 61^3, exxdiv='ewald', BeckeGrids (50,146),
prune None, treutler, small_rho_cutoff 0, conv_tol 1e-12).  PySCF runs from its default guess AND from our
converged densities (different basin vs Fock disagreement).  Ours: staged start (none -> ewald).
Usage: python3 run_uks_oracle.py [H|H2|tri ...]"""

import sys
import time

import numpy as np
from pyscf.dft import radi
from pyscf.pbc import df as pdf
from pyscf.pbc import dft as pdft
from pyscf.pbc import gto as pgto

sys.path.insert(0, ".")
import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from test_prototype import H2_A, H2_ATOMS, SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

SYS = {
    "H": (H2_A, [("H", (0.3, 0.2, 0.1))], "sto-3g", 1, 0),
    "H2": (H2_A, H2_ATOMS, "sto-3g", 2, 0),
    "tri": (TRI_A, TRI_ATOMS, SP_BASIS, 3, 1),
}
XCS = ("LDA,VWN", "PBE", "PBE0")
for name in sys.argv[1:] or list(SYS):
    a, atoms, basis, na, nb = SYS[name]
    t = time.time()
    cell = Cell(a, atoms, basis)
    I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    vm, S, h, enn = I["madelung"], I["S"], I["h"], I["enn"]
    jk = pd.dense_jk(I["I"])
    g = pd.pyscf_a1_grid(cell, 50, 146)
    pc = pgto.Cell(
        a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=na - nb
    )
    pc.precision = 1e-12
    pc.build()
    print(
        f"{name} (na {na}, nb {nb}): setup {time.time() - t:.0f}s  v_M {vm:.10f}",
        flush=True,
    )
    for xc in XCS:
        u = U.uks(S, h, jk, enn, na, nb, g, xc, kshift=vm, conv=1e-12, staged=True)
        r = (
            U.roks(
                S,
                h,
                jk,
                enn,
                na,
                nb,
                g,
                xc,
                kshift=vm,
                conv=1e-12,
                guess=(u["Da"], u["Db"]),
            )
            if nb > 0
            else None
        )
        for kind, ours in (("UKS", u), ("ROKS", r)):
            if ours is None:
                continue
            for gname in ("default", "ours"):
                mf = (pdft.UKS if kind == "UKS" else pdft.ROKS)(pc, xc=xc)
                mf.with_df = pdf.AFTDF(pc)
                mf.with_df.mesh = [61] * 3
                mf.exxdiv = "ewald"
                mf.grids = pdft.gen_grid.BeckeGrids(pc)
                mf.grids.atom_grid = (50, 146)
                mf.grids.prune = None
                mf.grids.radi_method = radi.treutler
                mf.small_rho_cutoff = 0.0
                mf.conv_tol = 1e-12
                mf.max_cycle = 200
                e = mf.kernel(
                    dm0=None
                    if gname == "default"
                    else np.array([ours["Da"], ours["Db"]])
                )
                s2 = mf.spin_square()[0]
                print(
                    f"   {xc:8s} {kind:4s} ours E {ours['e']:.12f} <S2> {ours['s2']:.10f} | PySCF[{gname:7s}] "
                    f"dE {e - ours['e']:+.1e} d<S2> {s2 - ours['s2']:+.1e} conv {mf.converged}",
                    flush=True,
                )
        print(
            f"   {xc:8s} UKS gaps {np.round(u['gaps'], 5)}  hyb v_M {u['hyb'] * vm:.5f}  trap {u['trap']}",
            flush=True,
        )
