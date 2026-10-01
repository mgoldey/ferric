"""Stage-2 Rust pins (crates/ferric-pbc/tests/pbc_rks.rs): the A2 grid with scheme ssf/becke, D = 10, at
ferric Lebedev orders (no 146/590), anchors + RKS energies; and the uniform oracle. Usage: python3 run_dft_ssf_pins.py {h2|tri}"""

import sys
import time

sys.path.insert(0, ".")
import numpy as np
from pbc_gamma import Cell, build_integrals
import pbc_dft as pd
from test_prototype import H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, SP_BASIS

np.set_printoptions(precision=15)
which = sys.argv[1]
if which in ("h2", "tri"):
    a, atoms, basis = (
        (H2_A, H2_ATOMS, "sto-3g") if which == "h2" else (TRI_A, TRI_ATOMS, SP_BASIS)
    )
    cell = Cell(a, atoms, basis)
    N = cell.mol.nelectron
    I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    S = I["S"]
    Dp = pd.probe_density(cell, S, kind="flat")
    for g in [(30, 50), (50, 110), (75, 110), (75, 302)]:
        for scheme in ("ssf", "becke"):
            t = time.time()
            gr = pd.PeriodicGrid(cell, *g, D=10.0, scheme=scheme)
            dn, ds, ex = pd.grid_anchors(gr, S, Dp, N)
            print(
                f"{which} {scheme} {g} npts {gr.size} dN {dn:.3e} dS {ds:.3e} ({time.time() - t:.0f}s)",
                flush=True,
            )
            if g == (75, 302) or (g == (75, 110) and which == "h2"):
                for xc in ("LDA,VWN", "PBE", "PBE0"):
                    e = pd.rks(
                        S,
                        I["h"],
                        pd.dense_jk(I["I"]),
                        I["enn"],
                        N,
                        gr,
                        xc,
                        kshift=I["madelung"],
                        conv=1e-12,
                    )[0]
                    print(f"   E {which} {scheme} {g} {xc} {e:.12f}", flush=True)
    n = 40 if which == "h2" else 48
    gu = pd.uniform_grid(cell, n)
    dn, ds, ex = pd.grid_anchors(gu, S, Dp, N)
    print(f"{which} uniform {n} dN {dn:.3e} dS {ds:.3e}")
    for xc in ("LDA,VWN", "PBE", "PBE0"):
        e = pd.rks(
            S,
            I["h"],
            pd.dense_jk(I["I"]),
            I["enn"],
            N,
            gu,
            xc,
            kshift=I["madelung"],
            conv=1e-12,
        )[0]
        print(f"   E {which} uniform{n} {xc} {e:.12f}", flush=True)
