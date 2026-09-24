"""Iteration 10 (3): does the Iteration-6 Ewald trap occur for hybrids?  tri 4H s+p triplet (na 3, nb 1),
A1 50x146 grid, pure-AFT I.  Functionals alpha*HF + (1-alpha)*PBE exchange, PBE correlation, plus pure 'HF'.

PREDICTIONS (before running): with fraction alpha the Madelung term is alpha v_M x (occupied projector) per spin,
so ewald and none still have IDENTICAL stationary densities (E offset -alpha v_M N/2), and a state that is
non-aufbau under none by delta (occupied level above a virtual by delta) becomes aufbau-self-consistent under
ewald iff delta < alpha v_M.  The necessary ewald-minimum criterion becomes per-spin gap >= alpha v_M.
  alpha = 0: no trap possible (v_M never enters).  alpha -> 1: reproduces the UHF trap (-1.812958714837 vs
  -1.827999723359 for 'HF').  Whether an intermediate alpha traps depends on whether a hole state exists as a
  stationary point of THAT functional with delta < alpha v_M -- not predictable a priori; measured two ways:
  (i) ewald from the core guess (what a naive SCF does); (ii) ewald from the UHF TRAP density (targets the hole
  state directly).  Each converged ewald state is re-evaluated under none: |[F_none, D]| (must be ~0: artifact
  check that the Madelung term is alpha x projector) and the occupation-aware none gap (negative = hole state).
Usage: [CORR=PBE|""] python3 run_uks_trap.py [alpha ...]   (CORR="" -> exchange-only functionals)"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from test_prototype import SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

t = time.time()
cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
vm, S, h, enn = I["madelung"], I["S"], I["h"], I["enn"]
jk = pd.dense_jk(I["I"])
g = pd.pyscf_a1_grid(cell, 50, 146)
na, nb, N = 3, 1, 4
print(f"setup {time.time() - t:.0f}s  v_M {vm:.10f}", flush=True)


def solve(xc, **kw):
    try:
        return U.uks(S, h, jk, enn, na, nb, g, xc, conv=1e-11, **kw), "diis"
    except (
        RuntimeError
    ):  # DIIS stagnation near an aufbau degeneracy -> level-shifted Roothaan, no DIIS
        return U.uks(
            S,
            h,
            jk,
            enn,
            na,
            nb,
            g,
            xc,
            conv=1e-11,
            maxiter=4000,
            level_shift=0.5,
            diis_start=10**6,
            **kw,
        ), "ls0.5"


def none_view(xc, r):
    hyb = U.hybrid_fraction(xc)
    e_n, Fa, Fb, _ = U._fock_energy(
        S, h, jk, enn, r["Da"], r["Db"], None if xc == "HF" else g, xc, hyb, 0.0
    )
    comm = max(
        abs(Fa @ r["Da"] @ S - S @ r["Da"] @ Fa).max(),
        abs(Fb @ r["Db"] @ S - S @ r["Db"] @ Fb).max(),
    )
    return (
        comm,
        e_n - r["e"] - hyb * vm * N / 2,
        (U.occ_gap(Fa, r["Da"], S), U.occ_gap(Fb, r["Db"], S)),
    )


trap_hf, _ = solve("HF", kshift=vm)
print(
    f"UHF trap state (ewald, core guess): E {trap_hf['e']:.12f}  <S2> {trap_hf['s2']:.8f}"
)
alphas = [float(x) for x in sys.argv[1:]] or [
    0.0,
    0.1,
    0.25,
    0.4,
    0.5,
    0.6,
    0.75,
    0.9,
    1.0,
]
print(
    "xc | start | E ewald | solver | E - E(staged) | ewald occ-gaps | alpha v_M | trap flag | under none: |[F,D]|, "
    "E_none-E_ewald-alpha vM N/2, occ-gaps(none) | <S2>"
)
for al in alphas:
    xc = f"{al}*HF + {1 - al}*PBE, {os.environ.get('CORR', 'PBE')}"
    hyb = U.hybrid_fraction(xc)
    st, sst = solve(xc, kshift=vm, staged=True)
    for start, kw in (
        ("staged", None),
        ("core", {}),
        ("UHF-trap D", {"guess": (trap_hf["Da"], trap_hf["Db"])}),
    ):
        r, sv = (st, sst) if kw is None else solve(xc, kshift=vm, **kw)
        comm, dn, gn = none_view(xc, r)
        print(
            f"   a={al:<4} {start:10s} {r['e']:.10f} {sv:5s} {r['e'] - st['e']:+.2e} {np.round(r['gaps'], 4)} {hyb * vm:.4f} "
            f"{r['trap']!s:5s} | {comm:.0e} {dn:+.0e} {np.round(gn, 4)} | {r['s2']:.5f}",
            flush=True,
        )
