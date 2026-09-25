"""Iteration 10 exactness anchors (Gamma UKS), independent of PySCF pbc.  Usage: python3 run_uks_anchor.py {h2|tri}

PREDICTIONS (written before running):
 (a) closed shell (na == nb, no mix) UKS == pbc_dft.rks on the same grid, LDA/PBE/PBE0 x exxdiv none/ewald:
     |dE| ~1e-13 (same stationary point, polarized kernel at rho_a = rho_b == unpolarized kernel).
     artifact hypotheses: hyb/2 per spin (RKS bookkeeping) -> PBE0 off O(1e-1); Madelung on full K -> off by
     exactly (1-hyb) v_M N/2 under ewald only (LDA: v_M N/2, PBE0: 0.75 v_M N/2); unpolarized XC -> INVISIBLE
     here (rho_a = rho_b), which is why (c) and the PySCF pins exist.
 (b) xc='HF' UKS == pbc_uhf.uhf (independent SCF loop) on open shells, both exxdiv: |dE| <= 1e-11, same <S2>.
 (c) open-shell E_xc/V_xc consistency: tr(V_s dD) == FD of E_xc along a random symmetric dD (per spin),
     relative 1e-8; the unpolarized mutant breaks tr(V_a dD) - tr(V_b dD) != FD difference.
 (d) ewald - none == -hyb v_M (N_a+N_b)/2 at the same stationary density (identity, open shell), 1e-11."""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_uhf import uhf  # noqa: E402
from test_prototype import H2_A, H2_ATOMS, SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "h2"
a, atoms, basis, (na, nb) = (
    (H2_A, H2_ATOMS, "sto-3g", (2, 0))
    if which == "h2"
    else (TRI_A, TRI_ATOMS, SP_BASIS, (3, 1))
)
cell = Cell(a, atoms, basis)
N = cell.mol.nelectron
t = time.time()
I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
vm, S, h, enn = I["madelung"], I["S"], I["h"], I["enn"]
jk = pd.dense_jk(I["I"])
g = pd.pyscf_a1_grid(cell, 50, 146)
print(
    f"{which}: ints+grid {time.time() - t:.0f}s, v_M {vm:.10f}, npts {g.size}",
    flush=True,
)
XCS = ("LDA,VWN", "PBE", "PBE0")

print(
    "(a) closed shell UKS - RKS   [xc exxdiv: correct | hyb_half | madelung_full_k | unpolarized]"
)
for xc in XCS:
    for ex in ("none", "ewald"):
        ks = vm if ex == "ewald" else 0.0
        er = pd.rks(S, h, jk, enn, N, g, xc, kshift=ks, conv=1e-12)[0]
        row = []
        for m in (None, "hyb_half_per_spin", "madelung_full_k", "unpolarized"):
            U._MUTANT = m
            try:
                row.append(
                    U.uks(S, h, jk, enn, N // 2, N // 2, g, xc, kshift=ks, conv=1e-12)[
                        "e"
                    ]
                    - er
                )
            finally:
                U._MUTANT = None
        print(
            f"   {xc:8s} {ex:5s} "
            + " | ".join(f"{x:+.2e}" for x in row)
            + f"   (predicted full-K shift {-(1 - U.hybrid_fraction(xc)) * vm * N / 2 if ex == 'ewald' else 0.0:+.4e})",
            flush=True,
        )

print(f"(b) open shell (na {na}, nb {nb}) UKS[xc=HF] - pbc_uhf.uhf")
for ex in ("none", "ewald"):
    ks = vm if ex == "ewald" else 0.0
    ref = uhf(S, h, I["I"], enn, na, nb, conv=1e-12, kshift=0.0)
    ref = uhf(
        S, h, I["I"], enn, na, nb, conv=1e-12, kshift=ks, guess=(ref["Da"], ref["Db"])
    )
    u = U.uks(S, h, jk, enn, na, nb, None, "HF", kshift=ks, conv=1e-12, staged=True)
    print(
        f"   {ex:5s} dE {u['e'] - ref['e']:+.2e}  d<S2> {u['s2'] - ref['s2']:+.1e}  E {u['e']:.12f}",
        flush=True,
    )

print(
    "(c) open-shell E_xc / V_xc FD consistency (rel. err per spin; unpolarized mutant)"
)
rng = np.random.default_rng(1)
u0 = U.uks(S, h, jk, enn, na, nb, g, "PBE", conv=1e-10)
dD = rng.standard_normal(S.shape)
dD = 0.5 * (dD + dD.T) * 1e-3
for m in (None, "unpolarized"):
    U._MUTANT = m
    try:
        _, Va, Vb = U.eval_vxc_uks(g, u0["Da"], u0["Db"], "PBE")
        eps = 1e-4
        fa = (
            U.eval_vxc_uks(g, u0["Da"] + eps * dD, u0["Db"], "PBE")[0]
            - U.eval_vxc_uks(g, u0["Da"] - eps * dD, u0["Db"], "PBE")[0]
        ) / (2 * eps)
        fb = (
            U.eval_vxc_uks(g, u0["Da"], u0["Db"] + eps * dD, "PBE")[0]
            - U.eval_vxc_uks(g, u0["Da"], u0["Db"] - eps * dD, "PBE")[0]
        ) / (2 * eps)
    finally:
        U._MUTANT = None
    print(
        f"   {str(m):12s} alpha {abs(np.sum(Va * dD) - fa) / abs(fa):.1e}  beta {abs(np.sum(Vb * dD) - fb) / abs(fb):.1e}"
        f"   (Va-Vb).dD {np.sum((Va - Vb) * dD):+.3e} vs FD {fa - fb:+.3e}"
    )

print("(d) open shell ewald - none vs -hyb v_M N/2")
for xc in XCS:
    un = U.uks(S, h, jk, enn, na, nb, g, xc, conv=1e-12)
    ue = U.uks(
        S, h, jk, enn, na, nb, g, xc, kshift=vm, conv=1e-12, guess=(un["Da"], un["Db"])
    )
    print(
        f"   {xc:8s} {ue['e'] - un['e'] - (-U.hybrid_fraction(xc) * vm * N / 2):+.1e}  |dD| {abs(ue['Da'] - un['Da']).max():.1e}"
        f"  gaps(none) {np.round(un['gaps'], 4)}  hyb v_M {U.hybrid_fraction(xc) * vm:.4f}",
        flush=True,
    )
