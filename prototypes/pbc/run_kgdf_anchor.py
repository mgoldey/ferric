"""Exactness anchors for pbc_kgdf (k-point RS-GDF per q).  Predictions written BEFORE the runs:

(a) trivial-aux limit.  Anchor cell (one s primitive al=0.5 per H) and the Iteration-2 aux (24 s functions of
    exponent 2al at the 8 half-lattice classes per pair type).  Claim: the SAME 24 centres span every q-Bloch pair
    density, because sum_T e^{iq.T} sum_L e^{ik'.L} c_L g(r - C_L - T) regroups (L = 2M + h) into
    sum_h [sum_M e^{i(k'.(2M+h) - q.M)} c_{2M+h}] X^q_h(r): a k'-dependent combination of the q-Bloch sums of the
    8 class centres.  PHYSICS: Jker/Kker from B match the dense pbc_kpts kernels to ~1e-11 at every (k,k').
    ARTIFACT signatures: wrong q phase on aux images -> O(1e-2) at q != 0 only, q = 0 untouched; G=0 bookkeeping at
    q != 0 -> error at q != 0 only; missing per-q lindep is INVISIBLE here unless the metric is singular, so the
    anchor is repeated with one aux function DUPLICATED (exactly singular J2(q) at every q): with the per-q cut it
    must stay exact, without it (q != 0) it must blow up.
(b) 1x1x1 mesh == pbc_gdf.build_gdf (H2/STO-3G, cc-pvdz-ri): B B^H == B B^T to ~1e-12 (same code path at q = 0).
(c) k-mesh RS-GDF == Gamma RS-GDF of the explicit supercell with the aux on every image.  Should be EXACT (not
    just converged-to-each-other): the supercell aux {chi_P(r - t_r) Bloch-summed over the supercell lattice} is,
    by a unitary DFT over the residues r, the direct sum over mesh q of the per-q aux spaces; the supercell metric
    is block-diagonal in q with blocks J2(q) (identical eigenvalues, so the same lindep cut keeps the same space),
    pair densities of momentum q only overlap the q block, and the supercell G = 0 term is the q = 0 block's.
    PHYSICS: E/cell equal to ~1e-11 (the SR/LR sums are each converged to prec 1e-13, not identical term sets).
    ARTIFACT: a q-phase error would give O(1e-3+) here too (n = 3 needed: at n = 2, e^{iq.T} is real).

Usage: python3 run_kgdf_anchor.py [a|b|c|tr] ...
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, madelung, rhf  # noqa: E402
from pbc_gdf import build_gdf, ferric_basis, jk_from_B  # noqa: E402
from pyscf import gto  # noqa: E402

H2_A = np.eye(3) * 4.0
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
AL = 0.5


def anchor_system(dup=False):
    cell = Cell(H2_A, H2_ATOMS, {"H": [[0, [AL, 1.0]]]})
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i, j in [(0, 0), (1, 1), (0, 1)]
        for h in np.ndindex(2, 2, 2)
    ]
    if dup:
        cen = cen + [cen[5]]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * AL, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


def kernel_err(kb, kg):
    Jf, Kf = KG.kernels_from_kB(kg)
    dK = abs(Kf - kb["Kker"]).max(axis=(2, 3, 4, 5))
    dJ = abs(Jf - kb["Jker"]).max()
    return dJ, dK


def e_k(kb, kg, ex, nelec=2):
    return PK.krhf(
        kb,
        nelec,
        conv=1e-12,
        kshift=kb["madelung"] if ex == "ewald" else 0.0,
        jk=None if kg is None else KG.jk_from_kB(kg),
    )[0]


def anchor_a(n):
    cell, aux = anchor_system()
    kb = PK.build_k(cell, n, exxdiv="ewald", verbose=False)
    q0 = [j for j in range(kb["Nk"])]
    for mut in (None, "q_phase_sign", "g0_all_q"):
        KG._MUTANT = mut
        kg = KG.build_kgdf(cell, n, auxmol=aux, spherical=False)
        KG._MUTANT = None
        dJ, dK = kernel_err(kb, kg)
        diag = max(dK[k, k] for k in q0)
        off = max(dK[k, j] for k in q0 for j in q0 if k != j) if kb["Nk"] > 1 else 0.0
        de = {ex: e_k(kb, kg, ex) - e_k(kb, None, ex) for ex in ("none", "ewald")}
        print(
            f"(a) {n} {mut or 'anchor'}: max|dJker| {dJ:.1e} max|dKker| q=0 {diag:.1e} q!=0 {off:.1e} "
            f"dE none {de['none']:.1e} ewald {de['ewald']:.1e}; kept/q "
            f"{[v['kept'] for v in kg['info']['per_q'].values()]} smin {min(v['smin'] for v in kg['info']['per_q'].values()):.1e}",
            flush=True,
        )
    cell, aux = anchor_system(dup=True)
    for mut in (None, "no_lindep_q"):
        KG._MUTANT = mut
        kg = KG.build_kgdf(cell, n, auxmol=aux, spherical=False)
        KG._MUTANT = None
        dJ, dK = kernel_err(kb, kg)
        try:
            de = e_k(kb, kg, "none") - e_k(kb, None, "none")
        except Exception as exc:  # noqa: BLE001
            de = f"SCF failed: {type(exc).__name__}"
        print(
            f"(a-dup) {n} {mut or 'per-q lindep'}: max|dJker| {dJ:.1e} max|dKker| {dK.max():.1e} dE {de}; "
            f"kept/q {[v['kept'] for v in kg['info']['per_q'].values()]} "
            f"smin/q {['%.0e' % v['smin'] for v in kg['info']['per_q'].values()]}",
            flush=True,
        )


def anchor_b():
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    g = build_gdf(cell, aux)
    kg = KG.build_kgdf(cell, (1, 1, 1), aux)
    Bk = kg["B"][(0, 0)]
    Ig = np.einsum("Pmn,Pls->mnls", g["B"], g["B"])
    Ik = np.einsum("Pmn,Pls->mnls", Bk, Bk.conj())
    print(
        f"(b) 1x1x1 vs pbc_gdf: kept {Bk.shape[0]} vs {g['B'].shape[0]}, max|dI| {abs(Ik - Ig).max():.1e}, "
        f"max|Im B| {abs(Bk.imag).max():.1e}"
    )
    kb = PK.build_k(cell, (1, 1, 1), verbose=False)
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        eg = rhf(
            kb["S"][0].real,
            kb["h"][0].real,
            None,
            kb["enn"],
            2,
            conv=1e-12,
            kshift=vm,
            jk=jk_from_B(g["B"]),
        )[0]
        ek = e_k(kb, kg, ex)
        print(f"(b) {ex}: E_kgdf {ek:.12f} E_gdf {eg:.12f} dE {ek - eg:.1e}")


def anchor_c(n=(1, 1, 3), auxname="cc-pvdz-ri", prec_g=1e-4):
    """Same loose-gcut 1e build on both sides (exact at any gcut, Iteration 9), RS-GDF JK on both sides."""
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    aux = ferric_basis(auxname, ["H"])
    gcut, th = PK.aft_gcut(cell, prec_g), 1e-8
    t = time.time()
    kb = PK.build_k(cell, n, gcut=gcut, thresh=th, verbose=False)
    sc = PK.supercell_cell(cell, n)
    g = PK.gamma_aft(sc, gcut=gcut, thresh=th)
    Nk = int(np.prod(n))
    print(f"  1e builds {time.time() - t:.0f}s", flush=True)
    t = time.time()
    kg = KG.build_kgdf(cell, n, aux, verbose=True)
    t1 = time.time()
    gs = build_gdf(sc, aux, verbose=True)
    print(
        f"  kgdf {t1 - t:.0f}s, supercell gdf {time.time() - t1:.0f}s; kept per q "
        f"{[v['kept'] for v in kg['info']['per_q'].values()]} (sum {sum(v['kept'] for v in kg['info']['per_q'].values())}) "
        f"vs supercell {gs['B'].shape[0]}",
        flush=True,
    )
    vm = madelung(sc)
    for ex in ("none", "ewald"):
        ks = vm if ex == "ewald" else 0.0
        es = (
            rhf(
                g["S"],
                g["h"],
                None,
                g["enn"],
                2 * Nk,
                conv=1e-12,
                kshift=ks,
                jk=jk_from_B(gs["B"]),
            )[0]
            / Nk
        )
        ek = e_k(kb, kg, ex)
        ed = e_k(kb, None, ex)
        print(
            f"(c) {n} {auxname} {ex}: E_kgdf {ek:.12f} E_sc_gdf/N {es:.12f} diff {ek - es:.1e}; "
            f"(E_kgdf - dense k-AFT at the loose anchor gcut {ek - ed:.2e}; at the converged gcut the fit error is -7.43e-5, run_kgdf_fiterr_split.py)",
            flush=True,
        )
    for mut in ("q_phase_sign",):
        KG._MUTANT = mut
        kgm = KG.build_kgdf(cell, n, aux)
        KG._MUTANT = None
        print(
            f"(c) MUTANT {mut}: E_kgdf - E_sc {e_k(kb, kgm, 'none') - es_none(g, gs, Nk):.2e}",
            flush=True,
        )


def es_none(g, gs, Nk):
    return (
        rhf(g["S"], g["h"], None, g["enn"], 2 * Nk, conv=1e-12, jk=jk_from_B(gs["B"]))[
            0
        ]
        / Nk
    )


def time_reversal():
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    kg = KG.build_kgdf(cell, (1, 2, 3), aux)
    KG._MUTANT = "no_time_reversal"
    kg2 = KG.build_kgdf(cell, (1, 2, 3), aux)
    KG._MUTANT = None
    Jf, Kf = KG.kernels_from_kB(kg)
    Jf2, Kf2 = KG.kernels_from_kB(kg2)
    print(
        f"(tr) 1x2x3: time-reversal fill vs brute force: max|dKker| {abs(Kf - Kf2).max():.1e} "
        f"max|dJker| {abs(Jf - Jf2).max():.1e}; q built {kg['info']['n_q_built']} of {kg['Nk']}"
    )


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["a", "b", "c", "tr"]:
        t = time.time()
        if arg == "a":
            anchor_a((1, 1, 3))
            anchor_a((2, 2, 2))
        elif arg == "a3":
            anchor_a((1, 1, 3))
        elif arg == "b":
            anchor_b()
        elif arg == "c":
            anchor_c()
        elif arg == "tr":
            time_reversal()
        print(f"[{arg}] {time.time() - t:.0f}s", flush=True)
