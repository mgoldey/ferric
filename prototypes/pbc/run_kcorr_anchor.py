"""Stage 9 exactness anchors for pbc_kcorr (k-point MP2 / dRPA).  Predictions written BEFORE the run:

(a) 1x1x1 mesh == pbc_mp2.gamma_mp2 / pbc_rpa.gamma_drpa on pbc_gamma's dense ERI (default gcut): identical terms
    (same K set, real vs complex-phased orbitals of the same Fock) => <= 1e-12, both conventions.
(b) N-mesh per cell == Gamma MP2/dRPA of the explicit supercell / N, SAME gcut on both sides.  Prediction: EXACT
    (1e-12), not merely convergent, because (i) {G + q} over the mesh is the supercell reciprocal lattice (Iteration 9),
    so the ov ERIs are the same numbers in a unitarily rotated basis; (ii) the supercell Fock is block-diagonal in the
    Bloch basis, so its canonical orbitals are Bloch orbitals up to rotations INSIDE degenerate eigenspaces (k and -k by
    time reversal; the supercell code returns real cos/sin mixtures), and canonical MP2/dRPA are invariant under
    rotations that commute with the Fock matrix (denominators are constant on a degenerate block); (iii) the supercell
    v_M IS the mesh v_M (Iteration 9), so the shifted denominators coincide too.  Would fail if: occupation per k
    differed from the supercell aufbau (not here: gap), or near-degeneracies were split differently (they are not
    split at all: same Fock).  dRPA by GL quadrature: also exact at FIXED n (the supercell Pi is block-diagonal in q,
    so ln det sums over q node by node), not just in the converged limit.
(c) trivial-aux RS-GDF B (Iteration 2/11 anchor aux) == dense AFT: <= 1e-11 (the B B^H kernels agreed to 1.8e-12).
(d) O(Pi^2) term of k-dRPA == direct KMP2 (2 sum |V|^2/D): exact identity per q after the w integral (<= 1e-12).
Mutations (pbc_kcorr._MUTANT) at 1x1x3 (complex q; at n = 2 every phase is real):
    kb_wrong: breaks (b) MP2 (wrong exchange partner / denominator), dRPA untouched;
    no_conj:  breaks (b) MP2 (q != 0 legs), dRPA untouched (Kq uses its own conj);
    no_madelung: shifted row becomes the unshifted value (breaks (b) shifted only, none untouched);
    rpa_k_wrong: breaks (b) dRPA only.
Artifact hypothesis: a convention error that is the same on both sides of (b) (e.g. a wrong 1/Nk power) would NOT be
caught by (b) alone -- (a) at Nk = 1 cannot see an Nk power either, so the PySCF KMP2 pin (run_kcorr_oracle.py) and
(d) are the independent checks of the normalisation.
Usage: python3 run_kcorr_anchor.py [a|b|btri|c|mut]"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kcorr as KC  # noqa: E402
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, build_integrals, madelung, rhf  # noqa: E402
from pbc_mp2 import denominators, gamma_mp2  # noqa: E402
from pbc_rpa import gamma_drpa  # noqa: E402
from pyscf import gto  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS, SP, TRI_A, TRI_ATOMS  # noqa: E402

CONVS = ("shifted", "unshifted")


def kcorr(cell, n, nelec, gcut=None, thresh=1e-14, kg=None, head=False):
    """k-RHF (exxdiv none; C is the same under ewald) + both routes' energies for both conventions."""
    kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, thresh=thresh, verbose=False)
    nocc = nelec // 2
    e, eps, it, C, occ = PK.krhf(kb, nelec, conv=1e-12, kshift=0.0, return_mo=True)
    assert all(int(o.sum()) == nocc and o[:nocc].all() for o in occ), (
        "occupation per k not uniform"
    )
    st = (
        KC.kB_ov(kg, C, nocc)
        if kg is not None
        else KC.aft_ov(cell, n, C, nocc, gcut=gcut, thresh=thresh, head=head)
    )
    out = {}
    for cv in CONVS:
        eo, ev = KC.k_denominators(eps, nocc, kb["madelung"], cv)
        out[cv] = dict(
            mp2=KC.kmp2(st, eo, ev)[0],
            plasmon=KC.kdrpa_plasmon(st, eo, ev),
            quad=KC.kdrpa_quad(st, eo, ev, n=40, aux=kg is not None)[0],
            dmp2=KC.direct_kmp2(st, eo, ev),
            so=KC.kdrpa_second_order(st, eo, ev),
        )
    return out, dict(kb=kb, eps=eps, C=C, st=st, e_hf=e)


def gamma_ref(S, h, I, enn, nelec, vm, N=1):
    nocc = nelec // 2
    e, eps, it, C = rhf(S, h, I, enn, nelec, conv=1e-12, kshift=0.0, return_mo=True)
    out = {}
    for cv in CONVS:
        eo, ev = denominators(eps, nocc, vm, cv)
        epsd = np.concatenate([eo, ev])
        out[cv] = dict(
            mp2=gamma_mp2(C, epsd, nocc, eri=I)[0] / N,
            plasmon=gamma_drpa(C, epsd, nocc, eri=I, method="plasmon") / N,
            quad=gamma_drpa(C, epsd, nocc, eri=I, method="quad", n=40) / N,
        )
    return out


def show(tag, k, g):
    for cv in CONVS:
        print(
            f"{tag} {cv}: KMP2 {k[cv]['mp2']:.13e} d {k[cv]['mp2'] - g[cv]['mp2']:.1e} | k-dRPA plasmon "
            f"{k[cv]['plasmon']:.13e} d {k[cv]['plasmon'] - g[cv]['plasmon']:.1e} quad40 d "
            f"{k[cv]['quad'] - g[cv]['quad']:.1e} | quad-plasmon {k[cv]['quad'] - k[cv]['plasmon']:.1e} | "
            f"(d) O(Pi^2) - direct KMP2 {k[cv]['so'] - k[cv]['dmp2']:.1e}",
            flush=True,
        )


def anchor_a():
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    g = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    k, _ = kcorr(cell, (1, 1, 1), 2)
    show(
        "(a) H2 1x1x1", k, gamma_ref(g["S"], g["h"], g["I"], g["enn"], 2, g["madelung"])
    )


def anchor_b(name, cell, n, nelec, prec=1e-4, th=1e-8, mutants=()):
    t = time.time()
    gcut = PK.aft_gcut(cell, prec)
    sc = PK.supercell_cell(cell, n)
    N = int(np.prod(n))
    g = PK.gamma_aft(sc, gcut=gcut, thresh=th)
    ref = gamma_ref(g["S"], g["h"], g["I"], g["enn"], nelec * N, madelung(sc), N)
    print(f"  supercell ref {time.time() - t:.0f}s", flush=True)
    k, _ = kcorr(cell, n, nelec, gcut=gcut, thresh=th)
    show(f"(b) {name} {n}", k, ref)
    for m in mutants:
        KC._MUTANT = m
        km, _ = kcorr(cell, n, nelec, gcut=gcut, thresh=th)
        KC._MUTANT = None
        for cv in CONVS:
            print(
                f"  MUTANT {m} {cv}: dMP2 {km[cv]['mp2'] - ref[cv]['mp2']:+.2e} d-dRPA plasmon "
                f"{km[cv]['plasmon'] - ref[cv]['plasmon']:+.2e} quad {km[cv]['quad'] - ref[cv]['quad']:+.2e}",
                flush=True,
            )


def anchor_c():
    al = 0.5
    cell = Cell(H2_A, H2_ATOMS, {"H": [[0, [al, 1.0]]]})
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i, j in [(0, 0), (1, 1), (0, 1)]
        for h in np.ndindex(2, 2, 2)
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * al, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    kg = KG.build_kgdf(cell, (1, 1, 3), auxmol=aux, spherical=False)
    kd, ctx = kcorr(cell, (1, 1, 3), 2)
    kbB, ctxB = kcorr(cell, (1, 1, 3), 2, kg=kg)
    # same orbitals on both routes (isolates the ov assembly from the SCF-on-B difference)
    stB = KC.kB_ov(kg, ctx["C"], 1)
    print(
        f"(c) max|V_B - V_AFT| same C {abs(stB['V'] - ctx['st']['V']).max():.1e} max|Kq diff| "
        f"{abs(stB['Kq'] - ctx['st']['Kq']).max():.1e}"
    )
    for cv in CONVS:
        eo, ev = KC.k_denominators(ctx["eps"], 1, ctx["kb"]["madelung"], cv)
        print(
            f"(c) same C {cv}: KMP2 B-AFT {KC.kmp2(stB, eo, ev)[0] - kd[cv]['mp2']:.1e}; k-dRPA aux-side quad40 - AFT "
            f"plasmon {KC.kdrpa_quad(stB, eo, ev, n=40)[0] - kd[cv]['plasmon']:.1e} (AFT quad40 - plasmon "
            f"{kd[cv]['quad'] - kd[cv]['plasmon']:.1e})"
        )
    show("(c) own B-SCF vs dense", kbB, kd)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c"]
    if "a" in which:
        anchor_a()
    if "b" in which:
        anchor_b("H2", Cell(H2_A, H2_ATOMS, "sto-3g"), (1, 1, 3), 2)
        anchor_b("H2", Cell(H2_A, H2_ATOMS, "sto-3g"), (2, 2, 2), 2)
    if "mut" in which:
        anchor_b(
            "H2",
            Cell(H2_A, H2_ATOMS, "sto-3g"),
            (1, 1, 3),
            2,
            mutants=("kb_wrong", "no_conj", "no_madelung", "rpa_k_wrong"),
        )
    if "btri" in which:
        anchor_b(
            "tri",
            Cell(TRI_A, TRI_ATOMS, SP),
            (1, 1, 3),
            4,
            prec=1e-4,
            th=1e-8,
            mutants=("kb_wrong", "no_conj"),
        )
    if "c" in which:
        anchor_c()
