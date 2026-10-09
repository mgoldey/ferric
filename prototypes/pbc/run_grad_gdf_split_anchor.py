"""Iteration 26 anchors: forces / stress / k-point forces for the RANGE-SPLIT RS-GDF (pbc_grad_gdf_split.py,
pbc_kgrad_gdf_split.py; no PySCF pbc anywhere).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_grad_gdf_split_anchor.py
       {anchor0 | h2 | h3 | tri | span | s_h2 | s_h3 | s_tri | k_anchor0 | k_gamma | k <h2_113|h3_113> [sc] [fd] [mut]}

Seen BEFORE these predictions were written (smoke runs of the correct code only):
  H2/STO-3G a=4, ET-40 aux, RHF none: lam = 0 split force vs Iteration 18 at the same D 3.5e-18 ('sym' ket) /
  2.2e-14 (explicit ip1+ip2 ket); lam = 1 (2 smooth orbital prims, 8/10 aux shells... see info) E_split - E_unsplit
  1.8e-13, F_split - F_unsplit 1.9e-12, analytic - FD (0,z) -2.42e-9, (1,y) +1.3e-10 (= Iteration 18's floor).
  Stress, same cell: lam = 0 vs Iteration 19 2.8e-17; lam = 1 an - FD (0,0) -9.3e-10, (1,2) 3.5e-10, (2,0) -1.6e-12.

PREDICTIONS (written before the runs below):
 Q1 nothing moved (lam = 0): split force == Iteration 18 force and split stress == Iteration 19 stress at the same D,
    <= 1e-14 (the kept calls reduce to the unsplit call; the second call and every moved term are exact zeros).
    k: lam = 0 split build == pbc_kgrad_gdf.build (J2, J3 per q) and split k force == Iteration 21b's, <= 1e-14.
 Q2 lam = 1, analytic vs central FD (h = 1e-4) of the prototype's OWN split energy at the Iteration 18 / 19 / 21b
    floors to ~2 digits: forces H2 2.4e-9, H3 2.7e-9, tri 2.7e-9; stress H2 6.7e-9, H3 5.2e-8, tri 2.1e-7 (h^2
    truncation, same energies to 1e-13); sum F ~1e-14; F(ewald) == F(none).
 Q3 exact-aux span (H2 one s(0.5), 24 ghost pair-product aux): w = 0.8 moves nothing, w = 1.2 moves EVERYTHING (the
    kept SR walks are exact zeros): F_split == F_dense (pbc_grad_open, same h, S, E_nn) <= 1e-11 in both.
 Q4 k: 1x1x1 split k force == Gamma split force (this module) at the same D <= 1e-12; 1x1x3 == the supercell Gamma
    split force on every copy <= 1e-11 (unfolded D); FD at Iteration 21b's floor (H2 (0,z) -2.99e-9);
    E_k(split) - E_k(unsplit) <= 1e-9.
 Q5 lam = 1 split force - unsplit force (each at its own D) ~ the energy difference scale, <= 1e-10 (split and unsplit
    are the same energy to 1e-13).
ARTIFACT HYPOTHESES (each mutant must miss FD / the independent reference by >> the floor):
 - no_dSss (G = 0 force without the smooth-piece overlap derivative): O(c0 |Y q^c| |dS_ss|) ~ 1e-3..1e-2.
 - g0_full (G = 0 with the FULL inputs dS, q: the unsplit term): ~ c0 |Y (q - q^c)| |dS| + c0 |Y q^c||dS_ss|, 1e-3..1e-2;
   NOT blind in the span limit with everything moved (the right term is then exactly zero, the wrong one is not).
 - no_ss_pair (drop the NEW (P^ss | X^c) derivative): O(1e-3..1e-2); blind if no orbital primitive is smooth.
 - moved_lr (moved blocks differentiated with v_LR): O(1e-2).
 - stress no_sr_kstrain (drop dv_SR/d eps): O(0.1) diagonal and off-diagonal (dv_SR ~ -2 G_i G_j v'), the moved block
   is an O(1) part of J3's G-space energy.
 - k gamma_kernel (v_SR(|G|) instead of v_SR(|G+q|) on the moved blocks): BLIND at 1x1x1 (q = 0 only); at 1x1x3
   O(1e-4..1e-2), largest where |q| is comparable to the smallest |G| (small G carry the moved block).
 - A partition error in a derivative (term omitted or double counted) is O(block) and does NOT shrink with h; an FD
   residual that equals Iteration 18/19/21b's residual to 2-3 digits is truncation of the same energy.
 - Because F_split == F_unsplit to ~1e-12 (Q5), "the split force is the unsplit force" would ALSO pass FD. The
   split derivative never evaluates a moved block in real space (structural), so the moved terms must carry real
   weight: reported below as the size of the G-space parts, and killed individually by no_ss_pair / moved_lr.
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_gdf as GG
import pbc_grad_gdf_split as SP
import pbc_grad_open as PGO
from pbc_gamma import Cell, madelung
from pbc_gdf import even_tempered, ferric_basis, jk_from_B
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H = 1e-4
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))]
H3_ATOMS = H2_ATOMS + [("H", (1.6, 0.9, 0.7))]
ET40 = even_tempered(["H"], 1, 0.3, 2.5, 5)
MUT_F = ["no_dSss", "g0_full", "no_ss_pair", "moved_lr"]
MUT_S = [
    "no_dSss",
    "vol_full_S",
    "no_Sss",
    "g0_full",
    "no_ss_pair",
    "moved_lr",
    "no_sr_kstrain",
]
ALL9 = [(i, j) for i in range(3) for j in range(3)]


def with_vm(ints, ex):
    d = dict(ints)
    if ex == "none":
        d["madelung"] = 0.0
    return d


def moved_info(gd):
    i = gd["info"]
    return (
        f"moved aux shells {i['n_aux_smooth']} (+{i['n_aux_mixed']} mixed), smooth orb prims "
        f"{i['n_orb_prims_smooth']}, a_orb {i['a_orb']:.3g}, a_aux {i['a_aux']:.3g}"
    )


def grad_at(
    cell, ints, gd, aux, r, lindep, jac=None, ket="sym", cache=None, mutant=None
):
    SP._MUTANT = mutant
    try:
        return SP.gamma_gdf_split_grad(
            cell,
            ints,
            gd,
            aux,
            r["Da"],
            r["Db"],
            r["Fa"],
            r["Fb"],
            r["hyb"],
            w=gd["w"],
            lindep=lindep,
            spherical=gd["spherical"],
            aux_jac=jac,
            ket=ket,
            cache=cache,
        )
    finally:
        SP._MUTANT = None


def build(cell, aux, w, lam, spherical, lindep):
    gd = SP.split_gdf(cell, aux, w, lam, spherical=spherical, lindep=lindep)
    gd["spherical"] = spherical
    return gd


# ============================================================================================ Gamma forces
def anchor0():
    """Q1: lam = 0 vs Iteration 18 (forces) and Iteration 19 (stress) at the same D."""
    import pbc_stress as ST

    cases = [
        (
            "H2 ET-40 RHF",
            Cell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g"),
            ET40,
            1,
            1,
            True,
            12.0,
        ),
        (
            "H3 cc-pvdz-ri UHF(2,1)",
            Cell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g"),
            ferric_basis("cc-pvdz-ri", ["H"]),
            2,
            1,
            False,
            12.0,
        ),
        (
            "tri cc-pvdz-ri UHF(3,1)",
            Cell(TRI_A, TRI_MOVED, SP_BASIS),
            ferric_basis("cc-pvdz-ri", ["H"]),
            3,
            1,
            False,
            10.0,
        ),
    ]
    for label, cell, abas, na, nb, rst, gcut in cases:
        t = time.time()
        aux = GG.make_auxmol(cell, abas)
        ints = GG.integrals(cell, gcut, "none")
        gdu = GG.gdf(cell, aux, 1.0, lindep=1e-10)
        gd0 = build(cell, aux, 1.0, 0.0, True, 1e-10)
        r = GG.energy(cell, ints, gdu, na, nb, "HF", None, rst)
        gu, _, _ = GG.gamma_gdf_grad(
            cell,
            ints,
            gdu,
            aux,
            r["Da"],
            r["Db"],
            r["Fa"],
            r["Fb"],
            1.0,
            w=1.0,
            lindep=1e-10,
        )
        cache = {}
        gs, _, _ = grad_at(cell, ints, gd0, aux, r, 1e-10, cache=cache)
        ge, _, _ = grad_at(cell, ints, gd0, aux, r, 1e-10, ket="explicit", cache=cache)
        print(
            f"  {label}: J3/J2 lam0 - unsplit {abs(gd0['J3'] - gdu['J3']).max():.1e}/{abs(gd0['J2'] - gdu['J2']).max():.1e}"
            f"  |F_split - F_it18| sym {abs(gs - gu).max():.1e} explicit-ket {abs(ge - gu).max():.1e}  |F| "
            f"{abs(gu).max():.3f} ({time.time() - t:.0f}s)",
            flush=True,
        )
    # stress, H2
    cell = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    spec = dict(
        na=1,
        nb=1,
        xc="HF",
        restricted=True,
        exxdiv="none",
        gcut=12.0,
        aux=ET40,
        w=1.0,
        lam=0.0,
        lindep=0.0,
        spherical=True,
    )
    ints, gd, aux, r = SP.solve(cell, spec)
    an, _ = SP.stress_analytic(cell, spec, ints, gd, aux, r)
    ints0, gd0, aux0, r0 = ST.gdf_solve(cell, dict(spec))
    an0, _ = ST.gdf_analytic(cell, dict(spec), ints0, gd0, aux0, r0)
    print(
        f"  stress H2 ET-40 lam0 vs Iteration 19: {abs(an - an0).max():.1e} (|sigma*Om| {abs(an0).max():.3f}), "
        f"E {r['e'] - r0['e']:+.1e}",
        flush=True,
    )


def gamma_case(
    label,
    cell,
    aux_of,
    gcut,
    cases,
    comps,
    w=1.0,
    lam=1.0,
    spherical=True,
    lindep=1e-10,
    jac_of=None,
    dense=False,
    fd=True,
    mutants=MUT_F,
    unsplit=True,
):
    """cases: [(name, na, nb, restricted, exxdiv)]."""
    t = time.time()
    aux = aux_of(cell)
    jac = jac_of(cell) if jac_of else None
    ints = GG.integrals(cell, gcut, "ewald")
    gd = build(cell, aux, w, lam, spherical, lindep)
    print(
        f"## {label}: w {w} lam {lam}; {moved_info(gd)}; build {time.time() - t:.0f}s",
        flush=True,
    )
    cache, ref = {}, {}
    gdu = GG.gdf(cell, aux, w, spherical=spherical, lindep=lindep) if unsplit else None
    for name, na, nb, rst, ex in cases:
        d = with_vm(ints, ex)
        g0 = ref.get(name)
        r = GG.energy(
            cell, d, gd, na, nb, "HF", None, rst, (g0["Da"], g0["Db"]) if g0 else None
        )
        g, parts, diag = grad_at(cell, d, gd, aux, r, lindep, jac=jac, cache=cache)
        muts = {
            m: grad_at(cell, d, gd, aux, r, lindep, jac=jac, cache=cache, mutant=m)[0]
            for m in mutants
        }
        ref[(name, ex)] = dict(r=r, g=g, parts=parts, muts=muts)
        ref.setdefault(name, r)
        gpart = sum(abs(parts[k]) for k in ("J3_bra_g", "J3_aux_g", "J2_g")).max()
        line = (
            f"  {name:12s} {ex:5s}: E {r['e']:.12f} it {r['it']} drop {diag['n_drop']}  |sumF| {abs(g.sum(0)).max():.1e}"
            f"  max|G-space 2e parts| {gpart:.2e}"
        )
        if unsplit:
            ru = GG.energy(cell, d, gdu, na, nb, "HF", None, rst, (r["Da"], r["Db"]))
            gu, _, _ = GG.gamma_gdf_grad(
                cell,
                d,
                gdu,
                aux,
                ru["Da"],
                ru["Db"],
                ru["Fa"],
                ru["Fb"],
                1.0,
                w=w,
                lindep=lindep,
                spherical=spherical,
                aux_jac=jac,
            )
            line += f"  E_split-E_unsplit {r['e'] - ru['e']:+.1e} |F_split-F_unsplit| {abs(g - gu).max():.1e}"
        if dense:
            rd = PGO.run_case(cell, na, nb, "HF", d, restricted=rst)
            line += f"  E-E_dense {r['e'] - rd['e']:+.1e} |F-F_dense| {abs(g - rd['grad']).max():.1e}"
            ref[(name, ex)]["dense"] = rd["grad"]
        print(line, flush=True)
        if dense:
            for m, gm in muts.items():
                print(
                    f"      mutant {m:10s} |F-F_dense| {abs(gm - rd['grad']).max():.2e}",
                    flush=True,
                )
    exs = sorted({ex for _, _, _, _, ex in cases})
    for name in dict.fromkeys(c[0] for c in cases):
        if len(exs) == 2 and all((name, e) in ref for e in exs):
            print(
                f"  {name}: F(ewald)-F(none) {abs(ref[(name, 'ewald')]['g'] - ref[(name, 'none')]['g']).max():.1e}"
            )
    if not fd:
        return ref
    ev = {}
    for A, x in comps:
        for s in (1, -1):
            t = time.time()
            cd = PGd.displaced(cell, A, x, s * H)
            ad = aux_of(cd)
            ii = GG.integrals(cd, gcut, "ewald")
            gdd = build(cd, ad, w, lam, spherical, lindep)
            for name, na, nb, rst, ex in cases:
                r0 = ref[(name, ex)]["r"]
                ev[(name, ex, A, x, s)] = GG.energy(
                    cd,
                    with_vm(ii, ex),
                    gdd,
                    na,
                    nb,
                    "HF",
                    None,
                    rst,
                    (r0["Da"], r0["Db"]),
                )["e"]
            print(f"    FD ({A},{x}) {s:+d} ({time.time() - t:.0f}s)", flush=True)
    for name, na, nb, rst, ex in cases:
        g = ref[(name, ex)]["g"]
        fdv = {
            (A, x): (ev[(name, ex, A, x, 1)] - ev[(name, ex, A, x, -1)]) / (2 * H)
            for A, x in comps
        }
        print(
            f"  | {name} {ex} | an - FD "
            + " ".join(f"({A},{x}) {g[A, x] - fdv[(A, x)]:+.2e}" for A, x in comps)
            + " |",
            flush=True,
        )
        for m, gm in ref[(name, ex)]["muts"].items():
            print(
                f"      mutant {m:10s} max|an-FD| {max(abs(gm[A, x] - fdv[(A, x)]) for A, x in comps):.2e}"
                f"  |sumF| {abs(gm.sum(0)).max():.1e}",
                flush=True,
            )
    return ref


def main_h2():
    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    cases = [
        (n_, 1, 1, rst, ex)
        for n_, rst in (("RHF", True), ("UHF(1,1)", False))
        for ex in ("none", "ewald")
    ]
    gamma_case(
        "H2/STO-3G a=4, ET-40 aux",
        cell,
        lambda c: GG.make_auxmol(c, ET40),
        12.0,
        cases,
        [(0, 0), (0, 2), (1, 1)],
        lindep=0.0,
    )


def main_h3():
    cell = Cell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
    rib = ferric_basis("cc-pvdz-ri", ["H"])
    cases = [("UHF(2,1)", 2, 1, False, ex) for ex in ("none", "ewald")]
    gamma_case(
        "H3/STO-3G a=4.5 doublet, cc-pvdz-ri",
        cell,
        lambda c: GG.make_auxmol(c, rib),
        12.0,
        cases,
        [(0, 0), (2, 1)],
    )


def main_tri():
    cell = Cell(TRI_A, TRI_MOVED, SP_BASIS)
    rib = ferric_basis("cc-pvdz-ri", ["H"])
    cases = [
        ("RHF", 2, 2, True, "none"),
        ("UHF(3,1)", 3, 1, False, "none"),
        ("UHF(3,1)", 3, 1, False, "ewald"),
    ]
    gamma_case(
        "tri 4H s+p TRI_MOVED, cc-pvdz-ri",
        cell,
        lambda c: GG.make_auxmol(c, rib),
        10.0,
        cases,
        [(2, 1), (0, 2)],
    )


def main_span():
    """Q3: exact aux span, w = 0.8 (nothing moved) and 1.2 (everything moved), vs dense AFT."""
    from run_grad_gdf_anchor import ghost_aux

    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, {"H": [[0, [0.5, 1.0]]]})
    cases = [
        (n_, 1, 1, rst, "none") for n_, rst in (("RHF", True), ("UHF(1,1)", False))
    ]
    for w in (0.8, 1.2):
        gamma_case(
            f"span H2 s(0.5) + 24 ghost pair aux",
            cell,
            lambda c: ghost_aux(c)[0],
            12.0,
            cases,
            [(0, 0), (1, 2)],
            w=w,
            spherical=False,
            lindep=0.0,
            jac_of=lambda c: ghost_aux(c)[1],
            dense=True,
            fd=(w == 1.2),
            unsplit=True,
        )


# ============================================================================================ Gamma stress
def stress_case(label, cell, spec, comps, mutants=MUT_S, h=H):
    import pbc_stress as ST

    t = time.time()
    ints, gd, aux, r = SP.solve(cell, spec)
    an, parts = SP.stress_analytic(cell, spec, ints, gd, aux, r)
    print(
        f"## stress {label}: {moved_info(gd)}; E {r['e']:.12f}; |an - an^T| {abs(an - an.T).max():.1e} "
        f"({time.time() - t:.0f}s)",
        flush=True,
    )
    gpart = max(abs(parts[k]).max() for k in ("J3_g", "J2_g"))
    print(
        f"  max|G-space 2e stress parts| {gpart:.2e}; J3_g0_vol {parts['J3_g0_vol'][0, 0]:+.4e} J2_g0_vol "
        f"{parts['J2_g0_vol'][0, 0]:+.4e}",
        flush=True,
    )
    # unsplit (Iteration 19) at its own D, for Q5
    ints0, gd0, aux0, r0 = ST.gdf_solve(cell, dict(spec), guess=(r["Da"], r["Db"]))
    an0, _ = ST.gdf_analytic(cell, dict(spec), ints0, gd0, aux0, r0)
    print(
        f"  E_split - E_unsplit {r['e'] - r0['e']:+.1e}; |dE/deps split - unsplit| {abs(an - an0).max():.1e}",
        flush=True,
    )
    muts = {}
    for m in mutants:
        SP._MUTANT = m
        try:
            muts[m] = SP.stress_analytic(cell, spec, ints, gd, aux, r)[0]
        finally:
            SP._MUTANT = None
    fd = SP.fd_stress(cell, spec, comps, (r["Da"], r["Db"]), h=h)
    print("  | comp | analytic dE/deps | FD | an - FD |")
    for c in comps:
        print(
            f"  | {c} | {an[c]:+.12f} | {fd[c]:+.12f} | {an[c] - fd[c]:+.2e} |",
            flush=True,
        )
    for m, sm in muts.items():
        miss = [sm[c] - fd[c] for c in comps]
        worst = int(np.argmax(np.abs(miss)))
        print(
            f"      mutant {m:14s} max|an-FD| {abs(miss[worst]):.2e} at {comps[worst]}; "
            f"diag/offdiag max {max(abs(sm[c] - fd[c]) for c in comps if c[0] == c[1]):.1e}/"
            f"{max([abs(sm[c] - fd[c]) for c in comps if c[0] != c[1]] or [0]):.1e}",
            flush=True,
        )
    print(f"  ({time.time() - t:.0f}s)")


def stress_mut_only(label, cell, spec, mutants=MUT_S):
    """|mutant - analytic| per mutant (no FD; valid wherever the miss is >> the analytic - FD floor)."""
    t = time.time()
    ints, gd, aux, r = SP.solve(cell, spec)
    an, _ = SP.stress_analytic(cell, spec, ints, gd, aux, r)
    print(f"## stress mutants {label} ({time.time() - t:.0f}s)", flush=True)
    for m in mutants:
        SP._MUTANT = m
        try:
            d = SP.stress_analytic(cell, spec, ints, gd, aux, r)[0] - an
        finally:
            SP._MUTANT = None
        off = d - np.diag(np.diag(d))
        print(
            f"      mutant {m:14s} max|mut - an| {abs(d).max():.2e}; diag/offdiag max {abs(np.diag(d)).max():.1e}/"
            f"{abs(off).max():.1e}",
            flush=True,
        )


def main_s_mut():
    import pbc_stress as ST

    cell = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    spec = dict(
        na=1,
        nb=1,
        xc="HF",
        restricted=True,
        exxdiv="none",
        gcut=12.0,
        aux=ET40,
        w=1.0,
        lam=1.0,
        lindep=0.0,
        spherical=True,
    )
    stress_mut_only("H2 ET-40 RHF none", cell, spec)


def main_s_h2():
    import pbc_stress as ST

    cell = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    spec = dict(
        na=1,
        nb=1,
        xc="HF",
        restricted=True,
        exxdiv="none",
        gcut=12.0,
        aux=ET40,
        w=1.0,
        lam=1.0,
        lindep=0.0,
        spherical=True,
    )
    stress_case("H2 ET-40 RHF none", cell, spec, ALL9)


def main_s_h3():
    import pbc_stress as ST

    cell = ST.FixedCell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
    spec = dict(
        na=2,
        nb=1,
        xc="HF",
        restricted=False,
        exxdiv="ewald",
        gcut=12.0,
        aux=ferric_basis("cc-pvdz-ri", ["H"]),
        w=1.0,
        lam=1.0,
        lindep=1e-10,
        spherical=True,
    )
    stress_case(
        "H3 cc-pvdz-ri UHF(2,1) ewald",
        cell,
        spec,
        [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)],
    )


def main_s_tri():
    import pbc_stress as ST

    cell = ST.FixedCell(TRI_A, TRI_MOVED, SP_BASIS)
    spec = dict(
        na=3,
        nb=1,
        xc="HF",
        restricted=False,
        exxdiv="ewald",
        gcut=10.0,
        aux=ferric_basis("cc-pvdz-ri", ["H"]),
        w=1.0,
        lam=1.0,
        lindep=1e-10,
        spherical=True,
    )
    stress_case(
        "tri cc-pvdz-ri UHF(3,1) ewald", cell, spec, [(0, 0), (2, 2), (0, 1), (1, 2)]
    )


# ============================================================================================== k-point
ET_SP = {
    "H": gto.parse("""
H S
  4.7  1.0
H S
  1.9  1.0
H S
  0.75 1.0
H S
  0.3  1.0
H P
  1.25 1.0
H P
  0.5  1.0
""")
}


def k_setup(cell, n, gcut, lam, lindep=1e-10, split=True):
    import pbc_kgrad_gdf as KGG
    import pbc_kgrad_gdf_split as KS
    import pbc_kpts as PK

    kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
    aux = KGG.make_auxmol(cell, ET_SP)
    g = (
        KS.build(cell, n, aux, w=1.0, lam=lam, lindep=lindep)
        if split
        else KGG.build(cell, n, aux, w=1.0, lindep=lindep)
    )
    return {"none": dict(kb, madelung=0.0), "ewald": kb}, g


def k_scf2(kbs, g, na, nb, rst, guess=None):
    import pbc_kgrad as KG
    import pbc_kgrad_gdf_split as KS

    jk = KS.jk(g)
    r0 = KG.kscf(kbs["none"], na, nb, restricted=rst, guess=guess, jk=jk)
    r1 = KG.kscf(
        kbs["ewald"], na, nb, restricted=rst, guess=(r0["Da"], r0["Db"]), jk=jk
    )
    return {"none": r0, "ewald": r1}


def kgrad_mut(cell, kb, g, r, rst, m):
    import pbc_kgrad_gdf_split as KS

    KS._MUTANT = m
    try:
        return KS.grad(cell, kb, g, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst)[
            0
        ]
    finally:
        KS._MUTANT = None


def main_k_anchor0():
    """Q1 (k): lam = 0 split build / force == pbc_kgrad_gdf (Iteration 21b), H2 1x1x3."""
    import pbc_kgrad_gdf as KGG
    import pbc_kgrad_gdf_split as KS
    from run_kgrad_anchor import H2

    cell = Cell(H2["a"], H2["atoms"], H2["basis"])
    t = time.time()
    kbs, g0 = k_setup(cell, (1, 1, 3), 10.0, 0.0)
    _, gu = k_setup(cell, (1, 1, 3), 10.0, 0.0, split=False)
    d3 = max(abs(a["J3"] - b["J3"]).max() for a, b in zip(g0["per_q"], gu["per_q"]))
    d2 = max(abs(a["J2"] - b["J2"]).max() for a, b in zip(g0["per_q"], gu["per_q"]))
    rs = k_scf2(kbs, gu, 1, 1, True)
    for ex in ("none", "ewald"):
        r = rs[ex]
        fs = KS.grad(
            cell, kbs[ex], g0, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=True
        )[0]
        fu = KGG.grad(
            cell, kbs[ex], gu, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=True
        )[0]
        print(
            f"  H2 1x1x3 lam0 {ex}: max_q|dJ3| {d3:.1e} |dJ2| {d2:.1e}  |F_split - F_it21b| {abs(fs - fu).max():.1e} "
            f"|F| {abs(fu).max():.3f} ({time.time() - t:.0f}s)",
            flush=True,
        )


def main_k_gamma():
    """Q4a: 1x1x1 split k force == Gamma split force (pbc_grad_gdf_split) at the same D; gamma_kernel blind here."""
    import pbc_kgrad_gdf_split as KS
    from run_kgrad_anchor import H2, H3

    for name, s, na, nb, rst in (
        ("H2 RHF", H2, 1, 1, True),
        ("H3 UHF(2,1)", H3, 2, 1, False),
    ):
        c = Cell(s["a"], s["atoms"], s["basis"])
        t = time.time()
        kbs, g = k_setup(c, (1, 1, 1), 10.0, 1.0)
        rs = k_scf2(kbs, g, na, nb, rst)
        gdg = build(c, g["auxmol"], 1.0, 1.0, True, 1e-10)
        for ex in ("none", "ewald"):
            r = rs[ex]
            gk = KS.grad(
                c, kbs[ex], g, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst
            )[0]
            ints = GG.integrals(c, 10.0, ex)
            Da, Db = r["Da"][0].real, r["Db"][0].real
            J_a, K_a = jk_from_B(gdg["B"])(Da)
            J_b, K_b = jk_from_B(gdg["B"])(Db)
            S = ints["S"]
            Fa = ints["h"] + J_a + J_b - K_a - ints["madelung"] * S @ Da @ S
            Fb = ints["h"] + J_a + J_b - K_b - ints["madelung"] * S @ Db @ S
            e_g = (
                0.5 * (np.sum((ints["h"] + Fa) * Da) + np.sum((ints["h"] + Fb) * Db))
                + ints["enn"]
            )
            gg, _, _ = SP.gamma_gdf_split_grad(
                c, ints, gdg, g["auxmol"], Da, Db, Fa, Fb, 1.0, w=1.0
            )
            gkm = kgrad_mut(c, kbs[ex], g, r, rst, "gamma_kernel")
            print(
                f"  {name} {ex}: E_k - E_gamma(split, same D) {r['e'] - e_g:+.1e}  |F_k - F_gamma| "
                f"{abs(gk - gg).max():.1e}  sumF {abs(gk.sum(0)).max():.1e}  gamma_kernel mutant |dF| "
                f"{abs(gkm - gk).max():.1e} ({time.time() - t:.0f}s)",
                flush=True,
            )


def main_k(name, flags):
    import pbc_kgrad as KG
    import pbc_kgrad_gdf as KGG
    import pbc_kgrad_gdf_split as KS
    import pbc_kpts as PK
    from pbc_kuhf import unfold_dm
    from run_kgrad_anchor import H2, H3

    CASES = {
        "h2_113": (H2, (1, 1, 3), 1, 1, True, 10.0, [(0, 0), (0, 2), (1, 1)]),
        "h3_113": (H3, (1, 1, 3), 2, 1, False, 10.0, [(0, 0), (2, 1)]),
    }
    s, n, na, nb, rst, gcut, comps = CASES[name]
    cell = Cell(s["a"], s["atoms"], s["basis"])
    t = time.time()
    kbs, g = k_setup(cell, n, gcut, 1.0)
    pc = g["pc"]
    print(
        f"## k {name}: mesh {n} ({na},{nb}); split build {g['wall']:.0f}s; aux shells smooth "
        f"{sum(all(g['crit'].aux_smooth(e) for e in g['auxmol'].bas_exp(i)) for i in range(g['auxmol'].nbas))}/"
        f"{g['auxmol'].nbas}; |q^c - q| {abs(pc.q_c - pc.q).max():.3f}; kept per q "
        f"{[int(p['keep'].sum()) for p in g['per_q']]}/{len(g['per_q'][0]['s'])}",
        flush=True,
    )
    rs = k_scf2(kbs, g, na, nb, rst)
    _, gu = k_setup(cell, n, gcut, 0.0, split=False)
    ref = {}
    for ex in ("none", "ewald"):
        r = rs[ex]
        ru = KG.kscf(
            kbs[ex], na, nb, restricted=rst, guess=(r["Da"], r["Db"]), jk=KGG.jk(gu)
        )
        gk, pk, dg = KS.grad(
            cell, kbs[ex], g, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst
        )
        fu = KGG.grad(
            cell, kbs[ex], gu, ru["Da"], ru["Db"], ru["Fa"], ru["Fb"], restricted=rst
        )[0]
        muts = {
            m: kgrad_mut(cell, kbs[ex], g, r, rst, m)
            for m in ("gamma_kernel", "no_dSss", "g0_full", "no_ss_pair")
        }
        ref[ex] = dict(r=r, g=gk, muts=muts)
        print(
            f"  {ex:5s}: E {r['e']:.12f} err {r['err']:.1e}; E_split - E_unsplit {r['e'] - ru['e']:+.1e}; "
            f"|F_split - F_unsplit| {abs(gk - fu).max():.1e}; sumF {abs(gk.sum(0)).max():.1e} ({time.time() - t:.0f}s)",
            flush=True,
        )
    print(f"  F(ewald) - F(none) {abs(ref['ewald']['g'] - ref['none']['g']).max():.1e}")
    if "sc" in flags:
        t = time.time()
        sc = PK.supercell_cell(cell, n)
        N = int(np.prod(n))
        aux_sc = KGG.make_auxmol(sc, ET_SP)
        gd = build(sc, aux_sc, 1.0, 1.0, True, 1e-10)
        ints = GG.integrals(sc, gcut, "ewald")
        vm_sc = madelung(sc)
        # energy-level gamma_kernel mutant (k energy with the Gamma kernel on the moved blocks)
        gmk = KS.build(
            cell,
            n,
            g["auxmol"],
            w=1.0,
            lam=1.0,
            lindep=1e-10,
            deriv=False,
            mutant="gamma_kernel",
        )
        print(f"  supercell split build {time.time() - t:.0f}s", flush=True)
        cache = {}
        for ex in ("none", "ewald"):
            ig = dict(ints, madelung=vm_sc if ex == "ewald" else 0.0)
            r = ref[ex]["r"]
            Da = unfold_dm(cell, kbs[ex], r["Da"]).real
            Db = unfold_dm(cell, kbs[ex], r["Db"]).real
            J_a, K_a = jk_from_B(gd["B"])(Da)
            J_b, K_b = jk_from_B(gd["B"])(Db)
            S = ints["S"]
            Fa = ints["h"] + J_a + J_b - K_a - ig["madelung"] * S @ Da @ S
            Fb = ints["h"] + J_a + J_b - K_b - ig["madelung"] * S @ Db @ S
            e_sc = (
                0.5 * (np.sum((ints["h"] + Fa) * Da) + np.sum((ints["h"] + Fb) * Db))
                + ints["enn"]
            )
            gsc, _, _ = SP.gamma_gdf_split_grad(
                sc, ig, gd, aux_sc, Da, Db, Fa, Fb, 1.0, w=1.0, cache=cache
            )
            gsc = gsc.reshape(N, cell.mol.natm, 3)
            print(
                f"  (sc) {ex:5s}: E_k - E_sc/N {r['e'] - e_sc / N:+.1e}; max_c|F_k - F_sc(c)| "
                f"{abs(gsc - ref[ex]['g'][None]).max():.1e}; spread {abs(gsc - gsc.mean(0)).max():.1e}",
                flush=True,
            )
            for m, gm in ref[ex]["muts"].items():
                print(
                    f"      mutant {m:12s} vs supercell {abs(gm - gsc[0]).max():.1e}",
                    flush=True,
                )
        e_k = _k_energy_at(kbs["none"], g, ref["none"]["r"], rst)
        e_m = _k_energy_at(kbs["none"], gmk, ref["none"]["r"], rst)
        print(
            f"  energy-level gamma_kernel build at the same D: E_mut - E {e_m - e_k:+.2e} (none)",
            flush=True,
        )
    if "fd" in flags:
        res = {}
        for A, x in comps:
            ev = {}
            for sgn in (1, -1):
                t = time.time()
                cd = PGd.displaced(cell, A, x, sgn * H)
                kd, gdd = k_setup(cd, n, gcut, 1.0)
                jk = KS.jk(gdd)
                for ex in ("none", "ewald"):
                    r0 = ref[ex]["r"]
                    ev[(ex, sgn)] = KG.kscf(
                        kd[ex],
                        na,
                        nb,
                        restricted=rst,
                        guess=(r0["Da"], r0["Db"]),
                        jk=jk,
                    )["e"]
                print(f"    FD ({A},{x}) {sgn:+d} ({time.time() - t:.0f}s)", flush=True)
            for ex in ("none", "ewald"):
                res[(ex, A, x)] = (ev[(ex, 1)] - ev[(ex, -1)]) / (2 * H)
        for ex in ("none", "ewald"):
            g = ref[ex]["g"]
            print(
                f"  | {ex} | an - FD "
                + " ".join(
                    f"({A},{x}) {g[A, x] - res[(ex, A, x)]:+.2e}" for A, x in comps
                )
                + " |"
            )
            for m, gm in ref[ex]["muts"].items():
                print(
                    f"      mutant {m:12s} max|an-FD| {max(abs(gm[A, x] - res[(ex, A, x)]) for A, x in comps):.2e}"
                )


def main_sc_ket():
    """Copy spread of the Gamma force on the H2 1x1x3 supercell (all copies must be equal): split 'sym' vs
    'explicit' ket vs the unsplit Iteration 18 force, each at its own SCF density."""
    import pbc_kgrad_gdf as KGG
    import pbc_kpts as PK
    from run_kgrad_anchor import H2

    cell = Cell(H2["a"], H2["atoms"], H2["basis"])
    sc = PK.supercell_cell(cell, (1, 1, 3))
    aux = KGG.make_auxmol(sc, ET_SP)
    ints = GG.integrals(sc, 10.0, "none")
    gd = build(sc, aux, 1.0, 1.0, True, 1e-10)
    gdu = GG.gdf(sc, aux, 1.0, lindep=1e-10)
    r = GG.energy(sc, ints, gd, 3, 3, "HF", None, True)
    ru = GG.energy(sc, ints, gdu, 3, 3, "HF", None, True, (r["Da"], r["Db"]))
    cache = {}
    for ket, mut in (
        ("sym", None),
        ("explicit", None),
        ("sym", "cancel_metric"),
        ("sym", "asym_wm"),
    ):
        g, parts, _ = grad_at(
            sc, ints, gd, aux, r, 1e-10, ket=ket, cache=cache, mutant=mut
        )
        ket = ket if mut is None else f"{ket} + {mut}"
        g = g.reshape(3, 2, 3)
        print(
            f"  split {ket:8s}: copy spread {abs(g - g.mean(0)).max():.1e}  sumF {abs(g.sum((0, 1))).max():.1e}",
            flush=True,
        )
        groups = dict(
            SR3=("J3_bra_sr", "J3_aux_sr"),
            G3=("J3_bra_g", "J3_aux_g"),
            J2_sr=("J2_sr",),
            J2_g=("J2_g",),
            J3_g0=("J3_g0",),
            one_e=("S", "T", "Vlr_basis", "Vlr_nuc", "nn"),
        )
        print(
            "    |sum_A part| "
            + ", ".join(
                f"{k} {abs(sum(parts[p] for p in v).sum(0)).max():.1e}"
                for k, v in groups.items()
            ),
            flush=True,
        )
    gu, pu, _ = GG.gamma_gdf_grad(
        sc,
        ints,
        gdu,
        aux,
        ru["Da"],
        ru["Db"],
        ru["Fa"],
        ru["Fb"],
        1.0,
        w=1.0,
        lindep=1e-10,
    )
    groups = dict(
        SR3=("J3_bra_sr", "J3_aux_sr"),
        LR3=("J3_bra_lr", "J3_aux_lr"),
        J2_sr=("J2_sr",),
        J2_lr=("J2_lr",),
        J3_g0=("J3_g0",),
    )
    print(
        "    unsplit |sum_A part| "
        + ", ".join(
            f"{k} {abs(sum(pu[p] for p in v).sum(0)).max():.1e}"
            for k, v in groups.items()
        ),
        flush=True,
    )
    gu = gu.reshape(3, 2, 3)
    print(
        f"  unsplit (Iteration 18): copy spread {abs(gu - gu.mean(0)).max():.1e}; E_split - E_unsplit "
        f"{r['e'] - ru['e']:+.1e}",
        flush=True,
    )


def _k_energy_at(kb, g, r, rst):
    """Energy per cell of the k RS-GDF build g at the fixed density r (one J/K evaluation, no SCF)."""
    import pbc_kgrad_gdf_split as KS

    Nk = kb["Nk"]
    jk = KS.jk(g)
    Da, Db = r["Da"], r["Db"]
    Ja, Ka = jk(Da)
    Jb, Kb = jk(Db)
    S, h = kb["S"], kb["h"]
    vM = kb["madelung"]
    e = 0.0
    for Ds, Ks in ((Da, Ka), (Db, Kb)):
        F = h + Ja + Jb - Ks - vM * np.einsum("kab,kbc,kcd->kad", S, Ds, S)
        e += 0.5 * np.einsum("kmn,knm->", h + F, Ds).real
    return e / Nk + kb["enn"]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "anchor0"
    t0 = time.time()
    if which == "k":
        main_k(sys.argv[2], set(sys.argv[3:]))
    else:
        {
            "anchor0": anchor0,
            "h2": main_h2,
            "h3": main_h3,
            "tri": main_tri,
            "span": main_span,
            "s_h2": main_s_h2,
            "s_h3": main_s_h3,
            "s_mut": main_s_mut,
            "sc_ket": main_sc_ket,
            "s_tri": main_s_tri,
            "k_anchor0": main_k_anchor0,
            "k_gamma": main_k_gamma,
        }[which]()
    print(f"[{which}] done in {time.time() - t0:.0f}s")
