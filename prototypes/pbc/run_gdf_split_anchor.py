"""Iteration 23 anchors / measurements for the RS-GDF range split (pbc_gdf_split.py; PySCF pbc only for the
diamond hcore/E_nn, which are common to both sides of every comparison).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_gdf_split_anchor.py
       {anchor0|span|tri [parts]|diamond <basis> [w]|neg <cell>|exactG <cell>|counts|wsweep}   (<cell> = tri | dsto)
(OMP_NUM_THREADS=1 matters: PySCF's libcint intor is OpenMP-threaded over every core by default.)

Seen BEFORE these predictions were written: H2/STO-3G a=4, cc-pvdz-ri, w=1: lam=0 is bitwise pbc_gdf (J2, J3, B);
lam=1 max|dJ3| 5.5e-12, max|dJ2| 1.8e-11, max|dERI| 1.4e-12 (moved: 4/6 aux shells, both orbital prims 0.1689).
Also seen: count_sr3_ferric reproduces ferric's own counters on diamond_prim STO-3G/cc-pvdz-ri EXACTLY
(SR3 16023080, SR2 159064).

PREDICTIONS (written before the runs below):
 P1 lam = 0 (criterion moves nothing): J2/J3/B BITWISE equal to pbc_gdf.build_gdf (tri s+p, diamond STO-3G).
 P2 lam = 1: max|dJ3|, max|dJ2| <= 1e-10 absolute, |dE| <= 1e-9, where d = split - unsplit.  Expected size: the
    moved block's G-sphere tail ~ (2/pi) q_pair q_aux e^{-gcut^2 s/4}/(gcut s/2) <= ~0.1 q q prec for s = 1/w^2,
    times an l-polynomial (G/2sqrt(a))^l of up to ~1e2 for f aux / d orbitals => 1e-13 .. 1e-10.
 P3 prec 1e-11 -> 1e-13 -> 1e-15 at lam = 1: the residual SHRINKS (it is truncation, not partition).
 P4 lam scan (tri): flat (<= 1e-10) for lam <= 1, then rising roughly like prec^{1/lam} times charges:
    lam 2 ~ 1e-7, lam 4 ~ 1e-4 .. 1e-3 in J3 (moving a block whose FT does NOT converge in the sphere).
 P5 w = 0.7 / 1.0 / 1.4 at lam = 1: dE (split - unsplit at the same w) <= 1e-9 at every w, and E_split(w) is
    w-independent to the unsplit's own level (Iteration 2: 1e-9).
 P6 exact-span anchor (H2, 24 pair-product aux): w = 0.8 moves nothing, w = 1.2 moves EVERYTHING (orbital 0.5 <=
    0.72, aux 1.0 <= 1.44): fitted ERI vs the pure-AFT exact I <= 1e-10 in both (Iteration 2: 1.2e-11 at w = 1.2).
 P7 mutants, each must miss the unsplit/exact by >= 1e-6 (predicted sizes): no_g0 ~ c0 S q ~ 1e-2..1;
    both (moved block also in real space) ~ |moved block| ~ 1e-1..1; double_ss_s ~ |ss x X^s| ~ 1e-2..1;
    lam = 4 ~ 1e-4..1e-3.
 P8 diamond_prim (cart orbital basis, cc-pvdz-ri spherical aux, w = 1): STO-3G and cc-pVDZ |dE| <= 1e-9.
 P9 counts (ferric's exact rule, zero-coefficient primitives kept): lam = 0 == unsplit EXACTLY; trim ratio at w = 1
    0.10-0.15 for cc-pVDZ/def2-universal-jkfit (the 0.095 estimate used the tighter V3 erfc envelope, this
    count keeps ferric's V0 form), similar for cc-pvdz-ri; w = 2 ~0.01-0.02; 'pieces' > 'trim'.
ARTIFACT HYPOTHESES: a partition error (block omitted, counted twice, wrong G=0) is O(block) >= 1e-3, does NOT
shrink with prec and moves with w; a correct split leaves a residual that shrinks with prec and is flat in lam <= 1.
"Agreement because nothing moved" is excluded by reporting the moved counts and |moved block| every time.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bench"))

from pyscf import gto  # noqa: E402

from pbc_gamma import Cell, build_integrals, rhf  # noqa: E402
from pbc_gdf import build_gdf, eri_from_B, ferric_basis, jk_from_B  # noqa: E402
from pbc_gdf_split import Criterion, build_gdf_split, count_sr3_ferric  # noqa: E402

H2_A = np.eye(3) * 4.0
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
TRI_A = np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]])
TRI_ATOMS = [
    ("H", (0.1, 0.2, 0.3)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (2.4, 2.5, 2.2)),
    ("H", (3.6, 2.9, 2.6)),
]
SP_BASIS = {
    "H": gto.parse(
        "H S\n 3.42525091 0.15432897\n 0.62391373 0.53532814\n 0.16885540 0.44463454\nH P\n 0.8 1.0\n"
    )
}


def dmax(a, b):
    return float(abs(np.asarray(a) - np.asarray(b)).max())


def energy(ints_or_hse, B, nelec):
    S, h, enn = ints_or_hse
    return rhf(S, h, None, enn, nelec, conv=1e-12, jk=jk_from_B(B))[0]


def compare(r, hse, nelec, label):
    """split (r) vs its in-run unsplit reference r['ref']."""
    ref = r["ref"]
    try:
        e_s = energy(hse, r["B"], nelec)
    except (
        RuntimeError
    ):  # a broken mutant may not even converge; the tensors still say how broken
        e_s = float("nan")
    e_u = energy(hse, ref["B"], nelec)
    p = r["parts"]
    row = dict(
        case=label,
        dJ3=dmax(r["J3"], ref["J3"]),
        dJ2=dmax(r["J2"], ref["J2"]),
        dERI=dmax(eri_from_B(r["B"]), eri_from_B(ref["B"]))
        if r["B"].shape[1] <= 16
        else None,
        dE=e_s - e_u,
        E_unsplit=f"{e_u:.12f}",
        E_split=f"{e_s:.12f}",
        moved3_max=float(abs(p["moved3"]).max()),
        blk_real_vs_G=dmax(p["R_full"] - p["R_kept"], p["moved3"]),
        kept=r["naux_kept"],
        kept_ref=ref["naux_kept"],
        **{
            k: r["info"][k]
            for k in (
                "n_aux_smooth",
                "n_aux_mixed",
                "n_orb_prims_smooth",
                "Pss0_vs_Sss",
            )
        },
    )
    print(
        json.dumps(
            {
                k: (float(f"{v:.4g}") if isinstance(v, float) else v)
                for k, v in row.items()
            }
        ),
        flush=True,
    )
    return row


# ------------------------------------------------------------------------------ cases
def main_anchor0():
    for name, cell, aux in (
        (
            "H2/sto-3g",
            Cell(H2_A, H2_ATOMS, "sto-3g"),
            ferric_basis("cc-pvdz-ri", ["H"]),
        ),
        (
            "tri/s+p",
            Cell(TRI_A, TRI_ATOMS, SP_BASIS),
            ferric_basis("cc-pvdz-ri", ["H"]),
        ),
    ):
        a = build_gdf(cell, aux)
        b = build_gdf_split(cell, aux, lam=0.0)
        print(
            name,
            "lam=0 g0=B vs pbc_gdf (bitwise, max|d|)",
            {
                k: (bool(np.array_equal(a[k], b[k])), dmax(a[k], b[k]))
                for k in ("J2", "J3", "B")
            },
            flush=True,
        )
        b = build_gdf_split(cell, aux, lam=0.0, g0="A")
        print(
            name,
            "lam=0 g0=A vs pbc_gdf",
            {
                k: (bool(np.array_equal(a[k], b[k])), dmax(a[k], b[k]))
                for k in ("J2", "J3", "B")
            },
            flush=True,
        )
        rb, ra = (
            build_gdf_split(cell, aux, lam=1.0),
            build_gdf_split(cell, aux, lam=1.0, g0="A"),
        )
        print(
            name,
            "lam=1 g0=A vs g0=B",
            {k: dmax(ra[k], rb[k]) for k in ("J2", "J3")},
            "max|J3|",
            float(abs(rb["J3"]).max()),
            "max|J2|",
            float(abs(rb["J2"]).max()),
            flush=True,
        )


def span_cell(w_unused=None):
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
    return cell, aux


def main_span():
    cell, aux = span_cell()
    ref = build_integrals(cell, None, exxdiv=None, verbose=False)
    hse = (ref["S"], ref["h"], ref["enn"])
    e_exact = rhf(ref["S"], ref["h"], ref["I"], ref["enn"], 2, conv=1e-12)[0]
    for w in (0.8, 1.0, 1.2):
        for mut, lam in (
            (None, 1.0),
            ("no_g0", 1.0),
            ("both", 1.0),
            ("double_ss_s", 1.0),
        ):
            if w == 0.8 and mut:
                continue
            r = build_gdf_split(cell, None, w=w, lam=lam, auxmol=aux, mutant=mut)
            print(
                json.dumps(
                    dict(
                        w=w,
                        mutant=mut,
                        lam=lam,
                        dI_vs_exact=float(f"{dmax(eri_from_B(r['B']), ref['I']):.3g}"),
                        dE_vs_exact=float(f"{energy(hse, r['B'], 2) - e_exact:.3g}"),
                        n_aux_smooth=r["info"]["n_aux_smooth"],
                        n_orb_smooth=r["info"]["n_orb_prims_smooth"],
                    )
                ),
                flush=True,
            )


def main_tri(parts=("lam", "prec", "w", "metric", "mut")):
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    ints = build_integrals(
        cell, None, exxdiv=None, verbose=False
    )  # pure AFT: S, h, E_nn only are used
    hse = (ints["S"], ints["h"], ints["enn"])
    rc = {}
    print("# lam scan, w=1, prec 1e-13", flush=True)
    for lam in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0) if "lam" in parts else ():
        compare(
            build_gdf_split(cell, aux, lam=lam, reference=True, ref_cache=rc),
            hse,
            4,
            f"lam={lam}",
        )
    print("# prec scan, lam=1, w=1", flush=True)
    for prec in (1e-11, 1e-15) if "prec" in parts else ():
        compare(
            build_gdf_split(
                cell, aux, lam=1.0, prec=prec, reference=True, ref_cache=rc
            ),
            hse,
            4,
            f"prec={prec:g}",
        )
    print("# w scan, lam=1", flush=True)
    es = {}
    for w in (0.7, 1.0, 1.4) if "w" in parts else ():
        r = build_gdf_split(cell, aux, w=w, lam=1.0, reference=True, ref_cache=rc)
        row = compare(r, hse, 4, f"w={w}")
        es[w] = (energy(hse, r["B"], 4), row["E_unsplit"])
    print("# split_metric=False (metric unsplit, J3 split)", flush=True)
    if "metric" in parts:
        compare(
            build_gdf_split(
                cell, aux, lam=1.0, split_metric=False, reference=True, ref_cache=rc
            ),
            hse,
            4,
            "J2 unsplit",
        )
    print("# mutants (lam=1, w=1)", flush=True)
    for mut in ("no_g0", "both", "double_ss_s") if "mut" in parts else ():
        compare(
            build_gdf_split(
                cell, aux, lam=1.0, reference=True, mutant=mut, ref_cache=rc
            ),
            hse,
            4,
            f"MUTANT {mut}",
        )
    print(
        "# E(w): split",
        {w: v[0] for w, v in es.items()},
        "unsplit",
        {w: v[1] for w, v in es.items()},
        flush=True,
    )


def strip_zeros(bas):
    """Same functions, zero-coefficient primitives removed per segmented column (Python speed only)."""
    return {
        s: [[sh[0]] + [p for p in sh[1:] if p[1] != 0.0] for sh in shells]
        for s, shells in bas.items()
    }


def main_diamond(basis, w=1.0):
    import cell_facts as cf
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    sym, xyz, lat = cf.read_cell("diamond_prim")
    atoms = [(s, tuple(c)) for s, c in zip(sym, xyz)]
    obs = strip_zeros(ferric_basis(basis, sym))
    aux = ferric_basis("cc-pvdz-ri", sym)
    cell = Cell(lat, atoms, obs)
    t0 = time.time()
    pc = pgto.Cell(
        a=lat, atom=atoms, basis=obs, unit="B", cart=True, verbose=0, precision=1e-12
    ).build()
    mf = pscf.RHF(pc, exxdiv=None).rs_density_fit(auxbasis=aux)
    h, enn = mf.get_hcore(), pc.energy_nuc()
    t1 = time.time()
    r = build_gdf_split(cell, aux, w=w, lam=1.0, reference=True, verbose=True)
    t2 = time.time()
    hse = (r["S"], h, enn)
    print(
        f"# diamond_prim {basis} w={w}: hcore {t1 - t0:.0f}s, build (split + unsplit reference) {t2 - t1:.0f}s; "
        f"|S_ours - S_pyscf| {dmax(r['S'], pc.pbc_intor('int1e_ovlp')):.2g}",
        flush=True,
    )
    compare(r, hse, 12, f"diamond {basis} w={w}")
    from pbc_gamma import madelung

    vm = madelung(cell)

    def fit(J2, J3):
        sv, U = np.linalg.eigh(J2)
        k = sv > 1e-10
        return np.einsum("Pk,mnP->kmn", U[:, k] / np.sqrt(sv[k]), J3)

    e_u = energy(hse, r["ref"]["B"], 12)
    for lab, J2, J3 in (
        ("split J3 + unsplit J2", r["ref"]["J2"], r["J3"]),
        ("unsplit J3 + split J2", r["J2"], r["ref"]["J3"]),
    ):
        print(
            f"# hybrid {lab}: E - E_unsplit = {energy(hse, fit(J2, J3), 12) - e_u:.3e}",
            flush=True,
        )
    for lab, B in (("split", r["B"]), ("unsplit", r["ref"]["B"])):
        e = rhf(r["S"], h, None, enn, 12, conv=1e-12, kshift=vm, jk=jk_from_B(B))[0]
        print(
            f"# exxdiv=ewald E({lab}) = {e:.10f}  (ferric sph-aux run: sto-3g -74.0034040288, cc-pvdz -74.9757090070"
            f" with a SPHERICAL orbital basis)",
            flush=True,
        )
    print(f"# SCF {time.time() - t2:.0f}s", flush=True)


def main_counts():
    import cell_facts as cf

    sym, xyz, lat = cf.read_cell("diamond_prim")
    for ob in ("sto-3g", "cc-pvdz"):
        for ax in ("cc-pvdz-ri", "def2-universal-jkfit"):
            obs, aux = cf.ferric_shells(ob, sym, xyz), cf.ferric_shells(ax, sym, xyz)
            for w in (1.0, 2.0):
                base = None
                for lab, crit, mode in (
                    ("unsplit", None, "trim"),
                    ("lam0", Criterion(w, 0.0), "trim"),
                    ("trim", Criterion(w, 1.0), "trim"),
                    ("twocall", Criterion(w, 1.0), "twocall"),
                    ("pieces", Criterion(w, 1.0), "pieces"),
                ):
                    t = time.time()
                    c = count_sr3_ferric(obs, aux, lat, w=w, crit=crit, mode=mode)
                    base = base or c
                    print(
                        json.dumps(
                            dict(
                                obs=ob,
                                aux=ax,
                                w=w,
                                case=lab,
                                n_sr3=c["n_sr3"],
                                ratio=round(c["n_sr3"] / base["n_sr3"], 4),
                                n_sr2=c["n_sr2"],
                                ratio2=round(c["n_sr2"] / base["n_sr2"], 4),
                                secs=round(time.time() - t, 1),
                            )
                        ),
                        flush=True,
                    )


def _cell_aux(which):
    if which == "tri":
        return Cell(TRI_A, TRI_ATOMS, SP_BASIS), ferric_basis("cc-pvdz-ri", ["H"]), 4
    import cell_facts as cf

    sym, xyz, lat = cf.read_cell("diamond_prim")
    return (
        Cell(
            lat, [(s, tuple(c)) for s, c in zip(sym, xyz)], ferric_basis("sto-3g", sym)
        ),
        ferric_basis("cc-pvdz-ri", sym),
        12,
    )


def main_wsweep():
    """SR3 triplets vs omega (ferric's rule) against the LR half-G count at gcut = 2 w sqrt(ln 1e13)."""
    import cell_facts as cf

    sym, xyz, lat = cf.read_cell("diamond_prim")
    obs, aux = (
        cf.ferric_shells("cc-pvdz", sym, xyz),
        cf.ferric_shells("cc-pvdz-ri", sym, xyz),
    )
    base = count_sr3_ferric(obs, aux, lat, w=1.0)["n_sr3"]
    for w in (0.7, 1.0, 1.4, 2.0):
        row = dict(w=w, n_g_half=cf.count_g_half(lat, 2 * w * np.sqrt(np.log(1e13))))
        for mode in ("unsplit", "trim", "twocall"):
            c = count_sr3_ferric(
                obs,
                aux,
                lat,
                w=w,
                crit=None if mode == "unsplit" else Criterion(w, 1.0),
                mode=mode,
            )
            row[mode] = c["n_sr3"]
            row[mode + "_vs_unsplit_w1"] = round(c["n_sr3"] / base, 4)
        print(json.dumps(row), flush=True)


def main_neg(which):
    """Negative control added after the tri lam scan (lam 2/3/4 moved the SAME prim sets and missed by only 5e-11:
    the rule's two conditions are each sufficient, so a 2-4x overshoot of the thresholds is still nearly convergent).
    'all' moves EVERY combination (a_orb = a_aux = inf): the compact ones do not converge at gcut."""
    cell, aux, nelec = _cell_aux(which)
    if which == "tri":
        ints = build_integrals(cell, None, exxdiv=None, verbose=False)
        hse = (ints["S"], ints["h"], ints["enn"])
    else:
        from pyscf.pbc import gto as pgto
        from pyscf.pbc import scf as pscf

        pc = pgto.Cell(
            a=cell.a,
            atom=cell.atoms,
            basis=cell.basis,
            unit="B",
            cart=True,
            verbose=0,
            precision=1e-12,
        ).build()
        hse = (
            None,
            pscf.RHF(pc, exxdiv=None).rs_density_fit(auxbasis=aux).get_hcore(),
            pc.energy_nuc(),
        )
    rc = {}
    for lab, crit in (
        ("lam=2", Criterion(1.0, 2.0)),
        ("lam=4", Criterion(1.0, 4.0)),
        ("orb all", Criterion(1.0, a_orb=1e9, a_aux=1.0)),
        ("aux all", Criterion(1.0, a_orb=0.5, a_aux=1e9)),
        ("all", Criterion(1.0, a_orb=1e9, a_aux=1e9)),
    ):
        r = build_gdf_split(cell, aux, crit=crit, reference=True, ref_cache=rc)
        if hse[0] is None:
            hse = (r["S"], hse[1], hse[2])
        compare(r, hse, nelec, f"{which} NEG {lab}")


def main_exactG(which):
    """Independent construction for the MOVED columns/rows: J2[P_s, :] and J3[:, P_s] (smooth aux, alpha <= w^2) as
    pure-G sums of the full G=0-dropped kernel on a LARGER sphere (|G| <= 17: tail e^{-G^2/4alpha} <= e^{-72}).
    Neither construction is the reference; both are compared with it (split: gcut 10.94 G sums; unsplit: erfc
    real-space lattice sums + LR)."""
    cell, aux, _ = _cell_aux(which)
    from pbc_gamma import pair_ft

    r = build_gdf_split(cell, aux, reference=True, spherical=False)
    auxmol = gto.M(atom=cell.atoms, basis=aux, unit="B", cart=True, verbose=0)
    crit = Criterion(1.0, 1.0)
    cols = np.concatenate(
        [
            [crit.aux_smooth(auxmol.bas_exp(i).max())] * auxmol.bas_len_cart(i)
            for i in range(auxmol.nbas)
        ]
    )
    G = cell.gvectors(17.0)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    v = 4 * np.pi / np.einsum("gi,gi->g", G, G) / cell.vol
    X = pbc_gdf_aux_ft(auxmol, G)
    J2x = ((X[cols].conj() * v) @ X.T).real
    nao = cell.mol.nao
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(r["S"]) / np.diag(P0))
    P = (pair_ft(cell, G) * pn[:, None, None] * pn[None, :, None]).reshape(
        nao * nao, -1
    )
    J3x = ((P.conj() * v) @ X[cols].T).real
    J3s, J3u = (
        r["J3"].reshape(nao * nao, -1)[:, cols],
        r["ref"]["J3"].reshape(nao * nao, -1)[:, cols],
    )
    # J3 is symmetrised in (m,n) by finish(); symmetrise the reference the same way
    J3x = 0.5 * (
        J3x.reshape(nao, nao, -1) + J3x.reshape(nao, nao, -1).transpose(1, 0, 2)
    ).reshape(nao * nao, -1)
    print(
        json.dumps(
            dict(
                cell=which,
                nG=len(G),
                n_smooth_aux_cart=int(cols.sum()),
                J2_rows_split=dmax(r["J2"][cols], J2x),
                J2_rows_unsplit=dmax(r["ref"]["J2"][cols], J2x),
                J3_cols_split=dmax(J3s, J3x),
                J3_cols_unsplit=dmax(J3u, J3x),
                max_J3x=float(abs(J3x).max()),
                max_J2x=float(abs(J2x).max()),
            )
        ),
        flush=True,
    )


def pbc_gdf_aux_ft(auxmol, G):
    from pbc_gdf import aux_ft

    return aux_ft(auxmol, G)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "diamond":
        main_diamond(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    elif which in ("neg", "exactG"):
        {"neg": main_neg, "exactG": main_exactG}[which](sys.argv[2])
    elif which == "tri" and len(sys.argv) > 2:
        main_tri(tuple(sys.argv[2:]))
    else:
        {
            "anchor0": main_anchor0,
            "span": main_span,
            "tri": main_tri,
            "counts": main_counts,
            "wsweep": main_wsweep,
        }[which]()
