"""Iteration 18 anchors for Gamma RS-GDF forces (pbc_grad_gdf.py; no PySCF pbc anywhere).
Usage: python3 run_grad_gdf_anchor.py {h2|span|lindep|hscan|h3|tri}

Every case: analytic force vs central FD (h = 1e-4) of the prototype's OWN RS-GDF energy (pbc_gdf B tensors, pure-AFT
h/S/E_nn from build_integrals, pbc_grad_open.scf to max|X^T[F_s,D_s]X| < 1e-11, seeded from the reference density).
The GDF is rebuilt at every displaced geometry with the SAME (geometry-independent) image/G ranges, aux moving with
its atom (or, for 'span', with the pair midpoints it sits on).

PREDICTIONS (written before the full runs; a first H2 smoke run of the correct code is the only thing seen):
 P1 correct code, no eigenvalue cut active: |analytic - FD| at the Iteration-16 FD floor (~2.4e-9 on H2 (0,z)),
    RHF and UHF, exxdiv none and ewald, s-only and s+p orbitals, cart and spherical (l<=2) aux.
 P2 sum_A F_A ~ 1e-15 (the fitted energy is exactly translation invariant: every centre moves).
 P3 F(ewald) == F(none) to SCF precision (same Madelung algebra as Iteration 16/17; the fit does not touch it).
 P4 UHF(n,n) == RHF to ~1e-12.
 P5 'span' (aux = the 24 exact pair-product Gaussians on pair-midpoint ghost centres, which move with BOTH atoms):
    E_gdf == E_dense (1e-12, Iteration 2) at every geometry, hence F_gdf == F_dense(pbc_grad_open, same h) to ~1e-11.
    This is the independent construction: the dense side never forms a fit, the GDF side never forms I.
 P6 lindep: with a cut that drops nothing, 'dk' == 'std' to roundoff. With a cut that drops k eigenvalues and the
    SAME count at +-h, 'dk' matches FD at the floor and 'std' misses by the kept-dropped Loewner term (size not
    predicted; expected to grow as the cut approaches the kept spectrum). Near-noise cuts (1e-10 on a set whose
    smallest eigenvalues are +-1e-10 metric noise) may make the FD itself meaningless (count flips / 1/s_k blow-up).
ARTIFACT HYPOTHESES (a broken term would give ...):
 - no_metric (drop sum Wm dJ2) and no_aux (drop the J3 aux-centre derivative): each O(1e-2) (a J3/J2 aux piece is
   ~1e-2 on H2), but their SUM is O(fit residual) because E_fit is stationary w.r.t. aux parameters when the fit is
   exact (robust DF: error quadratic in the residual) -- so in the 'span' limit the two mutants miss by the SAME
   amount, and 'drop both' would be nearly invisible there.  sum F is blind to no_metric (each 2c term is
   translation invariant) but not exactly to no_aux (3c aux piece alone is not).
 - no_g0 (drop -c0 dS q): O(c0 |Y q| |dS|) ~ 1e-2.
 - g0_dense (the dense-AFT M-term c0(-N D + alpha sum D_s S D_s)): IDENTICAL to the right term when the fit is exact
   (sum_P C_P q_P = S when the fit reproduces charges), so BLIND in 'span' and small (fit-residual sized) otherwise.
 - no_lr3 (drop the reciprocal part of dJ3): O(1e-2).
 - a wrong aux-centre chain rule in 'span' (not coded as a mutant: no_aux bounds it).
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_gdf as GG
import pbc_grad_open as PGO
from pbc_gamma import Cell
from pbc_gdf import even_tempered, ferric_basis
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H = 1e-4
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))]
H3_ATOMS = H2_ATOMS + [("H", (1.6, 0.9, 0.7))]
MUTANTS = ["no_metric", "no_aux", "no_g0", "g0_dense", "no_lr3"]


def with_vm(ints, ex):
    d = dict(ints)
    if ex == "none":
        d["madelung"] = 0.0
    return d


def fd_energies(cell, aux_of, gcut, cases, ref, w=1.0, spherical=True, grid_of=None, lindeps=None):
    """E(+h), E(-h) for every component in `ref['comps']` and every case; one integral + GDF build per displacement.
    cases: list of (label, na, nb, xc, restricted, exxdiv, lindep); returns dict[(label, ex, lindep, A, x)] = FD."""
    out, counts = {}, {}
    for A, x in ref["comps"]:
        ev = {}
        for s in (+1, -1):
            t = time.time()
            cd = PGd.displaced(cell, A, x, s * H)
            aux = aux_of(cd)[0]
            ints = GG.integrals(cd, gcut, "ewald")
            gd = GG.gdf(cd, aux, w, spherical=spherical, lindep=0.0)
            sv = np.linalg.eigvalsh(gd["J2"])
            grid = grid_of(cd) if grid_of else None
            for label, na, nb, xc, rst, ex, ld in cases:
                gdl = dict(gd, B=GG.B_from(gd["J2"], gd["J3"], ld))
                counts[(ld, A, x, s)] = int((sv <= ld).sum())
                r0 = ref["scf"][(label, ex, ld)]
                e = GG.energy(cd, with_vm(ints, ex), gdl, na, nb, xc, grid if xc != "HF" else None, rst,
                              (r0["Da"], r0["Db"]))["e"]
                ev[(label, ex, ld, s)] = e
            print(f"  FD ({A},{x}) s={s:+d} ({time.time() - t:.0f}s)", flush=True)
        for label, na, nb, xc, rst, ex, ld in cases:
            out[(label, ex, ld, A, x)] = (ev[(label, ex, ld, 1)] - ev[(label, ex, ld, -1)]) / (2 * H)
    return out, counts


def reference(cell, aux, jac, gcut, cases, w=1.0, spherical=True, grid=None, metrics=("dk",), mutants=()):
    ints = GG.integrals(cell, gcut, "ewald")
    t = time.time()
    gd = GG.gdf(cell, aux, w, spherical=spherical, lindep=0.0)
    print(f"  GDF build {time.time() - t:.0f}s: naux {gd['info']['naux']}, metric min/max "
          f"{gd['info']['metric_min']:.3e}/{gd['info']['metric_max']:.3e}, ranges "
          + ", ".join(f"{k} {v:.2f}" for k, v in gd["rng"].items()), flush=True)
    res = dict(scf={}, grad={}, mut={}, diag={}, e={})
    prev = {}
    t = time.time()
    cache = GG.deriv_ints(cell, ints, gd, aux, w, verbose=True)
    print(f"  derivative integrals {time.time() - t:.0f}s", flush=True)
    for label, na, nb, xc, rst, ex, ld in cases:
        t = time.time()
        gdl = dict(gd, B=GG.B_from(gd["J2"], gd["J3"], ld))
        d = with_vm(ints, ex)
        gs = prev.get(label)
        r = GG.energy(cell, d, gdl, na, nb, xc, grid if xc != "HF" else None, rst, gs)
        prev.setdefault(label, (r["Da"], r["Db"]))
        res["scf"][(label, ex, ld)] = r
        res["e"][(label, ex, ld)] = r["e"]
        for m in metrics:
            g, parts, diag = GG.gamma_gdf_grad(cell, d, gdl, aux, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                                               lindep=ld, spherical=spherical, metric=m, aux_jac=jac,
                                               grid=grid if xc != "HF" else None, xc=xc, restricted=rst,
                                               cache=cache)
            res["grad"][(label, ex, ld, m)] = g
            res["diag"][(label, ex, ld)] = diag
            if m == "dk":
                res.setdefault("parts", {})[(label, ex, ld)] = parts
        for mut in mutants:
            GG._MUTANT = mut
            try:
                g, _, _ = GG.gamma_gdf_grad(cell, d, gdl, aux, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                                            lindep=ld, spherical=spherical, aux_jac=jac,
                                            grid=grid if xc != "HF" else None, xc=xc, restricted=rst,
                                            cache=cache)
            finally:
                GG._MUTANT = None
            res["mut"][(label, ex, ld, mut)] = g
        print(f"  ref {label:14s} {ex:5s} lindep {ld:.0e}: E {r['e']:.12f} it {r['it']} err {r['err']:.1e} "
              f"drop {res['diag'][(label, ex, ld)]['n_drop']} ({time.time() - t:.0f}s)", flush=True)
    res["ints"], res["gd"] = ints, gd
    return res


def report(ref, fd, comps, cases, metrics=("dk",), mutants=(), counts=None):
    for label, na, nb, xc, rst, ex, ld in cases:
        g = ref["grad"][(label, ex, ld, "dk")]
        row = [f"{label:14s} {ex:5s} lindep {ld:.0e}"]
        for m in metrics:
            dif = [ref["grad"][(label, ex, ld, m)][A, x] - fd[(label, ex, ld, A, x)] for A, x in comps]
            row.append(f"{m}: " + " ".join(f"{v:+.2e}" for v in dif))
        row.append(f"|sumF| {abs(g.sum(0)).max():.1e}")
        if counts is not None:
            row.append("drop(+h/-h) " + ",".join(f"{counts[(ld, A, x, 1)]}/{counts[(ld, A, x, -1)]}" for A, x in comps))
        print("  " + " | ".join(row))
        for mut in mutants:
            gm = ref["mut"][(label, ex, ld, mut)]
            dif = max(abs(gm[A, x] - fd[(label, ex, ld, A, x)]) for A, x in comps)
            print(f"      mutant {mut:10s} max|an-FD| {dif:.2e}   |sumF| {abs(gm.sum(0)).max():.1e}")


def atoms_aux(basis):
    return lambda c: (GG.make_auxmol(c, basis), None)


# ================================================================================================= cases
def main_h2():
    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    aux_of = atoms_aux(even_tempered(["H"], 1, 0.3, 2.5, 5))
    cases = [(lab, 1, 1, "HF", rst, ex, 0.0) for lab, rst in (("RHF", True), ("UHF(1,1)", False))
             for ex in ("none", "ewald")]
    ref = reference(cell, aux_of(cell)[0], None, 12.0, cases, spherical=True, mutants=MUTANTS)
    ref["comps"] = [(0, 0), (0, 2), (1, 1)]
    fd, _ = fd_energies(cell, aux_of, 12.0, cases, ref)
    report(ref, fd, ref["comps"], cases, mutants=MUTANTS)
    g = ref["grad"]
    print(f"  F(ewald)-F(none) RHF {abs(g[('RHF', 'ewald', 0.0, 'dk')] - g[('RHF', 'none', 0.0, 'dk')]).max():.1e}; "
          f"UHF(1,1)-RHF {abs(g[('UHF(1,1)', 'none', 0.0, 'dk')] - g[('RHF', 'none', 0.0, 'dk')]).max():.1e}")
    for k, v in ref["parts"][("RHF", "none", 0.0)].items():
        print(f"    part {k:10s} (0,z) {v[0, 2]:+.8f}")


def ghost_aux(cell, al=0.5):
    cen, jac = [], []
    for i, j in [(0, 0), (1, 1), (0, 1)]:
        for h in np.ndindex(2, 2, 2):
            cen.append(0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a))
            r = np.zeros(len(cell.R))
            r[i] += 0.5
            r[j] += 0.5
            jac.append(r)
    aux = gto.M(atom=[("X", c) for c in cen], basis={"X": [[0, [2 * al, 1.0]]]}, unit="B", cart=True, verbose=0)
    return aux, np.array(jac)


def main_span():
    """P5: exact aux span; GDF force vs the dense-AFT force of pbc_grad_open (same h, S, E_nn) and vs FD."""
    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, {"H": [[0, [0.5, 1.0]]]})
    aux, jac = ghost_aux(cell)
    cases = [(lab, 1, 1, "HF", rst, ex, 0.0) for lab, rst in (("RHF", True), ("UHF(1,1)", False))
             for ex in ("none", "ewald")]
    ref = reference(cell, aux, jac, 12.0, cases, spherical=False, mutants=MUTANTS)
    for label, na, nb, xc, rst, ex, ld in cases:
        rd = PGO.run_case(cell, na, nb, xc, with_vm(ref["ints"], ex), restricted=rst)
        g = ref["grad"][(label, ex, ld, "dk")]
        print(f"  {label:9s} {ex:5s}: E_gdf-E_dense {ref['e'][(label, ex, ld)] - rd['e']:+.2e}  "
              f"max|F_gdf-F_dense| {abs(g - rd['grad']).max():.1e}  |F| {abs(rd['grad']).max():.4f}")
        for mut in MUTANTS:
            gm = ref["mut"][(label, ex, ld, mut)]
            print(f"      mutant {mut:10s} max|F-F_dense| {abs(gm - rd['grad']).max():.2e}  |sumF| {abs(gm.sum(0)).max():.1e}")
    ref["comps"] = [(0, 0), (1, 2)]
    fd, _ = fd_energies(cell, lambda c: ghost_aux(c), 12.0, cases[:2], ref, spherical=False)
    report(ref, fd, ref["comps"], cases[:2])


def main_lindep():
    """P6: one H2/STO-3G cell, two aux sets, several absolute cuts; 'dk' vs 'std' vs FD, dropped counts at +-h."""
    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    for name, spec, cuts in [
        ("ET l<=1 a0.3 b2.5 n5 (40 aux, clean spectrum)", (1, 0.3, 2.5, 5), [0.0, 1e-4, 1.7e-3, 3e-3]),
        ("ET l<=1 a0.1 b1.7 n9 (72 aux, noise-level tail)", (1, 0.1, 1.7, 9), [1e-10, 1e-8, 1e-6, 1e-4]),
    ]:
        print(f"[lindep] {name}", flush=True)
        aux_of = atoms_aux(even_tempered(["H"], *spec))
        cases = [("RHF", 1, 1, "HF", True, "none", c) for c in cuts]
        ref = reference(cell, aux_of(cell)[0], None, 12.0, cases, spherical=True, metrics=("dk", "std"))
        s = np.linalg.eigvalsh(ref["gd"]["J2"])
        print("  spectrum (lowest 10):", np.array2string(s[:10], precision=3), flush=True)
        for c in cuts:
            d = ref["diag"][("RHF", "none", c)]
            gd, gs = ref["grad"][("RHF", "none", c, "dk")], ref["grad"][("RHF", "none", c, "std")]
            print(f"  cut {c:.1e}: drop {d['n_drop']}, s_kept_min {d['s_kept_min']:.3e}, s_drop_max {d['s_drop_max']:.3e},"
                  f" max|F_dk - F_std| {abs(gd - gs).max():.2e}, E {ref['e'][('RHF', 'none', c)]:.12f}")
        ref["comps"] = [(0, 2), (1, 1)]
        fd, counts = fd_energies(cell, aux_of, 12.0, cases, ref)
        report(ref, fd, ref["comps"], cases, metrics=("dk", "std"), counts=counts)


def main_hscan():
    """Is the noise-cut residual FD truncation (~h^2), FD roundoff (~1/h) or an analytic inconsistency (h-flat)?
    72-aux noise-tail set, cuts 1e-10 / 1e-8, component (1,1) and (0,2), h = 2.5e-5 .. 2e-4."""
    cell = Cell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    aux_of = atoms_aux(even_tempered(["H"], 1, 0.1, 1.7, 9))
    cuts = [1e-10, 1e-8]
    cases = [("RHF", 1, 1, "HF", True, "none", c) for c in cuts]
    ref = reference(cell, aux_of(cell)[0], None, 12.0, cases, spherical=True, metrics=("dk", "std"))
    hs = [2.5e-5, 5e-5, 1e-4, 2e-4]
    for A, x in [(1, 1), (0, 2)]:
        ev = {}
        for h in hs:
            for s in (+1, -1):
                cd = PGd.displaced(cell, A, x, s * h)
                ints = GG.integrals(cd, 12.0, "ewald")
                gd = GG.gdf(cd, aux_of(cd)[0], 1.0, spherical=True, lindep=0.0)
                for c in cuts:
                    r0 = ref["scf"][("RHF", "none", c)]
                    gdl = dict(gd, B=GG.B_from(gd["J2"], gd["J3"], c))
                    ev[(c, h, s)] = GG.energy(cd, with_vm(ints, "none"), gdl, 1, 1, "HF", None, True,
                                              (r0["Da"], r0["Db"]))["e"]
        for c in cuts:
            gdk, gst = ref["grad"][("RHF", "none", c, "dk")][A, x], ref["grad"][("RHF", "none", c, "std")][A, x]
            fds = [(ev[(c, h, 1)] - ev[(c, h, -1)]) / (2 * h) for h in hs]
            print(f"  cut {c:.0e} ({A},{x}): dk - FD(h) " + " ".join(f"{gdk - f:+.2e}" for f in fds)
                  + " | std - FD(h) " + " ".join(f"{gst - f:+.2e}" for f in fds)
                  + f" | h = {hs}", flush=True)


def main_h3():
    cell = Cell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
    aux_of = atoms_aux(ferric_basis("cc-pvdz-ri", ["H"]))
    grid_of = lambda c: PGO.GradGrid(c, 40, 50, D=8.0, scheme="ssf", deriv=2)  # noqa: E731
    grid = grid_of(cell)
    cases = [(lab, 2, 1, xc, False, ex, 1e-10) for lab, xc in (("UHF doublet", "HF"), ("UKS PBE0", "PBE0"))
             for ex in ("none", "ewald")]
    ref = reference(cell, aux_of(cell)[0], None, 12.0, cases, spherical=True, grid=grid, mutants=MUTANTS)
    ref["comps"] = [(0, 0), (2, 1)]
    fd, _ = fd_energies(cell, aux_of, 12.0, cases, ref,
                        grid_of=lambda c: PGO.GradGrid(c, 40, 50, D=8.0, scheme="ssf", deriv=1, want_dw=False))
    report(ref, fd, ref["comps"], cases, mutants=MUTANTS)
    g = ref["grad"]
    for lab in ("UHF doublet", "UKS PBE0"):
        print(f"  {lab}: F(ewald)-F(none) {abs(g[(lab, 'ewald', 1e-10, 'dk')] - g[(lab, 'none', 1e-10, 'dk')]).max():.1e}")


def main_tri():
    cell = Cell(TRI_A, TRI_MOVED, SP_BASIS)
    aux_of = atoms_aux(ferric_basis("cc-pvdz-ri", ["H"]))
    cases = [("RHF", 2, 2, "HF", True, "none", 1e-10), ("UHF triplet", 3, 1, "HF", False, "none", 1e-10),
             ("UHF triplet", 3, 1, "HF", False, "ewald", 1e-10)]
    ref = reference(cell, aux_of(cell)[0], None, 10.0, cases, spherical=True, mutants=MUTANTS)
    ref["comps"] = [(2, 1), (0, 2)]
    fd, _ = fd_energies(cell, aux_of, 10.0, cases, ref)
    report(ref, fd, ref["comps"], cases, mutants=MUTANTS)
    g = ref["grad"]
    print(f"  UHF triplet F(ewald)-F(none) "
          f"{abs(g[('UHF triplet', 'ewald', 1e-10, 'dk')] - g[('UHF triplet', 'none', 1e-10, 'dk')]).max():.1e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "h2"
    t0 = time.time()
    {"h2": main_h2, "span": main_span, "lindep": main_lindep, "hscan": main_hscan, "h3": main_h3, "tri": main_tri}[which]()
    print(f"[{which}] done in {time.time() - t0:.0f}s")
