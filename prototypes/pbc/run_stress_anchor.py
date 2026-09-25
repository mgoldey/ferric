"""Iteration 19 exactness anchors for the Gamma stress tensor (pbc_stress.py).  Usage:
    python3 run_stress_anchor.py {h2|tri|h3|sr|ks|ksu|gdf_h2|gdf_h3|gdf_tri|hscan|pv|rescaled|pulay} [--no-mut]

ANCHOR: analytic dE/d eps_ij vs central FD (h = 1e-4) of the prototype's OWN energy on strained cells (lattice AND atoms
strained, fixed fractional coordinates, fixed Miller-index / image-index sets), all 9 components, both exxdiv.
Antisymmetric part: analytic sigma - sigma^T and FD(ij) - FD(ji) (rotational invariance).

PREDICTIONS (written before the full runs; only an H2 RHF smoke run, a triclinic RHF-ewald smoke run on 3 components
and an H2 s-only SR-route smoke run on 4 components had been seen, all at 2e-12 .. 3e-8):
 P1 correct code: |analytic - FD| at the FD truncation level h^2 f'''/6 ~ 1e-9 .. 1e-8 for HF (dense AFT and SR route).
    Richardson / h-scan must shrink it as h^2 (case hscan).
 P2 rotational invariance: HF and uniform-grid KS energies are exactly rotation invariant with fixed index sets
    (G and image sets rotate with the lattice), so analytic |sigma - sigma^T| ~ 1e-12 and FD(ij) - FD(ji) ~ FD noise.
    Atom-centred KS grids are NOT (Lebedev offsets are fixed in the lab frame): antisymmetric part at grid-error size
    (~1e-4 .. 1e-3 Ha, like sum F without grid response); the FD anchor still holds for every component.
 P3 exxdiv: sigma(ewald) - sigma(none) = -(alpha N/2) dv_M/d eps exactly (E_M = -alpha v_M N/2 for idempotent D_s);
    the force-level S-term cancels as for forces.  Madelung stress is O(v_M) ~ 0.2 .. 0.7 Ha per unit strain.
 P4 pressure: -tr(sigma)/3 equals P = -dE/dOmega from FD under isotropic scaling (chain rule; a consistency check
    only, NOT independent of P1).
ARTIFACT HYPOTHESES (mutants; each must miss FD by >> the P1 floor; a mutant at the floor = blind anchor => redesign):
    no_pulay, ft_no_centres, g_unstrained, no_volume: O(0.1 .. 1)  (each is a whole term of an O(1) energy)
    no_ewald_lr O(0.1 .. 1); no_ewald_bg = pi Z^2/(2 Omega w^2) on the diagonal only (0.1 for H2 a=4);
    no_madelung (ewald only) O(v_M N/2); madelung_s (ewald only) O(v_M |dS|) ~ 1e-2 .. 1e-1;
    no_c0_volume (SR route) = E_c0 = 3 c0 for H2 (0.15) on the diagonal; no_sr_images O(0.1 .. 1);
    xc_no_weight, xc_no_ao, xc_point_fixed (KS): O(1e-2 .. 1); on the uniform grid xc_no_weight = E_xc on the diagonal.
    Off-diagonal components are blind to every pure-volume mutant (no_volume, no_ewald_bg, no_c0_volume) by construction.
RS-GDF (gdf_*; predictions written after ONE H2 ET-40 RHF-ewald smoke run on 4 components, 9e-10 .. 8.5e-9):
 P5 dk metric at the P1 floor with no eigenvalue cut and with a cut dropping the same count at +-h; 'std' off by the
    kept-dropped term when the dropped directions carry energy (Iteration 18: 4e-8 .. 1.5e-7 for forces).
 P6 the J2 G = 0 term (-c0 q q^T) is CONSTANT under atom motion (forces never see it) but c0 ~ 1/Omega makes it a
    strain term: mutant no_j2_g0_vol O(c0 q^T Wm q) ~ 0.03 per diagonal component; no_j3_g0_vol likewise O(0.1).
    no_aux_ft (aux FT treated as strain-free), no_metric, no_g0: O(1e-2 .. 1e-1).
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_stress as ST
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H = 1e-4
ALL9 = [(i, j) for i in range(3) for j in range(3)]
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))]
H3_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))]
S05 = {"H": gto.parse("H S\n  0.5  1.0\n")}
GC_SR = 2 * np.sqrt(np.log(1e14))  # pbc_gamma default LR sphere at w = 1
MUT_ALL = ["no_pulay", "ft_no_centres", "g_unstrained", "no_volume", "no_ewald_lr", "no_ewald_bg"]
MUT_EW = ["no_madelung", "madelung_s"]
MUT_SR = ["no_c0_volume", "no_sr_images"]
MUT_XC = ["xc_no_weight", "xc_no_ao", "xc_point_fixed"]


def spec(na, nb, xc, restricted, exxdiv, gcut, w=None, grid=None, rc=None):
    return dict(na=na, nb=nb, xc=xc, restricted=restricted, exxdiv=exxdiv, gcut=gcut, w=w, grid=grid, rc=rc or {})


def CASES():
    h2 = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    tri = ST.FixedCell(TRI_A, TRI_MOVED, SP_BASIS)
    h3 = ST.FixedCell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
    h2s = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, S05)
    rc = dict(rcut_bra=13.0)  # away from the |L| = 4 sqrt(k) shells so the bra-image count cannot change under strain
    ag = ("atom", 40, 50, 8.0)
    return {
        "h2": (h2, [("RHF none", spec(1, 1, "HF", True, "none", 12.0)), ("RHF ewald", spec(1, 1, "HF", True, "ewald", 12.0)),
                    ("UHF(1,1) ewald", spec(1, 1, "HF", False, "ewald", 12.0))]),
        "tri": (tri, [("RHF none", spec(2, 2, "HF", True, "none", 10.0)), ("RHF ewald", spec(2, 2, "HF", True, "ewald", 10.0)),
                      ("UHF(3,1) none", spec(3, 1, "HF", False, "none", 10.0)),
                      ("UHF(3,1) ewald", spec(3, 1, "HF", False, "ewald", 10.0))]),
        "tri_uhf": (tri, [("UHF(3,1) ewald", spec(3, 1, "HF", False, "ewald", 10.0))]),
        "h3": (h3, [("UHF(2,1) none", spec(2, 1, "HF", False, "none", 12.0)),
                    ("UHF(2,1) ewald", spec(2, 1, "HF", False, "ewald", 12.0))]),
        "sr": (h2s, [("RHF none w=1", spec(1, 1, "HF", True, "none", GC_SR, w=1.0, rc=rc)),
                     ("RHF ewald w=1", spec(1, 1, "HF", True, "ewald", GC_SR, w=1.0, rc=rc)),
                     ("UHF(1,1) ewald w=1", spec(1, 1, "HF", False, "ewald", GC_SR, w=1.0, rc=rc))]),
        # atom grid: h = 2.5e-5 (at 1e-4 a few points cross the hard |r - X_B| <= D mask: run_stress_anchor gridcheck)
        "ks": (h3, [("UKS LDA none", dict(spec(2, 1, "LDA,VWN", False, "none", 12.0, grid=ag), fd_h=2.5e-5)),
                    ("UKS PBE none", dict(spec(2, 1, "PBE", False, "none", 12.0, grid=ag), fd_h=2.5e-5)),
                    ("UKS PBE0 ewald", dict(spec(2, 1, "PBE0", False, "ewald", 12.0, grid=ag), fd_h=2.5e-5))]),
        "ksu": (h3, [("UKS PBE none, uniform 24", spec(2, 1, "PBE", False, "none", 12.0, grid=("uniform", 24))),
                     ("UKS PBE0 ewald, uniform 24", spec(2, 1, "PBE0", False, "ewald", 12.0, grid=("uniform", 24)))]),
    }


def resid(an, fd):
    return max(abs(an[k] - v) for k, v in fd.items())


def run_case(name, mutants=True):
    cell, runs = CASES()[name]
    for label, sp in runs:
        t = time.time()
        ints, _, r = ST.solve(cell, sp)
        an, parts = ST.analytic(cell, sp, ints, r)
        ta = time.time() - t
        fd = ST.fd_stress(cell, sp, ALL9, (r["Da"], r["Db"]), h=sp.get("fd_h", H))
        fdm = np.array([[fd[(i, j)] for j in range(3)] for i in range(3)])
        print(f"\n== {name} {label} (FD h {sp.get('fd_h', H):.1e}): E = {r['e']:.12f}  Omega = {cell.vol:.6f}  (analytic {ta:.0f}s, total "
              f"{time.time() - t:.0f}s, p_err {parts['p_err']:.1e})", flush=True)
        print("   analytic dE/deps:\n" + np.array2string(an, precision=10, suppress_small=False))
        print("   analytic - FD:\n" + np.array2string(an - fdm, precision=2))
        print(f"   max|an-FD| = {resid(an, fd):.2e}   |an - an^T| = {abs(an - an.T).max():.1e}   "
              f"|FD - FD^T| = {abs(fdm - fdm.T).max():.1e}   -tr(sigma)/3 = {-np.trace(an) / (3 * cell.vol):.10f}")
        print("   parts: " + ", ".join(f"{k} tr={np.trace(v):+.6f}" for k, v in parts.items() if np.ndim(v) == 2))
        if not mutants:
            continue
        muts = MUT_ALL + (MUT_EW if sp["exxdiv"] == "ewald" else []) + (MUT_SR if sp["w"] else []) + (
            MUT_XC if sp["grid"] is not None else [])
        if sp["grid"] is not None and sp["grid"][0] == "uniform":
            muts = [m for m in muts if m != "xc_point_fixed"]
        for m in muts:
            ST._MUTANT = m
            try:
                am, _ = ST.analytic(cell, sp, ints, r)
            finally:
                ST._MUTANT = None
            off = max(abs(am[i, j] - fd[(i, j)]) for i, j in ALL9 if i != j)
            print(f"    MUTANT {m:15s}: max|an-FD| = {resid(am, fd):.2e}  (off-diagonal only {off:.1e})", flush=True)


def hscan():
    """FD truncation check: analytic - FD at h = 5e-5, 1e-4, 2e-4 and the Richardson (5e-5, 1e-4) value."""
    for name, lab, comps in (("h2", "RHF ewald", [(2, 2), (0, 0)]), ("tri", "RHF ewald", [(0, 0), (1, 1)]),
                             ("ksu", "UKS PBE none, uniform 24", [(0, 0), (2, 2)])):
        cell, runs = CASES()[name]
        sp = dict(runs)[lab]
        ints, _, r = ST.solve(cell, sp)
        an, _ = ST.analytic(cell, sp, ints, r)
        g = (r["Da"], r["Db"])
        fds = {h: ST.fd_stress(cell, sp, comps, g, h=h) for h in (5e-5, 1e-4, 2e-4)}
        for c in comps:
            rich = (4 * fds[5e-5][c] - fds[1e-4][c]) / 3
            print(f"{name} {lab} {c}: an-FD at h=5e-5/1e-4/2e-4: " + " / ".join(
                f"{an[c] - fds[h][c]:+.2e}" for h in (5e-5, 1e-4, 2e-4)) + f"   Richardson {an[c] - rich:+.1e}", flush=True)


def gridcheck():
    """Atom-grid weight strain derivative vs central FD of the weights point by point (pid-matched), and the KS LDA
    h-scan: separates hard-mask jumps of E(eps) (FD noise) from a defect in the analytic weight term."""
    cell = CASES()["ks"][0]
    g = ST.StrainGrid(cell, "atom", n_rad=40, n_ang=50, D=8.0, ao=False)
    idx = {p: k for k, p in enumerate(g.pid)}
    for (i, j) in ((0, 0), (2, 2), (1, 2), (2, 1)):
        for h in (1e-5, 1e-4):
            ws = []
            for s in (1, -1):
                eps = np.zeros((3, 3))
                eps[i, j] = s * h
                gs = ST.StrainGrid(cell.strained(eps), "atom", n_rad=40, n_ang=50, D=8.0, ao=False)
                w = np.full(g.size, np.nan)
                w[[idx[p] for p in gs.pid]] = gs.weights
                ws.append(w)
            fdw = (ws[0] - ws[1]) / (2 * h)
            d = abs(fdw - g.dW[:, i, j])
            big = np.sort(d)[::-1]
            print(f"({i},{j}) h={h:.0e}: max|dW_an - dW_FD| {big[0]:.2e}; 2nd/5th/20th largest {big[1]:.1e} / "
                  f"{big[4]:.1e} / {big[19]:.1e}; n > 1e-6: {int((d > 1e-6).sum())} of {g.size}; max|dW| "
                  f"{abs(g.dW[:, i, j]).max():.2e}", flush=True)
    sp = dict(CASES()["ks"][1])["UKS LDA none"]
    ints, _, r = ST.solve(cell, sp)
    an, _ = ST.analytic(cell, sp, ints, r)
    gu = (r["Da"], r["Db"])
    for c in ((2, 2), (0, 0), (1, 2)):
        row = []
        for h in (2.5e-5, 5e-5, 1e-4, 2e-4):
            row.append(an[c] - ST.fd_stress(cell, sp, [c], gu, h=h)[c])
        print(f"KS LDA atom grid {c}: an - FD at h = 2.5e-5/5e-5/1e-4/2e-4: " + " / ".join(f"{x:+.2e}" for x in row),
              flush=True)


def hscan2():
    """tri UHF(3,1) (2,2): the 2.1e-7 residual at h = 1e-4 -- truncation (h^2) or defect?"""
    cell, runs = CASES()["tri"]
    sp = dict(runs)["UHF(3,1) none"]
    ints, _, r = ST.solve(cell, sp)
    an, _ = ST.analytic(cell, sp, ints, r)
    g = (r["Da"], r["Db"])
    fds = {h: ST.fd_stress(cell, sp, [(2, 2)], g, h=h)[(2, 2)] for h in (2.5e-5, 5e-5, 1e-4, 2e-4)}
    print("tri UHF(3,1) none (2,2): an - FD at h = 2.5e-5/5e-5/1e-4/2e-4: " + " / ".join(
        f"{an[2, 2] - v:+.2e}" for v in fds.values()) + f"   Richardson(5e-5,1e-4) "
        f"{an[2, 2] - (4 * fds[5e-5] - fds[1e-4]) / 3:+.1e}", flush=True)


def hscan3():
    """H3 UHF(2,1) ewald (0,0): dense AFT vs RS-GDF residual at h = 1e-4 was -5.4e-8 / -5.2e-8 -- truncation?"""
    cell, runs = CASES()["h3"]
    sp = dict(runs)["UHF(2,1) ewald"]
    ints, _, r = ST.solve(cell, sp)
    an, _ = ST.analytic(cell, sp, ints, r)
    g = (r["Da"], r["Db"])
    f = {h: ST.fd_stress(cell, sp, [(0, 0)], g, h=h)[(0, 0)] for h in (5e-5, 1e-4)}
    print(f"dense h3 UHF ewald (0,0): an-FD h=5e-5/1e-4 {an[0, 0] - f[5e-5]:+.2e} / {an[0, 0] - f[1e-4]:+.2e}  "
          f"Richardson {an[0, 0] - (4 * f[5e-5] - f[1e-4]) / 3:+.1e}", flush=True)
    cell, runs, _ = gdf_cases()["gdf_h3"][0], gdf_cases()["gdf_h3"][1], None
    lab, sp, _ = runs[0]
    ints, gd, aux, r = ST.gdf_solve(cell, sp)
    an, _ = ST.gdf_analytic(cell, sp, ints, gd, aux, r)
    g = (r["Da"], r["Db"])
    f = {h: ST.gdf_fd_stress(cell, sp, [(0, 0)], g, h=h)[(0, 0)] for h in (5e-5, 1e-4)}
    print(f"gdf h3 {lab} (0,0): an-FD h=5e-5/1e-4 {an[0, 0] - f[5e-5]:+.2e} / {an[0, 0] - f[1e-4]:+.2e}  "
          f"Richardson {an[0, 0] - (4 * f[5e-5] - f[1e-4]) / 3:+.1e}", flush=True)


def pv():
    """Pressure: -tr(sigma)/3 vs -dE/dOmega from FD under isotropic scaling a -> (1+e) a (h = 1e-4)."""
    for name, lab in (("h2", "RHF ewald"), ("tri", "UHF(3,1) ewald"), ("sr", "RHF ewald w=1")):
        cell, runs = CASES()[name]
        sp = dict(runs)[lab]
        ints, _, r = ST.solve(cell, sp)
        an, _ = ST.analytic(cell, sp, ints, r)
        e = [ST.solve(cell.strained(s * H * np.eye(3)), sp, (r["Da"], r["Db"]))[2]["e"] for s in (1, -1)]
        dEde = (e[0] - e[1]) / (2 * H)  # dOmega/de = 3 Omega
        print(f"{name} {lab}: P analytic -tr(sigma)/3 = {-np.trace(an) / (3 * cell.vol):+.12f}  "
              f"P FD -dE/dOmega = {-dEde / (3 * cell.vol):+.12f}  diff {(-np.trace(an) + dEde) / (3 * cell.vol):.1e}",
              flush=True)


def rescaled():
    """The re-selected-sphere mistake: E(e) under isotropic scaling with FIXED vs RE-SELECTED G / image sets."""
    for name, lab in (("h2", "RHF none"), ("tri", "RHF none")):
        cell, runs = CASES()[name]
        sp = dict(runs)[lab]
        _, _, r = ST.solve(cell, sp)
        g = (r["Da"], r["Db"])
        es = np.linspace(-0.012, 0.012, 25)
        diffs = []
        for e in es:
            eps = e * np.eye(3)
            ef = ST.solve(cell.strained(eps), sp, g)[2]["e"]
            er = ST.solve(ST.rescaled_strain_cell(cell, eps), sp, g)[2]["e"]
            nf = len(cell.strained(eps).gvectors(sp["gcut"]))
            nr = len(ST.rescaled_strain_cell(cell, eps).gvectors(sp["gcut"]))
            diffs.append(er - ef)
            print(f"{name} e={e:+.4f}: E_fixed {ef:.12f}  E_reselected - E_fixed {er - ef:+.3e}  nG {nf} vs {nr}",
                  flush=True)
        d = np.array(diffs)
        print(f"{name}: max |E_reselected - E_fixed| over |e| <= 0.012: {abs(d).max():.2e}; largest step between "
              f"neighbours {abs(np.diff(d)).max():.2e}")


def pulay():
    """Truncation ('plane-wave Pulay') stress: analytic sigma of the fixed-index energy vs gcut (H2 RHF none)."""
    cell0 = CASES()["h2"][0]
    for gc in (8.0, 12.0, 16.0, 20.0, 24.0, 29.7):
        cell = ST.FixedCell(cell0.a, cell0.atoms, cell0.basis)
        sp = spec(1, 1, "HF", True, "none", gc)
        ints, _, r = ST.solve(cell, sp)
        an, _ = ST.analytic(cell, sp, ints, r)
        s = an / cell.vol
        print(f"gcut {gc:5.1f} nG {len(ints['G']):6d}: E {r['e']:.12f}  sigma_xx {s[0, 0]:+.10f}  sigma_zz "
              f"{s[2, 2]:+.10f}  sigma_yz {s[1, 2]:+.10f}", flush=True)


MUT_GDF = ["no_metric", "no_aux_ft", "no_aux_ft3", "no_j2_g0_vol", "no_j3_g0_vol", "no_g0", "no_sr_images", "g_unstrained",
           "no_volume", "no_pulay", "ft_no_centres"]


def gdf_cases():
    from pbc_gdf import even_tempered

    et40 = even_tempered(["H"], 1, 0.3, 2.5, 5)
    h2 = ST.FixedCell(np.eye(3) * 4.0, H2_ATOMS, "sto-3g")
    h3 = ST.FixedCell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
    tri = ST.FixedCell(TRI_A, TRI_MOVED, SP_BASIS)
    g = dict(w=1.0, lindep=0.0, spherical=True)
    return {
        "gdf_h2": (h2, [("RHF none ET40", dict(spec(1, 1, "HF", True, "none", 12.0), aux=et40, **g), ALL9),
                        ("RHF ewald ET40", dict(spec(1, 1, "HF", True, "ewald", 12.0), aux=et40, **g), ALL9),
                        ("RHF none ET40 cut 3e-3", dict(spec(1, 1, "HF", True, "none", 12.0), aux=et40,
                                                        **dict(g, lindep=3e-3)), [(0, 0), (2, 2), (1, 2)])]),
        "gdf_h3": (h3, [("UHF(2,1) ewald cc-pvdz-ri", dict(spec(2, 1, "HF", False, "ewald", 12.0), aux="cc-pvdz-ri", **g),
                         ALL9),
                        ("UKS PBE0 ewald cc-pvdz-ri, atom grid", dict(spec(2, 1, "PBE0", False, "ewald", 12.0,
                                                                            grid=("atom", 40, 50, 8.0)),
                                                                       aux="cc-pvdz-ri", fd_h=2.5e-5, **g),
                         [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)])]),
        "gdf_h3ks": (h3, [("UKS PBE0 ewald cc-pvdz-ri, atom grid", dict(spec(2, 1, "PBE0", False, "ewald", 12.0,
                                                                              grid=("atom", 40, 50, 8.0)),
                                                                         aux="cc-pvdz-ri", fd_h=2.5e-5, **g),
                           [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (1, 2), (0, 2)])]),
        "gdf_tri": (tri, [("UHF(3,1) ewald cc-pvdz-ri", dict(spec(3, 1, "HF", False, "ewald", 10.0), aux="cc-pvdz-ri", **g),
                           [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)])]),
    }


def run_gdf(name, mutants=True):
    cell, runs = gdf_cases()[name]
    for label, sp, comps in runs:
        t = time.time()
        ints, gd, aux, r = ST.gdf_solve(cell, sp)
        an, parts = ST.gdf_analytic(cell, sp, ints, gd, aux, r)
        ta = time.time() - t
        guess = (r["Da"], r["Db"])
        fd = ST.gdf_fd_stress(cell, sp, comps, guess, h=sp.get("fd_h", H))
        kept = {s: ST.gdf_solve(cell.strained(s * 1e-4 * np.eye(3)), sp, guess)[1]["info"]["naux_kept"]
                for s in (1, -1)} if sp["lindep"] > 0 else {}
        print(f"\n== {name} {label}: E = {r['e']:.12f}  naux {gd['info']['naux']} kept {gd['info']['naux_kept']} "
              f"(at +-1e-4 isotropic: {kept})  (analytic {ta:.0f}s, total {time.time() - t:.0f}s, p_err "
              f"{parts['p_err']:.1e}, x_err {parts['x_err']:.1e})", flush=True)
        print("   analytic dE/deps:\n" + np.array2string(an, precision=10))
        print("   analytic - FD: " + ", ".join(f"{c}: {an[c] - v:+.2e}" for c, v in fd.items()))
        print(f"   max|an-FD| = {resid(an, fd):.2e}   |an - an^T| = {abs(an - an.T).max():.1e}")
        print("   parts: " + ", ".join(f"{k} tr={np.trace(v):+.6f}" for k, v in parts.items() if np.ndim(v) == 2))
        if sp["lindep"] > 0:
            ast, _ = ST.gdf_analytic(cell, sp, ints, gd, aux, r, metric="std")
            print(f"   metric 'std' (kept block only): max|an-FD| = {resid(ast, fd):.2e}; |dk - std| = "
                  f"{abs(an - ast).max():.1e}; diag {parts['diag']}")
        if not mutants or sp["lindep"] > 0:
            continue
        for m in MUT_GDF:
            ST._MUTANT = m
            try:
                am, _ = ST.gdf_analytic(cell, sp, ints, gd, aux, r)
            finally:
                ST._MUTANT = None
            print(f"    MUTANT {m:15s}: max|an-FD| = {resid(am, fd):.2e}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what.startswith("gdf"):
        run_gdf(what, mutants="--no-mut" not in sys.argv)
    elif what in ("hscan", "hscan2", "hscan3", "pv", "rescaled", "pulay", "gridcheck"):
        globals()[what]()
    else:
        run_case(what, mutants="--no-mut" not in sys.argv)
