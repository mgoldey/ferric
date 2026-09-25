"""Iteration 20 exactness anchors for Gamma ROHF / ROKS forces (pbc_grad_ro.py; no PySCF pbc anywhere).
Usage: python3 run_grad_ro_anchor.py {h3|h4|tri|closed|h3sr|gdf|hscan_h4|hscan_tri}

Every case: analytic force vs central FD (h = 1e-4) of the prototype's OWN Gamma ROHF/ROKS energy. The SSF grid is
REBUILT at every displaced geometry (FD_move, so grid response is included). Pure-AFT dense J/K; exxdiv none and ewald
come from one integral build, and ewald is seeded from the converged none MOs (staged). Every displaced SCF is seeded
from the REFERENCE MOs (Loewdin-orthonormalised in the new metric), is forced through at least one diagonalisation, and
stops at max|ROHF orbital gradient| < 1e-11. Its state is checked against the reference: the closed and open subspace
overlaps ||C_blk^T S C_ref,blk||^2 must equal (nd, no) to O(h), and the spin margins must keep their sign.

PREDICTIONS (written before running anything but one H3 ROHF smoke component):
 P1 correct code (W_ro = sym(D_a F_a D_a + D_a F_b D_b), polarized XC, per-spin Madelung/exchange): |analytic - FD_move|
    at the FD floor of Iterations 16-17 (~1e-9..4e-9 on H3/H4 STO-3G at h = 1e-4). On the tri s+p triplet the floor is
    the large-f''' truncation of Iteration 17 (~1e-8..3e-7 at h = 1e-4), so Richardson over (1e-4, 5e-5) there.
 P2 |sum F| ~ 1e-15..1e-14 (the grid response is included).
 P3 F(ewald) == F(none) to SCF precision (<= 1e-10). D_s S D_s = D_s per spin for ROHF too.
 P4 closed shell (no = 0): ROHF == RHF, ROKS == RKS forces to ~1e-12.
 P5 W_uhf = D_a F_a D_a + D_b F_b D_b is an IDENTITY at convergence. W_ro - W_uhf = 1/2 sym(C_o (f_b)_oc C_c^T), and
    (f_b)_co is the SCF residual (< 1e-11), so |F(W_uhf) - F(W_ro)| <~ 1e-11. It is reported but cannot fail. The task
    brief expected it to differ; the derivation says it does not, and this run tests the derivation.
 P6 ROHF vs UHF (UKS) force at the same geometry: different states, so different forces. Size not predicted beyond
    "nonzero, far above the floor". It should grow with spin contamination (E_RO - E_U).
ARTIFACT HYPOTHESES (if the code is broken I expect ...), each with its W algebra printed so an identity is visible:
 - w_feff_eps (W from the Roothaan F_eff eigenpairs): dW = C_o 1/2(f_b - f_a)_oo C_o^T - [C_c (f_a)_co C_o^T + h.c.],
   nonzero (H3 smoke: |(f_a - f_b)_oo| = 0.25, |(f_a)_co| = 0.068), so the miss is O(1e-2);
 - w_no_co: proportional to |(f_a)_co| (H3 0.068, H4 twoH2 0.024): O(1e-3..1e-2);
 - w_spin_sum (1/2 D Fbar D): O(1e-2);
 - madelung_total: visible only with ewald AND alpha > 0 (ROHF, PBE0), O(1e-2..1e-1); blind for none or LDA/PBE;
 - exch_total: alpha > 0 only, O(1e-2);
 - xc_rks_form (ROKS through the RKS XC path): O(1e-2..1e-1) on every open-shell KS case;
 - a wrong displaced state (hole state / swapped open orbital): an O(1e-3..1) FD jump. The continuity columns catch
   it before the difference is read as a force error.
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_dft as pd
import pbc_grad as PGd
import pbc_grad_open as O
import pbc_grad_ro as R
from pbc_gamma import Cell
from test_prototype import SP_BASIS, TRI_A, TRI_ATOMS

H = 1e-4
H3 = dict(a=np.eye(3) * 4.5, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))],
          basis="sto-3g", gcut=12.0, D=8.0, comps=[(0, 0), (2, 1), (1, 2)], nd=1, no=1)
H4 = dict(a=np.eye(3) * 5.0, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (2.6, 2.4, 2.2)),
                                    ("H", (2.7, 2.35, 3.65))],
          basis="sto-3g", gcut=12.0, D=8.0, comps=[(0, 0), (2, 1), (3, 2)], nd=1, no=2)
TRI = dict(a=TRI_A, atoms=TRI_ATOMS, basis=SP_BASIS, gcut=10.0, D=8.0, comps=[(2, 1), (0, 2)], nd=1, no=2)
H3_SR = dict(a=np.eye(3) * 4.5, atoms=H3["atoms"], basis={"H": gto.parse("H S\n  0.5  1.0\n")}, gcut=None, D=8.0,
             comps=[(0, 0), (2, 1)], nd=1, no=1)

# name: (system, [(label, xc, level_shift)])
SYSTEMS = {
    "h3": (H3, [("ROHF", "HF", None), ("ROKS LDA", "LDA,VWN", None), ("ROKS PBE", "PBE", None),
                ("ROKS PBE0", "PBE0", None)]),
    "h4": (H4, [("ROHF", "HF", None), ("ROKS LDA", "LDA,VWN", None), ("ROKS PBE", "PBE", None),
                ("ROKS PBE0", "PBE0", None)]),
    "tri": (TRI, [("ROKS LDA", "LDA,VWN", 0.05), ("ROKS PBE0", "PBE0", 0.05)]),
}
MUTS_HF = ["w_uhf", "w_feff_eps", "w_no_co", "w_spin_sum", "madelung_total", "exch_total"]
MUTS_XC = MUTS_HF + ["xc_rks_form"]


def with_vm(ints, ex):
    d = dict(ints)
    if ex == "none":
        d["madelung"] = 0.0
    return d


def grad_with(mut, cell, ints, r, w, grid, xc):
    R._MUTANT = mut
    try:
        g, _, diag = R.gamma_ro_grad(cell, ints, r, w=w, grid=grid, xc=xc)
    finally:
        R._MUTANT = None
    return g, diag


def main(which):
    if which == "closed":
        return main_closed()
    if which == "h3sr":
        return main_sr()
    if which == "gdf":
        return main_gdf()
    if which.startswith("hscan_"):
        return main_hscan(which[len("hscan_"):])
    sysd, cases = SYSTEMS[which]
    nd, no = sysd["nd"], sysd["no"]
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    t = time.time()
    ints = O.integrals(cell, None, "ewald", gcut=sysd["gcut"])
    grid = O.GradGrid(cell, 40, 50, D=sysd["D"], scheme="ssf", deriv=2)
    print(f"[{which}] nao {cell.mol.nao}, (nd, no) = ({nd}, {no}), v_M {ints['madelung']:.10f}, npts {grid.size}, "
          f"build {time.time() - t:.0f}s", flush=True)
    jk = pd.dense_jk(ints["I"])
    ref, uref = {}, {}
    for label, xc, ls in cases:
        for ex in ("none", "ewald"):
            t = time.time()
            C0 = None if ex == "none" else ref[(label, "none")]["scf"]["C"]
            r = R.run_case(cell, nd, no, xc, with_vm(ints, ex), grid=grid, C0=C0, level_shift=ls)
            ref[(label, ex)] = r
            sc = r["scf"]
            # UHF / UKS at the same geometry (P6), seeded from the ROHF densities
            u = O.scf(ints["S"], ints["h"], jk, ints["enn"], nd + no, nd, grid, xc, with_vm(ints, ex)["madelung"],
                      guess=(sc["Da"], sc["Db"]))
            gu, _ = O.gamma_grad(cell, with_vm(ints, ex), u["Da"], u["Db"], u["Fa"], u["Fb"], u["hyb"],
                                 grid=None if xc == "HF" else grid, xc=xc)
            uref[(label, ex)] = (u["e"], gu)
            d = r["diag"]
            print(f"  ref {label:10s} {ex:5s} E {r['e']:.12f} it {sc['it']} err {sc['err']:.1e} gaps "
                  f"({sc['gaps'][0]:+.4f}, {sc['gaps'][1]:+.4f}) |(fb)co| {d['fb_co']:.1e} |(fa)co| {d['fa_co']:.4f} "
                  f"|(fa-fb)oo| {d['fab_oo']:.4f} |W_ro-W_uhf| {d['dW_ro_uhf']:.1e}  UHF E {u['e']:.12f} "
                  f"({time.time() - t:.0f}s)", flush=True)
    hs = (H, H / 2) if which == "tri" else (H,)
    fd, cont = {}, {}
    for A, x in sysd["comps"]:
        for h in hs:
            ev = {}
            for s in (+1, -1):
                t = time.time()
                cd = PGd.displaced(cell, A, x, s * h)
                ints_d = O.integrals(cd, None, "ewald", gcut=sysd["gcut"])
                grid_d = O.GradGrid(cd, 40, 50, D=sysd["D"], scheme="ssf", deriv=1, want_dw=False)
                for label, xc, ls in cases:
                    for ex in ("none", "ewald"):
                        r0 = ref[(label, ex)]["scf"]
                        rd = R.energy_only(cd, nd, no, xc, with_vm(ints_d, ex), grid_d, r0["C"], level_shift=ls)
                        ev[(label, ex, s)] = rd["e"]
                        oc, oo = R.occ_overlap(rd["C"], r0["C"], ints_d["S"], nd, no)
                        cont.setdefault((label, ex), []).append((oc - nd, oo - no, min(rd["gaps"]), rd["it"]))
                print(f"  FD ({A},{x}) h={h:.0e} s={s:+d} ({time.time() - t:.0f}s)", flush=True)
            for label, xc, ls in cases:
                for ex in ("none", "ewald"):
                    fd[(label, ex, A, x, h)] = (ev[(label, ex, 1)] - ev[(label, ex, -1)]) / (2 * h)

    def fdbest(label, ex, A, x):
        if len(hs) == 1:
            return fd[(label, ex, A, x, H)]
        return (4 * fd[(label, ex, A, x, H / 2)] - fd[(label, ex, A, x, H)]) / 3  # Richardson

    print(f"\n[{which}] RESULTS (comps {sysd['comps']}; FD {'Richardson(1e-4, 5e-5)' if len(hs) > 1 else 'h=1e-4'})")
    print("| case | exxdiv | an - FD per comp | max | an - FD(h=1e-4) max | |sumF| | continuity max|d overlap| | "
          "min margin at +-h | it at +-h |")
    for label, xc, ls in cases:
        for ex in ("none", "ewald"):
            g = ref[(label, ex)]["grad"]
            dif = [g[A, x] - fdbest(label, ex, A, x) for A, x in sysd["comps"]]
            d1 = max(abs(g[A, x] - fd[(label, ex, A, x, H)]) for A, x in sysd["comps"])
            c = np.array(cont[(label, ex)])
            print(f"| {label} | {ex} | " + " ".join(f"{v:+.2e}" for v in dif) + f" | {max(map(abs, dif)):.1e} | "
                  f"{d1:.1e} | {abs(g.sum(0)).max():.1e} | {abs(c[:, :2]).max():.1e} | {c[:, 2].min():+.4f} | "
                  f"{int(c[:, 3].min())}-{int(c[:, 3].max())} |", flush=True)
    print("\nIdentities / comparisons:")
    for label, xc, ls in cases:
        gn, ge = ref[(label, "none")]["grad"], ref[(label, "ewald")]["grad"]
        de = ref[(label, "ewald")]["e"] - ref[(label, "none")]["e"]
        hyb = ref[(label, "none")]["scf"]["hyb"]
        print(f"  {label:10s} F(ewald)-F(none) {abs(ge - gn).max():.1e}; E(ewald)-E(none) {de:+.10f} "
              f"(pred -alpha v_M N/2 = {-hyb * ints['madelung'] * (2 * nd + no) / 2:+.10f})")
        for ex in ("none", "ewald"):
            eu, gu = uref[(label, ex)]
            print(f"      [{ex}] ROHF - UHF-type: dE {ref[(label, ex)]['e'] - eu:+.3e}, max|dF| "
                  f"{abs(ref[(label, ex)]['grad'] - gu).max():.3e}, |F_RO| {abs(ref[(label, ex)]['grad']).max():.3f}")
    print("\nMUTANTS: max|an(mut) - FD| over comps; algebra = max|F(mut) - F(correct)| (analytic, same state); "
          "|W_mut - W_ro| where W changes")
    for label, xc, ls in cases:
        muts = MUTS_HF if xc == "HF" else MUTS_XC
        for ex in ("none", "ewald"):
            r = ref[(label, ex)]
            row = []
            for m in muts:
                gm, dm = grad_with(m, cell, with_vm(ints, ex), r["scf"], None, grid, xc)
                d = max(abs(gm[A, x] - fdbest(label, ex, A, x)) for A, x in sysd["comps"])
                alg = abs(gm - r["grad"]).max()
                wtxt = f" dW {dm['dW_used_ro']:.1e}" if m.startswith("w_") else ""
                row.append(f"{m} {d:.1e} (alg {alg:.1e}{wtxt}; sF {abs(gm.sum(0)).max():.0e})")
            print(f"  {label:10s} {ex:5s} " + " | ".join(row), flush=True)


def main_hscan(which):
    """Separate FD truncation (~h^2), grid-energy discontinuity (h-erratic, absent with a FROZEN grid) and an analytic
    error (h-flat, present with BOTH grids).  ROKS LDA and the Iteration-17 UKS LDA CONTROL on the same cell, grid
    and displacements, exxdiv none.  an(resp) vs FD_move and an(noresp) vs FD_frozen at h = 2.5e-5 .. 2e-4."""
    sysd, _ = SYSTEMS[which]
    nd, no = sysd["nd"], sysd["no"]
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    ints = O.integrals(cell, None, "none", gcut=sysd["gcut"])
    grid = O.GradGrid(cell, 40, 50, D=sysd["D"], scheme="ssf", deriv=2)
    xc = "LDA,VWN"
    ro = R.run_case(cell, nd, no, xc, ints, grid=grid, level_shift=0.05)
    u = O.run_case(cell, nd + no, nd, xc, ints, grid=grid, guess=(ro["scf"]["Da"], ro["scf"]["Db"]))
    print(f"[hscan {which}] ROKS E {ro['e']:.12f}  UKS E {u['e']:.12f}", flush=True)

    def noresp(r):
        return r["grad"] - r["parts"].get("xc_point", 0) - r["parts"].get("xc_weight", 0)

    hs = [2.5e-5, 5e-5, 1e-4, 2e-4]
    for A, x in sysd["comps"]:
        ev = {}
        for h in hs:
            for s in (1, -1):
                cd = PGd.displaced(cell, A, x, s * h)
                ints_d = O.integrals(cd, None, "none", gcut=sysd["gcut"])
                gm = O.GradGrid(cd, 40, 50, D=sysd["D"], scheme="ssf", deriv=1, want_dw=False)
                gf = O.GradGrid.frozen(grid, cd, deriv=1)
                for gname, gg in (("move", gm), ("frozen", gf)):
                    ev[("RO", gname, h, s)] = R.energy_only(cd, nd, no, xc, ints_d, gg, ro["scf"]["C"], level_shift=0.05)["e"]
                    ev[("U", gname, h, s)] = O.energy_only(cd, nd + no, nd, xc, ints_d, gg, False,
                                                           (u["scf"]["Da"], u["scf"]["Db"]))
        for lab, r in (("ROKS", ro), ("UKS", u)):
            k = "RO" if lab == "ROKS" else "U"
            fm = [(ev[(k, "move", h, 1)] - ev[(k, "move", h, -1)]) / (2 * h) for h in hs]
            ff = [(ev[(k, "frozen", h, 1)] - ev[(k, "frozen", h, -1)]) / (2 * h) for h in hs]
            print(f"  ({A},{x}) {lab:4s} an(resp)-FD_move " + " ".join(f"{r['grad'][A, x] - f:+.2e}" for f in fm)
                  + " | an(noresp)-FD_frozen " + " ".join(f"{noresp(r)[A, x] - f:+.2e}" for f in ff)
                  + f"   (h = {hs})", flush=True)


def main_closed():
    """P4: no open shell. ROHF(nd=1, no=0) vs pbc_grad_open RHF, ROKS PBE / PBE0 vs RKS, H2 a=4 STO-3G, none/ewald."""
    cell = Cell(np.eye(3) * 4.0, [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))], "sto-3g")
    ints = O.integrals(cell, None, "ewald", gcut=12.0)
    grid = O.GradGrid(cell, 40, 50, D=8.0, scheme="ssf", deriv=2)
    for xc in ("HF", "PBE", "PBE0"):
        for ex in ("none", "ewald"):
            r = R.run_case(cell, 1, 0, xc, with_vm(ints, ex), grid=grid)
            rr = O.run_case(cell, 1, 1, xc, with_vm(ints, ex), grid=grid, restricted=True)
            print(f"  closed {xc:5s} {ex:5s}: E_RO - E_R {r['e'] - rr['e']:+.1e}, max|F_RO - F_R| "
                  f"{abs(r['grad'] - rr['grad']).max():.1e}, |W_ro - W_uhf| {r['diag']['dW_ro_uhf']:.1e}", flush=True)


def main_sr():
    """ROHF doublet on the Ewald-split route (w = 1; SR 3c V_ne + SR 4c ERI with per-spin Gam; compact s basis)."""
    sysd = H3_SR
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    ints = O.integrals(cell, 1.0, "ewald")
    C0 = None
    for ex in ("none", "ewald"):
        t = time.time()
        r = R.run_case(cell, 1, 1, "HF", with_vm(ints, ex), C0=C0, w=1.0)
        C0 = r["scf"]["C"]
        fd = {}
        for A, x in sysd["comps"]:
            e = []
            for s in (1, -1):
                cd = PGd.displaced(cell, A, x, s * H)
                e.append(R.energy_only(cd, 1, 1, "HF", with_vm(O.integrals(cd, 1.0, "ewald"), ex), None, C0)["e"])
            fd[(A, x)] = (e[0] - e[1]) / (2 * H)
        d = max(abs(r["grad"][k] - v) for k, v in fd.items())
        print(f"  ROHF doublet SR route {ex:5s}: E {r['e']:.12f} max|an-FD| {d:.1e} |sumF| "
              f"{abs(r['grad'].sum(0)).max():.1e} ({time.time() - t:.0f}s)", flush=True)
        for m in ("w_uhf", "w_feff_eps", "w_no_co", "exch_total") + (("madelung_total",) if ex == "ewald" else ()):
            R._MUTANT = m
            try:
                g, _, _ = R.gamma_ro_grad(cell, with_vm(ints, ex), r["scf"], w=1.0)
            finally:
                R._MUTANT = None
            print(f"      mutant {m:14s} max|an-FD| {max(abs(g[k] - v) for k, v in fd.items()):.1e}")


def main_gdf():
    """RS-GDF composition: H3 ROHF and ROKS PBE0 (cc-pvdz-ri, lindep 1e-10, grid (40, 50)), FD with the GDF rebuilt."""
    import pbc_grad_gdf as GG
    from pbc_gdf import ferric_basis, jk_from_B

    sysd = H3
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    aux_basis = ferric_basis("cc-pvdz-ri", ["H"])
    ld = 1e-10

    def setup(c, dw):
        ints = GG.integrals(c, sysd["gcut"], "ewald")
        aux = GG.make_auxmol(c, aux_basis)
        gd = GG.gdf(c, aux, 1.0, spherical=True, lindep=0.0)
        gd = dict(gd, B=GG.B_from(gd["J2"], gd["J3"], ld))
        grid = O.GradGrid(c, 40, 50, D=sysd["D"], scheme="ssf", deriv=2 if dw else 1, want_dw=dw)
        return ints, aux, gd, grid

    t = time.time()
    ints, aux, gd, grid = setup(cell, True)
    cache = {}
    print(f"[gdf] setup {time.time() - t:.0f}s naux {gd['info']['naux']}", flush=True)
    cases = [("ROHF", "HF"), ("ROKS PBE0", "PBE0")]
    ref = {}
    for label, xc in cases:
        C0 = None
        for ex in ("none", "ewald"):
            d = with_vm(ints, ex)
            r = R.rohf_scf(d["S"], d["h"], jk_from_B(gd["B"]), d["enn"], 1, 1, grid, xc, d["madelung"], C0=C0)
            C0 = r["C"]
            g, parts, diag = R.gamma_ro_grad_gdf(cell, d, gd, aux, r, lindep=ld, grid=grid, xc=xc, cache=cache)
            muts = {}
            for m in ("w_uhf", "w_feff_eps", "w_no_co"):
                R._MUTANT = m
                try:
                    muts[m] = R.gamma_ro_grad_gdf(cell, d, gd, aux, r, lindep=ld, grid=grid, xc=xc, cache=cache)[0]
                finally:
                    R._MUTANT = None
            ref[(label, ex)] = (r, g, muts)
            print(f"  ref {label} {ex}: E {r['e']:.12f} it {r['it']} drop {diag['n_drop']} ({time.time() - t:.0f}s)",
                  flush=True)
    fd = {}
    for A, x in sysd["comps"][:2]:
        ev = {}
        for s in (1, -1):
            t = time.time()
            cd = PGd.displaced(cell, A, x, s * H)
            ints_d, _, gd_d, grid_d = setup(cd, False)
            for label, xc in cases:
                for ex in ("none", "ewald"):
                    d = with_vm(ints_d, ex)
                    rd = R.rohf_scf(d["S"], d["h"], jk_from_B(gd_d["B"]), d["enn"], 1, 1, grid_d, xc, d["madelung"],
                                    C0=ref[(label, ex)][0]["C"])
                    ev[(label, ex, s)] = rd["e"]
            print(f"  FD ({A},{x}) s={s:+d} ({time.time() - t:.0f}s)", flush=True)
        for label, xc in cases:
            for ex in ("none", "ewald"):
                fd[(label, ex, A, x)] = (ev[(label, ex, 1)] - ev[(label, ex, -1)]) / (2 * H)
    for label, xc in cases:
        for ex in ("none", "ewald"):
            r, g, muts = ref[(label, ex)]
            dif = [g[A, x] - fd[(label, ex, A, x)] for A, x in sysd["comps"][:2]]
            print(f"  {label:10s} {ex:5s} an-FD " + " ".join(f"{v:+.2e}" for v in dif)
                  + f" |sumF| {abs(g.sum(0)).max():.1e} | "
                  + " | ".join(f"{m} {max(abs(gm[A, x] - fd[(label, ex, A, x)]) for A, x in sysd['comps'][:2]):.1e}"
                               for m, gm in muts.items()))
        print(f"  {label}: F(ewald)-F(none) {abs(ref[(label, 'ewald')][1] - ref[(label, 'none')][1]).max():.1e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
