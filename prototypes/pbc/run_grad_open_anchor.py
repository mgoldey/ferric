"""Iteration 17 exactness anchors for Gamma UHF / RKS / UKS forces (pbc_grad_open.py; no PySCF pbc anywhere).
Usage: python3 run_grad_open_anchor.py {h2|h3|tri|h3sr}

For every case: analytic force vs central FD (h = 1e-4) of the prototype's OWN Gamma energy, two FD flavours:
  FD_move   the SSF grid is REBUILT at the displaced geometry (points ride on their home atom, weights recomputed)
            = the true derivative of the energy as defined; must match analytic WITH grid response;
  FD_frozen points and weights of the reference geometry, AOs re-evaluated at the displaced atoms
            = isolates the AO-derivative XC term; must match analytic WITHOUT grid response.
Pure-AFT dense J/K (gcut as the RHF anchors), exxdiv none and ewald from ONE integral build (only v_M differs).

PREDICTIONS (written before running):
 P1 correct code: |analytic(resp) - FD_move| and |analytic(noresp) - FD_frozen| at the FD truncation floor (h^2 f'''/6,
    ~1e-9 as in Iteration 16), for UHF, RKS, UKS, LDA/PBE/PBE0, both exxdiv.
 P2 sum_A F_A ~ 1e-15 with grid response (exact translation invariance of the energy, grid included); WITHOUT it
    sum F = the grid-response total, O(grid error) ~1e-4..1e-3 at this coarse (40, 50) grid.
 P3 |analytic(noresp) - FD_move| = size of the grid response: grid-error sized, O(1e-4..1e-3) here, shrinking with
    the grid (not measured to convergence -- scope).
 P4 F(ewald) == F(none) to ~1e-12 when both SCFs land on the same state (per-spin Madelung cancellation, any alpha).
 P5 closed shell: UHF(n,n) == RHF (pbc_grad) and UKS(n,n) == RKS to ~1e-12 (polarized kernel at rho_a = rho_b).
ARTIFACT HYPOTHESES (if broken I expect ...):
 - per-spin W replaced by 1/2 D Fbar D: invisible for closed shell, O(1e-2) on open shells;
 - Madelung S-term on total D (RHF form): invisible closed shell and exxdiv=none; O(v_M) ~ 1e-1 on open shells, ewald;
 - exchange Gam on total D: invisible closed shell; O(1e-2) open shell UHF/PBE0;
 - dropping the XC AO term: O(1e-1) everywhere (the XC force is a large fraction of the electronic force);
 - dropping the GGA sigma part: O(1e-2), PBE/PBE0 only;
 - dropping point motion or weight derivative: each O(1e-3..1e-2) and of opposite sign, largely cancelling each other
   (their sum is the small grid response) -> vs FD_move each mutant misses by its own size, far above 1e-9;
 - a wrong image fold in the weight derivative: sum F != 0 with response (P2 fails).
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_open as O
from pbc_gamma import Cell
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H = 1e-4
H2 = dict(a=np.eye(3) * 4.0, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))], basis="sto-3g", gcut=12.0,
          D=8.0, comps=[(0, 0), (0, 2), (1, 1)])
H3 = dict(a=np.eye(3) * 4.5, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))],
          basis="sto-3g", gcut=12.0, D=8.0, comps=[(0, 0), (2, 1), (1, 2)])
TRI = dict(a=TRI_A, atoms=TRI_MOVED, basis=SP_BASIS, gcut=10.0, D=8.0, comps=[(2, 1), (0, 2)])
H3_SR = dict(a=np.eye(3) * 4.5, atoms=H3["atoms"], basis={"H": gto.parse("H S\n  0.5  1.0\n")}, gcut=None, D=8.0,
             comps=[(0, 0), (2, 1)])

SYSTEMS = {
    # name: (system, [(label, na, nb, xc, restricted)])
    "h2": (H2, [("RHF", 1, 1, "HF", True), ("UHF(1,1)", 1, 1, "HF", False),
                ("RKS LDA", 1, 1, "LDA,VWN", True), ("RKS PBE", 1, 1, "PBE", True), ("RKS PBE0", 1, 1, "PBE0", True),
                ("UKS(1,1) PBE", 1, 1, "PBE", False), ("UKS triplet PBE", 2, 0, "PBE", False)]),
    "h3": (H3, [("UHF doublet", 2, 1, "HF", False), ("UKS LDA", 2, 1, "LDA,VWN", False),
                ("UKS PBE", 2, 1, "PBE", False), ("UKS PBE0", 2, 1, "PBE0", False)]),
    "tri": (TRI, [("RHF", 2, 2, "HF", True), ("UHF triplet", 3, 1, "HF", False),
                  ("RKS PBE", 2, 2, "PBE", True), ("UKS(2,2) PBE", 2, 2, "PBE", False),
                  ("UKS LDA", 3, 1, "LDA,VWN", False), ("UKS PBE", 3, 1, "PBE", False),
                  ("UKS PBE0", 3, 1, "PBE0", False)]),
}
MUTS_HF_OPEN = ["w_spin_sum", "madelung_total", "exch_total"]
MUTS_XC = ["xc_no_ao", "xc_no_gga", "no_point_motion", "no_weight_deriv", "w_spin_sum", "exch_total", "madelung_total"]


def build(sysd, cell, want_dw=True, w=None):
    ints = O.integrals(cell, w, "ewald", gcut=sysd["gcut"])
    grid = O.GradGrid(cell, 40, 50, D=sysd["D"], scheme="ssf", deriv=2 if want_dw else 1, want_dw=want_dw)
    return ints, grid


def with_vm(ints, ex):
    d = dict(ints)
    if ex == "none":
        d["madelung"] = 0.0
    return d


def main(which):
    if which == "h3sr":
        return main_sr()
    sysd, cases = SYSTEMS[which]
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    t = time.time()
    ints, grid = build(sysd, cell)
    print(f"[{which}] nao {cell.mol.nao}, v_M {ints['madelung']:.10f}, npts {grid.size}, build {time.time() - t:.0f}s",
          flush=True)
    ref = {}
    for label, na, nb, xc, rst in cases:
        for ex in ("none", "ewald"):
            t = time.time()
            # staged (Iteration 6): ewald starts from the converged exxdiv=none state -> same state for P4
            gs = None if ex == "none" else (ref[(label, "none")]["scf"]["Da"], ref[(label, "none")]["scf"]["Db"])
            r = O.run_case(cell, na, nb, xc, with_vm(ints, ex), grid=grid, restricted=rst, guess=gs)
            ref[(label, ex)] = r
            print(f"  ref {label:16s} {ex:5s} E {r['e']:.12f} it {r['scf']['it']} err {r['scf']['err']:.1e} "
                  f"({time.time() - t:.0f}s)", flush=True)
    # finite differences: one integral build + one moving grid + one frozen AO table per displacement
    fd_move, fd_frz = {}, {}
    for A, x in sysd["comps"]:
        ev = {}
        for s in (+1, -1):
            t = time.time()
            cd = PGd.displaced(cell, A, x, s * H)
            ints_d, grid_d = build(sysd, cd, want_dw=False)
            grid_f = O.GradGrid.frozen(grid, cd, deriv=1)
            for label, na, nb, xc, rst in cases:
                for ex in ("none", "ewald"):
                    r0 = ref[(label, ex)]["scf"]
                    gs = (r0["Da"], r0["Db"])
                    em = O.energy_only(cd, na, nb, xc, with_vm(ints_d, ex), grid_d, rst, gs)
                    ef = em if xc == "HF" else O.energy_only(cd, na, nb, xc, with_vm(ints_d, ex), grid_f, rst, gs)
                    ev[(label, ex, s)] = (em, ef)
            print(f"  FD ({A},{x}) s={s:+d} done ({time.time() - t:.0f}s)", flush=True)
        for label, na, nb, xc, rst in cases:
            for ex in ("none", "ewald"):
                (mp, fp), (mm, fm) = ev[(label, ex, 1)], ev[(label, ex, -1)]
                fd_move[(label, ex, A, x)] = (mp - mm) / (2 * H)
                fd_frz[(label, ex, A, x)] = (fp - fm) / (2 * H)

    print(f"\n[{which}] RESULTS  (max over comps {sysd['comps']}; noresp = analytic minus xc_point minus xc_weight)")
    print("| case | exxdiv | an(resp)-FD_move | an(noresp)-FD_frozen | an(noresp)-FD_move | sumF resp | sumF noresp |")
    for label, na, nb, xc, rst in cases:
        for ex in ("none", "ewald"):
            r = ref[(label, ex)]
            g = r["grad"]
            gn = g - r["parts"].get("xc_point", 0) - r["parts"].get("xc_weight", 0)
            d1 = max(abs(g[A, x] - fd_move[(label, ex, A, x)]) for A, x in sysd["comps"])
            d2 = max(abs(gn[A, x] - fd_frz[(label, ex, A, x)]) for A, x in sysd["comps"])
            d3 = max(abs(gn[A, x] - fd_move[(label, ex, A, x)]) for A, x in sysd["comps"])
            print(f"| {label} | {ex} | {d1:.1e} | {d2 if xc != 'HF' else float('nan'):.1e} | {d3:.1e} | "
                  f"{abs(g.sum(0)).max():.1e} | {abs(gn.sum(0)).max():.1e} |", flush=True)
    print("\nF(ewald) - F(none) per case:")
    for label, *_ in cases:
        print(f"  {label:16s} {abs(ref[(label, 'ewald')]['grad'] - ref[(label, 'none')]['grad']).max():.1e}  "
              f"dE(ewald-none) {ref[(label, 'ewald')]['e'] - ref[(label, 'none')]['e']:+.10f}")
    # closed-shell identities
    labels = [c[0] for c in cases]
    for a_, b_ in (("UHF(1,1)", "RHF"), ("UKS(1,1) PBE", "RKS PBE"), ("UKS(2,2) PBE", "RKS PBE")):
        if a_ in labels and b_ in labels:
            for ex in ("none", "ewald"):
                print(f"  closed shell {a_} - {b_} [{ex}]: max|dF| "
                      f"{abs(ref[(a_, ex)]['grad'] - ref[(b_, ex)]['grad']).max():.1e}")
    if "RHF" in labels:
        for ex in ("none", "ewald"):
            rr = PGd.gamma_rhf_grad(cell, cell.mol.nelectron, exxdiv=ex, gcut=sysd["gcut"], ints=with_vm(ints, ex))
            print(f"  RHF(pbc_grad, Iteration 16) - RHF(this) [{ex}]: {abs(rr['grad'] - ref[('RHF', ex)]['grad']).max():.1e}")

    print("\nMUTANTS: max|analytic(resp) - FD_move| (and |sum F|)")
    for label, na, nb, xc, rst in cases:
        muts = MUTS_XC if xc != "HF" else MUTS_HF_OPEN
        for ex in ("none", "ewald"):
            r = ref[(label, ex)]
            sc = r["scf"]
            row = []
            for m in muts:
                if m == "xc_no_gga" and O.pd.xc_family(xc) != "GGA":
                    continue
                O._MUTANT = m
                try:
                    g, _ = O.gamma_grad(cell, with_vm(ints, ex), sc["Da"], sc["Db"], sc["Fa"], sc["Fb"], sc["hyb"],
                                        grid=None if xc == "HF" else grid, xc=xc, restricted=rst)
                finally:
                    O._MUTANT = None
                d = max(abs(g[A, x] - fd_move[(label, ex, A, x)]) for A, x in sysd["comps"])
                row.append(f"{m} {d:.1e} (sF {abs(g.sum(0)).max():.0e})")
            print(f"  {label:16s} {ex:5s} " + " | ".join(row), flush=True)


def main_sr():
    """UHF doublet on the Ewald-split route (w = 1; SR 3c V_ne + SR 4c ERI with the per-spin Gam; compact s basis so
    pbc_gamma's default SR cutoffs are converged), exxdiv none/ewald, plus the exch_total mutant."""
    sysd = H3_SR
    cell = Cell(sysd["a"], sysd["atoms"], sysd["basis"])
    t = time.time()
    ints = O.integrals(cell, 1.0, "ewald")
    print(f"[h3sr] nao {cell.mol.nao}, ints {time.time() - t:.0f}s", flush=True)
    gs0 = None  # ewald starts from the converged exxdiv=none state (staged, Iteration 6)
    for ex in ("none", "ewald"):
        t = time.time()
        r = O.run_case(cell, 2, 1, "HF", with_vm(ints, ex), w=1.0, guess=gs0)
        gs0 = (r["scf"]["Da"], r["scf"]["Db"])
        fd = {}
        for A, x in sysd["comps"]:
            e = [O.energy_only(PGd.displaced(cell, A, x, s * H), 2, 1, "HF",
                               with_vm(O.integrals(PGd.displaced(cell, A, x, s * H), 1.0, "ewald"), ex), None, False,
                               (r["scf"]["Da"], r["scf"]["Db"])) for s in (1, -1)]
            fd[(A, x)] = (e[0] - e[1]) / (2 * H)
        d = max(abs(r["grad"][k] - v) for k, v in fd.items())
        print(f"  UHF doublet SR route {ex:5s}: E {r['e']:.12f} max|an-FD| {d:.1e} |sumF| {abs(r['grad'].sum(0)).max():.1e}"
              f" ({time.time() - t:.0f}s)", flush=True)
        sc = r["scf"]
        for m in ("exch_total", "w_spin_sum") + (("madelung_total",) if ex == "ewald" else ()):
            O._MUTANT = m
            try:
                g, _ = O.gamma_grad(cell, with_vm(ints, ex), sc["Da"], sc["Db"], sc["Fa"], sc["Fb"], 1.0, w=1.0)
            finally:
                O._MUTANT = None
            print(f"    MUTANT {m:15s} {max(abs(g[k] - v) for k, v in fd.items()):.1e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h2")
