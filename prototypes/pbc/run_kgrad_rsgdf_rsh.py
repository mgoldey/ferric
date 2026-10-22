"""Issue #492 item 1 anchors: RSH exchange forces on k-point RS-GDF (pbc_kgrad_rsgdf_rsh.py).

Usage: python run_kgrad_rsgdf_rsh.py [anchor] [fd] [total] [mut]     (default: all)

System: triclinic H2 / STO-3G (run_kgrad_anchor.TRI2 geometry), 1x1x2 mesh, ET-sp aux (run_kgrad_gdf.ET_SP), a FIXED
model density D_s(k) = C C^+ (lowest generalised eigenvector of diag(1, 2) against S(k)); the force is dE/dR at
fixed D_s(k) of the exchange pieces only.

PREDICTIONS / ARTIFACT HYPOTHESES (written before the first run):
 A0 shared code: fit_xgrad(plain fit, with_j) == pbc_kgrad_gdf.grad's gdf parts to rounding (<= 1e-13); fit_energy
    exchange == -(1/2Nk) sum tr[K D] of jk_from_kB (<= 1e-14).  A slip in the copied passes shows here, not later.
 A1 w -> infinity (w = 1e7, gcut 24 covering the Coulomb tail of the fitted pairs): the attenuated fit IS the plain
    Coulomb fit with K = 0 dropped (the plain fit's -c0 terms remove exactly the SR sums' K = 0 content), so E and F
    equal the plain fit's to its build precision (~1e-10).  If the attenuated mode wrongly needed c0, or the plain
    fit's G0 force were not the K=0 content of the SR derivative, A1 would miss by O(c0) ~ 1e-2.
 A2 FD of the attenuated exchange alone, w = 0.11 (HSE) and 0.33 (CAM): central differences h = 1e-3..1e-5, O(h^2)
    then a rounding floor (~1e-9 at 1e-5).  An analytic slip (missing metric, wrong aux sign) is O(1e-3..1e-2).
 A3 total RSH exchange force, HSE06 (c_sr .25, c_lr 0, w .11) and CAM-B3LYP (.19, .65, .33), incl. mixed Madelung,
    vs FD at the same floor.
 Mutants must move F by >> the FD floor; the scalar ones (no_scale, madelung_*) by construction.
"""

import os
import sys
import time

import numpy as np

import pbc_kgrad as KG
import pbc_kgrad_gdf as KGG
import pbc_kgrad_rsgdf_rsh as R
from pbc_gamma import Cell
from pbc_kgdf import jk_from_kB
from run_kgrad_anchor import TRI2
from run_kgrad_gdf import ET_SP

MESH = (1, 1, 2)
W0 = 1.0
# metric cut of the ATTENUATED fit only (the plain fit keeps 1e-10); env ATT_LINDEP, see the conditioning note
ATT_LINDEP = float(os.environ.get("ATT_LINDEP", "1e-10"))
HSE06 = dict(c_sr=0.25, c_lr=0.0, omega=0.11)
CAMB3 = dict(c_sr=0.19, c_lr=0.65, omega=0.33)
COMPS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
HS = (1e-3, 1e-4, 1e-5)


def cell_at(atoms=None):
    return Cell(TRI2["a"], atoms or TRI2["atoms"], "sto-3g")


def moved(ia, x, h):
    at = [(s, np.array(r, float)) for s, r in TRI2["atoms"]]
    at[ia][1][x] += h
    return [(s, tuple(r)) for s, r in at]


def model_density(Sk):
    Ds = []
    for S in Sk:
        Hm = np.diag(np.arange(1.0, S.shape[0] + 1))
        Sv, Uv = np.linalg.eigh(S)
        X = Uv / np.sqrt(Sv)
        e, V = np.linalg.eigh(X.conj().T @ Hm @ X)
        C = X @ V[:, :1]
        Ds.append(C @ C.conj().T)
    D = np.array(Ds)
    return [D, D]


def plain(cell, deriv=True):
    return KGG.build(cell, MESH, KGG.make_auxmol(cell, ET_SP), w=W0, deriv=deriv)


def att(cell, omega, **kw):
    kw.setdefault("lindep", ATT_LINDEP)
    return R.build_att(cell, MESH, KGG.make_auxmol(cell, ET_SP), omega, w0=W0, **kw)


def energies(cell, Ds, fns):
    gp = plain(cell, deriv=False)
    e = dict(full=R.fit_energy(gp, Ds), S=gp["S"])
    for f in fns:
        e[f["omega"]] = R.fit_energy(att(cell, f["omega"]), Ds)
    return e


def rsh_total(f, Ffull, Fatt, FM1, vf, vl, mutant=None):
    """F = c_sr F_full + (c_lr - c_sr) F_att + v_M^mix FM1 (FM1 = Madelung force per unit v_M)."""
    s = 1.0 if mutant == "no_scale" else f["c_lr"] - f["c_sr"]
    return (
        f["c_sr"] * Ffull
        + s * Fatt
        + R.v_mix(f["c_sr"], f["c_lr"], vf, vl, mutant) * FM1
    )


def main(which):
    t0 = time.time()
    cell = cell_at()
    gp = plain(cell)
    Ds = model_density(gp["S"])
    Nk = gp["Nk"]
    print(
        f"# triclinic H2/STO-3G mesh {MESH}, aux {gp['auxmol'].nao} cart, w0 {W0}, att lindep {ATT_LINDEP:.0e}; plain build {gp['wall']:.0f}s"
    )
    Fp = R.fit_xgrad(cell, gp, Ds)["total"]
    out = {}
    if "anchor" in which:
        # A0: shared code vs pbc_kgrad_gdf.grad (non-2e terms stubbed out)
        real_kgrad = KG.kgrad
        KG.kgrad = lambda *a, **k: (0.0, {})
        try:
            _, parts, _ = KGG.grad(
                cell, None, gp, Ds[0], Ds[1], 0 * Ds[0], 0 * Ds[1], restricted=True
            )
        finally:
            KG.kgrad = real_kgrad
        Fj = R.fit_xgrad(cell, gp, Ds, with_j=True)["total"]
        ref = parts["gdf_orb"] + parts["gdf_aux"] + parts["gdf_g0"]
        print(
            f"A0 |F(fit_xgrad, J+K) - F(pbc_kgrad_gdf.grad)| = {abs(Fj - ref).max():.2e}  (|F| {abs(ref).max():.3e})"
        )
        jk = jk_from_kB(gp)
        eK_ref = sum(-0.5 / Nk * np.einsum("kmn,knm->", jk(D)[1], D).real for D in Ds)
        print(
            f"A0 |E_K(fit_energy) - E_K(jk_from_kB)| = {abs(R.fit_energy(gp, Ds) - eK_ref):.2e}  (E_K {eK_ref:.10f})"
        )
        # A1: w -> infinity
        ga = att(cell, 1e7, gcut=24.0)
        ea, ep = R.fit_energy(ga, Ds), R.fit_energy(gp, Ds)
        Fa = R.fit_xgrad(cell, ga, Ds)["total"]
        print(
            f"A1 w=1e7 gcut 24 (nK/q {[p['nK'] for p in ga['per_q']]}): |E_att - E_plain| = {abs(ea - ep):.2e}, "
            f"|F_att - F_plain| = {abs(Fa - Fp).max():.2e}  (|F_plain| {abs(Fp).max():.3e}, sumF_att {abs(Fa.sum(0)).max():.1e})"
        )
        ga2 = att(cell, 1e7, gcut=16.0)
        print(
            f"   (gcut 16 for contrast: |dE| {abs(R.fit_energy(ga2, Ds) - ep):.2e}, "
            f"|dF| {abs(R.fit_xgrad(cell, ga2, Ds)['total'] - Fp).max():.2e})"
        )
    fns = (HSE06, CAMB3)
    Fatt = {}
    for f in fns:
        g = att(cell, f["omega"])
        Fatt[f["omega"]] = R.fit_xgrad(cell, g, Ds)["total"]
        print(
            f"# att w={f['omega']}: nK/q {[p['nK'] for p in g['per_q']]}, kept {[int(p['keep'].sum()) for p in g['per_q']]}"
            f"/{len(g['per_q'][0]['s'])}, |F_att| {abs(Fatt[f['omega']]).max():.3e}, sumF {abs(Fatt[f['omega']].sum(0)).max():.1e}"
        )
    vf, vl = {}, {}
    for f in fns:
        vf[f["omega"]], vl[f["omega"]] = R.madelung_pair(cell, MESH, f["omega"])
    FM1 = R.madelung_grad(cell, gp["kpts"], Ds, gp["S"], 1.0, gp["rg"]["rcut_pair"])
    if "fd" in which or "total" in which:
        res = {}
        for ia, x in COMPS if "fd" in which else COMPS[:2]:
            for h in HS:
                ep_ = energies(cell_at(moved(ia, x, h)), Ds, fns)
                em_ = energies(cell_at(moved(ia, x, -h)), Ds, fns)
                res[(ia, x, h)] = (ep_, em_)
        print(
            "A2 attenuated exchange alone: analytic - FD (central) per component and h"
        )
        for f in fns:
            w = f["omega"]
            for h in HS:
                errs = [
                    Fatt[w][ia, x]
                    - (res[(ia, x, h)][0][w] - res[(ia, x, h)][1][w]) / (2 * h)
                    for ia, x in COMPS
                    if (ia, x, h) in res
                ]
                print(f"   w={w} h={h:.0e}: max|err| {max(map(abs, errs)):.2e}")
        print(
            "A3 total RSH exchange (c_sr full + (c_lr-c_sr) att + mixed Madelung): analytic - FD"
        )
        for name, f in (("HSE06", HSE06), ("CAM-B3LYP", CAMB3)):
            w = f["omega"]
            vm = R.v_mix(f["c_sr"], f["c_lr"], vf[w], vl[w])
            Ftot = rsh_total(f, Fp, Fatt[w], FM1, vf[w], vl[w])
            for h in HS:
                errs = []
                for ia, x in COMPS:
                    if (ia, x, h) not in res:
                        continue
                    E = []
                    for e in res[(ia, x, h)]:
                        E.append(
                            f["c_sr"] * e["full"]
                            + (f["c_lr"] - f["c_sr"]) * e[w]
                            + R.madelung_energy(None, e["S"], Ds, vm)
                        )
                    errs.append(Ftot[ia, x] - (E[0] - E[1]) / (2 * h))
                print(
                    f"   {name} (v_M^mix {vm:.6f}) h={h:.0e}: max|err| {max(map(abs, errs)):.2e}"
                )
            out[name] = Ftot
            print(f"   {name} |F_x^RSH| {abs(Ftot).max():.3e}")
    if "mut" in which:
        print("MUTATION LEDGER: max|F_mut - F| (exchange force, Ha/Bohr)")
        for name, f in (("HSE06", HSE06), ("CAM-B3LYP", CAMB3)):
            w = f["omega"]
            F = rsh_total(f, Fp, Fatt[w], FM1, vf[w], vl[w])
            for m in ("keep_c0", "omega0"):
                Fm_att = R.fit_xgrad(cell, att(cell, w, mutant=m), Ds)["total"]
                d = abs(rsh_total(f, Fp, Fm_att, FM1, vf[w], vl[w]) - F).max()
                print(f"   {name:9s} {m:17s} {d:.2e}")
            for m in (
                "no_scale",
                "madelung_unmixed",
                "madelung_full_lr",
                "madelung_scaled",
            ):
                d = abs(
                    rsh_total(f, Fp, Fatt[w], FM1, vf[w], vl[w], mutant=m) - F
                ).max()
                print(f"   {name:9s} {m:17s} {d:.2e}")
        print(
            f"   (v_M^full {vf[0.11]:.6f}, v_M^LR(.11) {vl[0.11]:.6f}, v_M^LR(.33) {vl[0.33]:.6f}; |FM1| {abs(FM1).max():.3e})"
        )
    print(f"# wall {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"anchor", "fd", "total", "mut"})
