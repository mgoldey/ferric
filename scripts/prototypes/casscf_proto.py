#!/usr/bin/env python
"""Two-step CASSCF prototype in numpy on ferric's integral blocks.

Correctness reference for a Rust CASSCF (T048). Integrals come from
``ferric.AoIntegrals`` (``hcore``, ``jk``, ``ao2mo``) and the starting
orbitals from ``ferric.run_rhf``. The CAS-CI step uses ``pyscf.fci``
(``direct_spin1``) behind the ``PyscfFci.solve`` interface, so a ferric-ci solver
with RDMs can replace it without touching the orbital code. Nothing about
speed may be quoted from this prototype. Equations: casscf_derivation.md.

One macro iteration:

1. inactive Fock F^I and core energy from ``jk(D_core)``;
2. CAS-CI in the active space (h_eff = C_a^T F^I C_a, (tu|vw));
3. 1-/2-RDMs, active Fock F^A from ``jk(C_a gamma C_a^T)``, (pu|vw) from
   ``ao2mo(C, C_a, C_a, C_a)``, generalized Fock F;
4. orbital gradient g_pq = 2 (F_pq - F_qp) over core-active, core-virtual
   and active-virtual pairs;
5. BFGS step (initial inverse Hessian = 1 / approximate diagonal Hessian),
   orbital update C <- C expm(K).

Anchors, in order (``--only`` picks one):

* ``full``: active space = full space; 2(F - F^T) is zero to CI convergence
  and the correlation energy E - E_RHF equals PySCF FCI on PySCF's own
  integrals (the total energies differ by the RHF-level integral offset).
* ``fd``: analytic gradient vs a Richardson-extrapolated 5-point central
  difference of the CAS-CI energy, every nonredundant rotation, reported per
  rotation class; then the same comparison with each mutation of the
  gradient, which must fail.
* ``ref``: converged energies vs ``pyscf.mcscf.CASSCF`` for H2O CAS(4,4)/6-31G
  and N2 CAS(6,6)/6-31G. PySCF converges from its own RHF orbitals on
  ferric's AO integrals (gated) and on its own integrals (reported).

Usage (one heavy op at a time on the shared box):
  cd ~/qc/ferric && box-lease run --label wp5 -- env OPENBLAS_NUM_THREADS=1 \\
    uv run --no-sync python ~/qc/ferric-wp5/scripts/prototypes/casscf_proto.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import ferric

ROOT = Path(__file__).resolve().parents[2]
MOLS = ROOT / "testdata" / "molecules"

JK_THRESH = 1e-14
CLASSES = ("core-act", "core-virt", "act-virt")


def log(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------- CI solver


@dataclass
class CiResult:
    e: float  # active-space energy (no core energy)
    rdm1: np.ndarray  # spin-summed gamma_tu
    rdm2: np.ndarray  # spin-summed Gamma_tuvw, E2 = 1/2 sum (tu|vw) Gamma_tuvw
    ci: np.ndarray
    residual: float  # ||H c - E c||


class PyscfFci:
    """CAS-CI on pyscf.fci.direct_spin1: dense eigh up to ``dense_max``
    determinants, Davidson above. The interface a ferric-ci solver must meet:
    ``solve(h1, eri, ncas, nelecas, ci0) -> CiResult`` with PySCF's RDM
    convention (E_act = sum h_tu g_tu + 1/2 sum (tu|vw) G_tuvw)."""

    name = "pyscf.fci.direct_spin1"

    def __init__(self, conv_tol: float = 1e-14, dense_max: int = 2500):
        self.conv_tol = conv_tol
        self.dense_max = dense_max

    def solve(self, h1, eri, ncas, nelecas, ci0=None) -> CiResult:
        from pyscf import fci
        from pyscf.fci import cistring

        s = fci.direct_spin1.FCI()
        s.verbose = 0
        s.conv_tol = self.conv_tol
        s.max_cycle = 500
        ndet = cistring.num_strings(ncas, nelecas[0]) * cistring.num_strings(
            ncas, nelecas[1]
        )
        if ndet <= self.dense_max:
            s.pspace_size = ndet
            e, ci = s.kernel(h1, eri, ncas, nelecas)
        else:
            s.pspace_size = 0
            s.conv_tol_residual = 1e-9
            e, ci = s.kernel(h1, eri, ncas, nelecas, ci0=ci0)
        ci = np.asarray(ci).reshape(cistring.num_strings(ncas, nelecas[0]), -1)
        h2 = s.absorb_h1e(h1, eri, ncas, nelecas, 0.5)
        r = s.contract_2e(h2, ci, ncas, nelecas).reshape(ci.shape) - e * ci
        # PySCF's flag tracks the 1e-14 energy tolerance. The energy error is
        # second order in the residual, so a residual below RESID_MAX bounds it
        # near 1e-14 / gap, under every anchor bar.
        if not s.converged and np.linalg.norm(r) > RESID_MAX:
            raise RuntimeError(
                f"CAS-CI residual {np.linalg.norm(r):.1e} after {s.max_cycle} Davidson cycles "
                f"({ncas} orbitals, {nelecas} electrons)"
            )
        d1, d2 = s.make_rdm12(ci, ncas, nelecas)
        return CiResult(e, d1, d2, ci, float(np.linalg.norm(r)))


RESID_MAX = 1e-7


def full_space_bar(e: float) -> float:
    """Bar on |E_proto - E_FCI| when both run on the same integrals.

    Both are PySCF FCI solves of one orbital-invariant problem in two orbital
    bases, so they differ by each solve's energy tolerance (1e-14 Ha) and the
    rounding of the two MO transforms, allowed at 1e-13 |E| (about 450 ulps).
    MEASURED 2026-11-12: |dE| = 1.3e-13 (water/STO-3G), 2.8e-13 (N2/STO-3G).
    """
    return 2e-14 + 1e-13 * abs(e)


# ---------------------------------------------------------------- CASSCF core


class Cas:
    """Core/active/virtual partition of the MOs and the CAS energy functional."""

    def __init__(self, ints, ncore, ncas, nelecas, solver):
        self.ints = ints
        self.h = ints.hcore()
        self.enuc = ints.nuclear_repulsion()
        self.nc, self.na = ncore, ncas
        self.nelecas = nelecas
        self.solver = solver
        self.nmo = None
        self.ci0 = None

    def setup_pairs(self, nmo):
        nc, na = self.nc, self.na
        self.nmo = nmo
        core = range(nc)
        act = range(nc, nc + na)
        virt = range(nc + na, nmo)
        # p > q, rotation kappa_pq; class order matches CLASSES.
        pairs = [(t, i, 0) for t in act for i in core]
        pairs += [(a, i, 1) for a in virt for i in core]
        pairs += [(a, t, 2) for a in virt for t in act]
        self.pairs = np.array([(p, q) for p, q, _ in pairs], dtype=int).reshape(-1, 2)
        self.cls = np.array([c for _, _, c in pairs], dtype=int)

    def pack(self, g_mat):
        return g_mat[self.pairs[:, 0], self.pairs[:, 1]]

    def rotate(self, c, kvec):
        k = np.zeros((self.nmo, self.nmo))
        k[self.pairs[:, 0], self.pairs[:, 1]] = kvec
        k[self.pairs[:, 1], self.pairs[:, 0]] = -kvec
        return c @ expm(k)

    def _inactive(self, c):
        cc = c[:, : self.nc]
        if self.nc:
            dc = 2.0 * cc @ cc.T
            j, k = self.ints.jk(dc, thresh=JK_THRESH)
            fi = self.h + j - 0.5 * k
            ecore = self.enuc + 0.5 * np.sum(dc * (self.h + fi))
        else:
            fi, ecore = self.h.copy(), self.enuc
        return fi, ecore

    def energy(self, c, warm=False):
        """CAS-CI energy at orbitals c (CI relaxed)."""
        nc, na = self.nc, self.na
        fi, ecore = self._inactive(c)
        ca = c[:, nc : nc + na]
        eri = self.ints.ao2mo(ca, ca, ca, ca)
        res = self.solver.solve(
            ca.T @ fi @ ca, eri, na, self.nelecas, self.ci0 if warm else None
        )
        return ecore + res.e, res

    def gradient(self, c, mutation=None):
        """Energy, packed gradient and intermediates at orbitals c."""
        nc, na = self.nc, self.na
        occ = slice(nc, nc + na)
        fi_ao, ecore = self._inactive(c)
        ca = c[:, occ]
        paaa = self.ints.ao2mo(c, ca, ca, ca)  # (pu|vw), p over all MOs
        res = self.solver.solve(
            ca.T @ fi_ao @ ca, paaa[occ], na, self.nelecas, self.ci0
        )
        self.ci0 = res.ci
        g1, g2 = res.rdm1, res.rdm2
        j, k = self.ints.jk(ca @ g1 @ ca.T, thresh=JK_THRESH)
        fa_ao = j - 0.5 * k
        fi, fa = c.T @ fi_ao @ c, c.T @ fa_ao @ c
        f = np.zeros_like(fi)
        f[:, :nc] = 2.0 * (
            fi[:, :nc] + (0.0 if mutation == "drop_FA_core" else fa[:, :nc])
        )
        f[:, occ] = fi[:, occ] @ g1
        if mutation != "drop_2rdm":
            f[:, occ] += np.einsum("puvw,tuvw->pt", paaa, g2)
        gmat = 2.0 * (f - f.T)
        g = self.pack(gmat)
        if mutation == "flip_core_virt":
            g[self.cls == 1] *= -1.0
        if mutation == "zero_act_virt":
            g[self.cls == 2] = 0.0
        return {
            "e": ecore + res.e,
            "g": g,
            "gmat": gmat,
            "fi": fi,
            "fa": fa,
            "fgen": f,
            "rdm1": g1,
            "ci": res,
        }

    def approx_hdiag(self, x):
        """Approximate diagonal orbital Hessian (Chaban-Schmidt-Gordon form)."""
        ft = x["fi"] + x["fa"]
        fg = x["fgen"]
        g1 = x["rdm1"]
        nc = self.nc
        h = np.empty(len(self.cls))
        for n, ((p, q), cl) in enumerate(zip(self.pairs, self.cls)):
            if cl == 1:  # a, i
                h[n] = 4.0 * (ft[p, p] - ft[q, q])
            elif cl == 2:  # a, t
                gtt = g1[q - nc, q - nc]
                h[n] = 2.0 * gtt * ft[p, p] - 2.0 * fg[q, q]
            else:  # t, i
                gtt = g1[p - nc, p - nc]
                h[n] = (
                    4.0 * (ft[p, p] - ft[q, q]) + 2.0 * gtt * ft[q, q] - 2.0 * fg[p, p]
                )
        return h


def casscf(cas, c0, gtol=1e-7, etol=1e-12, max_iter=300, max_step=0.2, hfloor=0.05):
    """Two-step CASSCF: CI solve, then a BFGS orbital step, until converged."""
    c = c0.copy()
    cas.setup_pairs(c.shape[1])
    e_old, g_old, s_old, hinv = None, None, None, None
    for it in range(1, max_iter + 1):
        x = cas.gradient(c)
        e, g = x["e"], x["g"]
        gn = np.linalg.norm(g)
        if g.size == 0:
            # No nonredundant rotations (ncore = 0, ncas = nmo): CAS-CI is the answer.
            return {"e": e, "c": c, "iters": it, "gnorm": 0.0, "x": x}
        if e_old is not None and abs(e - e_old) < etol and gn < gtol:
            return {"e": e, "c": c, "iters": it, "gnorm": gn, "x": x}
        hd = np.maximum(np.abs(cas.approx_hdiag(x)), hfloor)
        if hinv is None or (e_old is not None and e > e_old + 1e-10):
            hinv = np.diag(1.0 / hd)  # (re)start
        else:
            y = g - g_old
            sy = s_old @ y
            if sy > 1e-14:
                rho = 1.0 / sy
                v = np.eye(len(g)) - rho * np.outer(s_old, y)
                hinv = v @ hinv @ v.T + rho * np.outer(s_old, s_old)
        step = -hinv @ g
        smax = np.abs(step).max()
        if smax > max_step:
            step *= max_step / smax
        c = cas.rotate(c, step)
        e_old, g_old, s_old = e, g, step
    raise RuntimeError(f"CASSCF not converged in {max_iter} iterations (|g|={gn:.2e})")


# ---------------------------------------------------------------- set-up


def ferric_setup(xyz, basis):
    mol = ferric.Molecule.from_xyz(str(MOLS / xyz))
    bs = ferric.BasisSet.bundled(basis)
    ints = ferric.AoIntegrals(mol, bs)
    r = ferric.run_rhf(mol, bs, energy_conv=1e-12, density_conv=1e-10)
    if not r.converged:
        raise RuntimeError("ferric RHF did not converge")
    return mol, ints, np.asarray(r.mo_coefficients()), r.energy


def pyscf_mol(mol, basis):
    from pyscf import gto

    atoms = [(s, xyz) for s, xyz in zip(mol.symbols(), mol.coords_bohr())]
    return gto.M(atom=atoms, unit="Bohr", basis=basis, verbose=0)


def pyscf_rhf(pmol):
    from pyscf import scf

    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    assert mf.converged
    return mf


# ---------------------------------------------------------------- anchors


def anchor_full(solver):
    """Active space = full space: zero F asymmetry and E = FCI on the same integrals."""
    from pyscf import fci

    log("\n== anchor 1: active space = full space ==")
    log(
        f"{'system':16s} {'ndet':>6s} {'E(proto)':>18s} {'E(FCI, same ints)':>18s} "
        f"{'dE':>9s} {'dE(RHF)':>9s} {'max|2(F-F^T)|':>14s} {'CI resid':>9s}"
    )
    rows = []
    for xyz, basis in (("water.xyz", "sto-3g"), ("n2.xyz", "sto-3g")):
        mol, ints, c, e_rhf = ferric_setup(xyz, basis)
        nmo, nel = c.shape[1], mol.nelec()
        cas = Cas(ints, 0, nmo, (nel // 2, nel // 2), solver)
        cas.setup_pairs(nmo)
        x = cas.gradient(c)
        # The reference FCI runs on ferric's integrals, so dE tests the
        # solver alone; the full-space FCI energy does not depend on which
        # RHF's orbitals it is expanded in.
        mf = pyscf_rhf_on_ferric_integrals(pyscf_mol(mol, basis), ints)
        fs = fci.FCI(mf)
        fs.conv_tol = 1e-14
        fs.max_cycle = 500
        e_fci = fs.kernel()[0]
        if not fs.converged:
            raise RuntimeError(f"PySCF FCI reference did not converge ({xyz}/{basis})")
        ndet = x["ci"].ci.size
        row = (
            f"{xyz[:-4]}/{basis}",
            ndet,
            x["e"],
            e_fci,
            x["e"] - e_fci,
            e_rhf - mf.e_tot,
            np.abs(x["gmat"]).max(),
            x["ci"].residual,
        )
        rows.append(row)
        log(
            f"{row[0]:16s} {row[1]:6d} {row[2]:18.10f} {row[3]:18.10f} "
            f"{row[4]:9.1e} {row[5]:9.1e} {row[6]:14.1e} {row[7]:9.1e}"
        )
    return rows


def fd_gradient(cas, c, h):
    """5-point central difference of the CAS-CI energy, one rotation at a time."""
    n = len(cas.cls)
    g = np.empty(n)
    for k in range(n):
        e = {}
        for m in (-2, -1, 1, 2):
            kv = np.zeros(n)
            kv[k] = m * h
            e[m] = cas.energy(cas.rotate(c, kv))[0]
        g[k] = (e[-2] - 8.0 * e[-1] + 8.0 * e[1] - e[2]) / (12.0 * h)
    return g


MUTATIONS = ("drop_FA_core", "drop_2rdm", "flip_core_virt", "zero_act_virt")


def anchor_fd(solver, h, bar, rng_seed=7):
    log(
        f"\n== anchor 2: analytic gradient vs Richardson-extrapolated 5-point FD "
        f"(h={h:g}, h/2; bar {bar:.0e}) =="
    )
    out = {}
    for name, xyz, basis, ncore, ncas, nelecas in SYSTEMS:
        mol, ints, c_rhf, _ = ferric_setup(xyz, basis)
        cas = Cas(ints, ncore, ncas, nelecas, solver)
        cas.setup_pairs(c_rhf.shape[1])
        # Generic point: RHF orbitals plus a random rotation of size ~0.05.
        rng = np.random.default_rng(rng_seed)
        c = cas.rotate(c_rhf, 0.05 * rng.standard_normal(len(cas.cls)))
        t0 = time.time()
        g_fd = fd_gradient(cas, c, h)
        g_fd2 = fd_gradient(cas, c, h / 2)
        # Richardson: the 5-point stencil error is h^4 f5/30, so
        # (16 FD(h/2) - FD(h)) / 15 removes it; FD(h) - FD(h/2) is the size
        # of the removed term.
        g_rich = (16.0 * g_fd2 - g_fd) / 15.0
        log(f"{name}: {len(cas.cls)} rotations, FD in {time.time() - t0:.1f} s")
        log(f"  {'gradient':16s} " + " ".join(f"{k:>22s}" for k in CLASSES))
        log(
            f"  {'max|g_fd|':16s} "
            + " ".join(f"{np.abs(g_fd[cas.cls == i]).max():22.2e}" for i in range(3))
        )
        log(
            f"  {'FD(h)-FD(h/2)':16s} "
            + " ".join(
                f"{np.abs((g_fd - g_fd2)[cas.cls == i]).max():22.2e}" for i in range(3)
            )
        )
        res = {}
        for mut in (None,) + MUTATIONS:
            ga = cas.gradient(c, mutation=mut)["g"]
            err = [np.abs((ga - g_rich)[cas.cls == i]).max() for i in range(3)]
            res[mut or "none"] = err
            tag = mut or "analytic"
            verdict = " ".join(
                f"{e:13.2e} {'pass' if e <= bar else 'FAIL':>8s}" for e in err
            )
            log(f"  {tag:16s} {verdict}")
        out[name] = res
    return out


SYSTEMS = (
    ("H2O CAS(4,4)/6-31G", "water.xyz", "6-31g", 3, 4, (2, 2)),
    ("N2 CAS(6,6)/6-31G", "n2.xyz", "6-31g", 4, 6, (3, 3)),
)


def fd_hessian_min_eig(cas, c, h=1e-4):
    """Lowest eigenvalue of the CI-relaxed orbital Hessian, FD of the gradient."""
    n = len(cas.cls)
    hes = np.empty((n, n))
    for k in range(n):
        kv = np.zeros(n)
        kv[k] = h
        hes[:, k] = (
            cas.gradient(cas.rotate(c, kv))["g"] - cas.gradient(cas.rotate(c, -kv))["g"]
        ) / (2 * h)
    hes = 0.5 * (hes + hes.T)
    return np.linalg.eigvalsh(hes)[0], np.diag(hes)


def pyscf_casscf(mf, ncas, nelecas, gtol):
    from pyscf import mcscf

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = gtol
    mc.max_cycle_macro = 200
    mc.fcisolver.conv_tol = 1e-14
    t0 = time.time()
    mc.kernel()
    return mc, time.time() - t0, float(np.linalg.norm(mc.get_grad()))


def pyscf_rhf_on_ferric_integrals(pmol, ints):
    """PySCF RHF whose hcore, overlap, ERIs and E_nuc are ferric's.

    PySCF converges its own orbitals from these integrals; only the integral
    values are shared with the prototype.
    """
    from pyscf import ao2mo, scf

    nao = ints.nao
    assert pmol.nao_nr() == nao
    h, s, enuc = ints.hcore(), ints.overlap(), ints.nuclear_repulsion()
    pmol.incore_anyway = True
    mf = scf.RHF(pmol)
    mf.get_hcore = lambda *a: h
    mf.get_ovlp = lambda *a: s
    mf.energy_nuc = lambda *a: enuc
    mf._eri = ao2mo.restore(8, ints.eri(), nao)
    mf.init_guess = "1e"  # the minao guess assumes PySCF AO order
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    assert mf.converged
    return mf


def anchor_ref(solver, gtol, gtol_pyscf):
    from pyscf.fci import spin_op

    log("\n== anchor 3: converged CASSCF vs pyscf.mcscf.CASSCF ==")
    rows = []
    for name, xyz, basis, ncore, ncas, nelecas in SYSTEMS:
        mol, ints, c_rhf, e_rhf = ferric_setup(xyz, basis)
        cas = Cas(ints, ncore, ncas, nelecas, solver)
        t0 = time.time()
        r = casscf(cas, c_rhf, gtol=gtol)
        t_proto = time.time() - t0
        lam, hdiag_fd = fd_hessian_min_eig(cas, r["c"])
        hd_approx = cas.approx_hdiag(r["x"])
        rel = np.abs(hd_approx - hdiag_fd) / np.abs(hdiag_fd)
        s2 = spin_op.spin_square0(r["x"]["ci"].ci, ncas, nelecas)[0]

        # (a) PySCF CASSCF on ferric's integrals: only convergence differs.
        mf_f = pyscf_rhf_on_ferric_integrals(pyscf_mol(mol, basis), ints)
        mc_f, t_f, g_f = pyscf_casscf(mf_f, ncas, nelecas, gtol_pyscf)
        # E - E_min <= |g|^2 / (2 lambda_min) near a minimum, for each code,
        # plus PySCF's energy-change tolerance.
        # PySCF's packed gradient omits this script's factor 2 on some
        # entries, so 2*g_f bounds its norm in this script's convention.
        if not (np.isfinite(lam) and lam > 0.0):
            raise RuntimeError(
                f"{name}: lowest orbital-Hessian eigenvalue {lam} is not a positive "
                "finite curvature, so the converged point is not a minimum and the "
                "energy bar is undefined"
            )
        bar = (r["gnorm"] ** 2 + (2.0 * g_f) ** 2) / (2.0 * lam) + mc_f.conv_tol
        if not np.isfinite(bar):
            raise RuntimeError(f"{name}: energy bar is not finite ({bar})")
        # (b) PySCF end to end (its own integrals); integral offset included.
        mf_p = pyscf_rhf(pyscf_mol(mol, basis))
        mc_p, t_p, g_p = pyscf_casscf(mf_p, ncas, nelecas, gtol_pyscf)
        row = {
            "system": name,
            "e_proto": r["e"],
            "e_pyscf_ferric_ints": mc_f.e_tot,
            "de_same_ints": r["e"] - mc_f.e_tot,
            "bar": bar,
            "rhf_same_ints": e_rhf - mf_f.e_tot,
            "e_pyscf_native": mc_p.e_tot,
            "de_native": r["e"] - mc_p.e_tot,
            "rhf_native": e_rhf - mf_p.e_tot,
            "g_proto": r["gnorm"],
            "g_pyscf_ferric_ints": g_f,
            "g_pyscf_native": g_p,
            "pyscf_converged": (mc_f.converged, mc_p.converged),
            "lam_min": lam,
            "iters": r["iters"],
            "s2": s2,
            "hdiag_relerr_median": float(np.median(rel)),
            "hdiag_relerr_max": float(rel.max()),
            "t_proto": t_proto,
            "t_pyscf": (t_f, t_p),
        }
        rows.append(row)
        for k, v in row.items():
            log(f"  {k:22s} {v}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("full", "fd", "ref"))
    ap.add_argument("--fd-step", type=float, default=1e-3)
    ap.add_argument("--fd-bar", type=float, default=1e-8)
    ap.add_argument("--gtol", type=float, default=1e-7)
    ap.add_argument("--gtol-pyscf", type=float, default=1e-6)
    a = ap.parse_args()
    log(f"ferric build {ferric.__build__}")
    solver = PyscfFci()
    ok = True
    if a.only in (None, "full"):
        for row in anchor_full(solver):
            # Same integrals: dE is the solver difference (full_space_bar).
            # |2(F-F^T)| is first order in the CI error,
            # bounded by 10 x the CI residual.
            ok &= abs(row[4]) <= full_space_bar(row[3]) and row[6] <= max(
                10.0 * row[7], 1e-12
            )
    if a.only in (None, "fd"):
        res = anchor_fd(solver, a.fd_step, a.fd_bar)
        for name, r in res.items():
            ok &= max(r["none"]) <= a.fd_bar
            for mut in MUTATIONS:
                if max(r[mut]) <= a.fd_bar:
                    log(f"MUTATION NOT CAUGHT: {name} {mut}")
                    ok = False
    if a.only in (None, "ref"):
        for row in anchor_ref(solver, a.gtol, a.gtol_pyscf):
            ok &= bool(row["pyscf_converged"][0])
            ok &= abs(row["de_same_ints"]) <= row["bar"]
    log("\nRESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
