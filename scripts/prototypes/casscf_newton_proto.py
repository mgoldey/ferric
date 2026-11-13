#!/usr/bin/env python
"""Second-order orbital step for two-step CASSCF: prototype and iteration counts.

Decides the optimizer of ``crates/ferric-ci/src/casscf.rs`` by measured
macro-iteration counts (round-2 task 14). Integrals are ferric's
(``ferric.AoIntegrals``: hcore, ERIs, nuclear repulsion) and the start is
ferric's RHF, as in ``casscf_proto.py``; contractions use PySCF's ``ao2mo``
and ``dot_eri_dm`` on those integrals, and the CAS-CI is ``pyscf.fci``.
Nothing about speed may be quoted from this script; it counts macro
iterations and Hessian-vector products.

Orbital Hessian at fixed RDMs. ``C(e) = C exp(e X)``, ``X`` antisymmetric
over the nonredundant pairs. The frame derivative of the gradient matrix
``G = 2 (F - F^T)`` (each quantity expressed in the rotated orbitals) is

    dF^I = F^I X - X F^I + C^T (J - K/2)[dD^c] C,  dD^c = 2 (C X)_c C_c^T + transpose
    dF^A = F^A X - X F^A + C^T (J - K/2)[dD^a] C,  dD^a = (C X)_a g C_a^T + transpose
    d(pu|vw) = sum_r X_rp (ru|vw) + X_ru (pr|vw) + X_rv (pu|rw) + X_rw (pu|rv)
    dF_pi = 2 (dF^I + dF^A)_pi,  dF_pt = sum_u dF^I_pu g_ut + sum_uvw d(pu|vw) G_tuvw
    Hf x  = pack 2 (dF - dF^T)

with ``(pr|vw)`` = ppaa and ``(pu|rw)`` = papa. From
``exp(eX) exp(dA) = exp(eX + dA + ed [X, A]/2 + ...)``, the symmetric
Hessian of ``E(C exp(K))`` at ``K = 0`` is ``H x = Hf x - pack [G, X] / 2``.

Optimizers compared (all two-step: one CAS-CI per macro iteration):

* ``bfgs``: the shipped inverse BFGS from the approximate diagonal.
* ``ah``: augmented-Hessian step on the exact fixed-RDM Hessian, Davidson
  on ``[[0, g^T], [g, H]]``, trust radius from the energy ratio.
* ``sr1-m``: ``ah`` with multi-secant SR1 corrections of ``H`` from the last
  ``m`` steps (unstable for ``m >= 3``).
* ``scale``: ``ah`` with each unclipped step lengthened by the previous
  actual/predicted ratio clamped to ``[1, 2]`` (the Rust step measures the
  curvature ratio along the step instead; see theory.md);
  ``-eta.x`` variants stop the Davidson at a constant relative residual.

Measured counts and the choice: ``crates/ferric-ci/theory.md``.

Usage (one heavy op at a time):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 PYTHONPATH=<shim> \\
    /home/matt/qc/ferric/.venv/bin/python scripts/prototypes/casscf_newton_proto.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.scf import hf
from scipy.linalg import expm

import ferric

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOLS = ROOT / "testdata" / "molecules"

# name, xyz, basis, n_core, n_active, (n_alpha, n_beta)
SYSTEMS = {
    "h2o": ("water.xyz", "6-31g", 3, 4, (2, 2)),
    "n2": ("n2.xyz", "6-31g", 4, 6, (3, 3)),
    "o2": ("o2.xyz", "6-31g", 4, 6, (5, 3)),
    "benzene": ("benzene.xyz", "cc-pvdz", 18, 6, (3, 3)),
    "n2-cas1010": ("n2.xyz", "cc-pvdz", 2, 10, (5, 5)),
}

VERBOSE = False


def log(*a):
    print(*a, flush=True)


class Cas:
    def __init__(self, name, mutate=None):
        xyz, basis, nc, na, nel = SYSTEMS[name]
        text = (MOLS / xyz).read_text()
        self.fmol = ferric.Molecule.from_xyz_string(text)
        bs = ferric.BasisSet.bundled(basis)
        ints = ferric.AoIntegrals(self.fmol, bs)
        self.nao = ints.nao
        self.h = np.asarray(ints.hcore())
        self.s = np.asarray(ints.overlap())
        self.enuc = ints.nuclear_repulsion()
        self.eri8 = ao2mo.restore(8, np.asarray(ints.eri()), self.nao)
        r = ferric.run_rhf(self.fmol, bs, energy_conv=1e-12, density_conv=1e-10)
        assert r.converged
        self.c0 = np.asarray(r.mo_coefficients())
        self.e_rhf = r.energy
        self.basis = basis
        self.nc, self.na, self.nel = nc, na, nel
        self.nmo = self.c0.shape[1]
        self.mutate = mutate
        act = range(nc, nc + na)
        virt = range(nc + na, self.nmo)
        pairs = [(t, i) for t in act for i in range(nc)]
        pairs += [(a, i) for a in virt for i in range(nc)]
        pairs += [(a, t) for a in virt for t in act]
        self.pairs = np.array(pairs, dtype=int)
        self.ci0 = None
        self.n_hvp = 0
        self.n_eval = 0

    # -- packing
    def pack(self, m):
        return m[self.pairs[:, 0], self.pairs[:, 1]]

    def unpack(self, v):
        k = np.zeros((self.nmo, self.nmo))
        k[self.pairs[:, 0], self.pairs[:, 1]] = v
        k[self.pairs[:, 1], self.pairs[:, 0]] = -v
        return k

    def rotate(self, c, v):
        return c @ expm(self.unpack(v))

    def jk(self, dms):
        vj, vk = hf.dot_eri_dm(self.eri8, np.asarray(dms), hermi=1)
        return vj, vk

    # -- energy functional
    def evaluate(self, c, rdms=None):
        """Energy, gradient and intermediates. With ``rdms`` given, the RDMs
        are held fixed (no CI solve) and the energy is not computed."""
        self.n_eval += rdms is None
        nc, na = self.nc, self.na
        a = slice(nc, nc + na)
        cc, ca = c[:, :nc], c[:, a]
        dc = 2.0 * cc @ cc.T
        vj, vk = self.jk([dc])
        fi_ao = self.h + vj[0] - 0.5 * vk[0]
        ecore = self.enuc + 0.5 * np.sum(dc * (self.h + fi_ao))
        ppaa = ao2mo.general(self.eri8, (c, c, ca, ca), compact=False).reshape(
            self.nmo, self.nmo, na, na
        )
        papa = ao2mo.general(self.eri8, (c, ca, c, ca), compact=False).reshape(
            self.nmo, na, self.nmo, na
        )
        paaa = ppaa[:, a]
        e = None
        if rdms is None:
            s = fci.direct_spin1.FCI()
            s.verbose = 0
            s.conv_tol = 1e-14
            s.max_cycle = 500
            s.pspace_size = 0
            h1 = ca.T @ fi_ao @ ca
            eact, civ = s.kernel(h1, paaa[a].copy(), na, self.nel, ci0=self.ci0)
            self.ci0 = civ
            dm1, dm2 = s.make_rdm12(civ, na, self.nel)
            e = ecore + eact
        else:
            dm1, dm2 = rdms
        vj, vk = self.jk([ca @ dm1 @ ca.T])
        fa_ao = vj[0] - 0.5 * vk[0]
        fi, fa = c.T @ fi_ao @ c, c.T @ fa_ao @ c
        f = np.zeros_like(fi)
        f[:, :nc] = 2.0 * (fi[:, :nc] + fa[:, :nc])
        f[:, a] = fi[:, a] @ dm1 + np.einsum("puvw,tuvw->pt", paaa, dm2)
        gm = 2.0 * (f - f.T)
        return dict(
            e=e,
            g=self.pack(gm),
            gm=gm,
            c=c,
            fi=fi,
            fa=fa,
            f=f,
            dm1=dm1,
            dm2=dm2,
            ppaa=ppaa,
            papa=papa,
        )

    def hvp_frame(self, pt, x):
        """Frame derivative of the packed gradient at fixed RDMs."""
        nc, na = self.nc, self.na
        a = slice(nc, nc + na)
        c, dm1, dm2 = pt["c"], pt["dm1"], pt["dm2"]
        xm = self.unpack(x)
        cx = c @ xm
        ddc = 2.0 * cx[:, :nc] @ c[:, :nc].T
        ddc += ddc.T
        dda = cx[:, a] @ dm1 @ c[:, a].T
        dda += dda.T
        vj, vk = self.jk([ddc, dda])
        gi = c.T @ (vj[0] - 0.5 * vk[0]) @ c
        ga = c.T @ (vj[1] - 0.5 * vk[1]) @ c
        dfi = pt["fi"] @ xm - xm @ pt["fi"] + gi
        dfa = pt["fa"] @ xm - xm @ pt["fa"] + ga
        ppaa, papa = pt["ppaa"], pt["papa"]
        paaa = ppaa[:, a]
        xa = xm[:, a]
        aa = np.einsum("ru,prvw->puvw", xa, ppaa)
        if self.mutate == "flip_aa":
            aa = -aa
        dpaaa = (
            np.einsum("rp,ruvw->puvw", xm, paaa)
            + aa
            + np.einsum("rv,purw->puvw", xa, papa)
            + np.einsum("rw,purv->puvw", xa, papa)
        )
        df = np.zeros_like(dfi)
        df[:, :nc] = 2.0 * (dfi[:, :nc] + dfa[:, :nc])
        df[:, a] = dfi[:, a] @ dm1 + np.einsum("puvw,tuvw->pt", dpaaa, dm2)
        return self.pack(2.0 * (df - df.T)), xm

    def hvp(self, pt, x):
        self.n_hvp += 1
        hf_x, xm = self.hvp_frame(pt, x)
        gm = pt["gm"]
        return hf_x - 0.5 * self.pack(gm @ xm - xm @ gm)

    def approx_hdiag(self, pt):
        ft = pt["fi"] + pt["fa"]
        fg = pt["f"]
        nc = self.nc
        h = np.empty(len(self.pairs))
        for n, (p, q) in enumerate(self.pairs):
            if q >= nc:  # (a, t)
                h[n] = 2.0 * pt["dm1"][q - nc, q - nc] * ft[p, p] - 2.0 * fg[q, q]
            elif p >= nc + self.na:  # (a, i)
                h[n] = 4.0 * (ft[p, p] - ft[q, q])
            else:  # (t, i)
                gtt = pt["dm1"][p - nc, p - nc]
                h[n] = (
                    4.0 * (ft[p, p] - ft[q, q]) + 2.0 * gtt * ft[q, q] - 2.0 * fg[p, p]
                )
        return h


# ----------------------------------------------------------------- anchors


def anchor_hvp(name, h=1e-3, seed=3):
    """Hf x against a central difference of the fixed-RDM frame gradient.

    Central difference error h^2/6 g''' is estimated by FD(h) - FD(h/2)
    (Richardson: the extrapolated value removes 3/4 of it)."""
    out = {}
    for mut in (None, "flip_aa"):
        cas = Cas(name, mutate=mut)
        rng = np.random.default_rng(seed)
        c = cas.rotate(cas.c0, 0.05 * rng.standard_normal(len(cas.pairs)))
        pt = cas.evaluate(c)
        rdms = (pt["dm1"], pt["dm2"])
        x = rng.standard_normal(len(cas.pairs))
        x /= np.linalg.norm(x)

        def fd(step):
            gp = cas.evaluate(cas.rotate(c, step * x), rdms)["g"]
            gm = cas.evaluate(cas.rotate(c, -step * x), rdms)["g"]
            return (gp - gm) / (2 * step)

        f1, f2 = fd(h), fd(h / 2)
        rich = (4 * f2 - f1) / 3
        an, xm = cas.hvp_frame(pt, x)
        err = np.abs(an - rich).max()
        trunc = np.abs(f1 - f2).max()
        # symmetry of H: y.Hx = x.Hy
        y = rng.standard_normal(len(cas.pairs))
        cas.mutate = mut
        sym = abs(y @ cas.hvp(pt, x) - x @ cas.hvp(pt, y))
        out[mut or "none"] = (err, trunc, np.abs(an).max(), sym)
        log(
            f"{name} hvp {mut or 'analytic':8s}: max|Hf x - FD| {err:.2e}, "
            f"FD(h)-FD(h/2) {trunc:.2e}, max|Hf x| {np.abs(an).max():.2e}, "
            f"|y.Hx - x.Hy| {sym:.2e}"
        )
    return out


# ----------------------------------------------------------------- optimizers


def converged(e, e_old, gn, gtol, etol):
    return e_old is not None and abs(e - e_old) < etol and gn < gtol


def run_bfgs(cas, gtol=1e-7, etol=1e-12, max_iter=300, max_step=0.2, hfloor=0.05):
    c = cas.c0.copy()
    e_old = g_old = s_old = hinv = None
    for it in range(1, max_iter + 1):
        pt = cas.evaluate(c)
        e, g = pt["e"], pt["g"]
        gn = np.linalg.norm(g)
        if converged(e, e_old, gn, gtol, etol):
            return dict(e=e, it=it, gn=gn, conv=True)
        hd = np.maximum(np.abs(cas.approx_hdiag(pt)), hfloor)
        if hinv is None or (e_old is not None and e > e_old + 1e-10):
            hinv = np.diag(1.0 / hd)
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
    return dict(e=e, it=max_iter, gn=gn, conv=False)


def ah_davidson(hop, g, precond, tol, max_iter=40):
    """Lowest eigenpair of [[0, g^T], [g, H]]; returns step s and the
    predicted energy change g.s + s.H s / 2 (from the subspace)."""
    gn = np.linalg.norm(g)
    # Start: the steepest-descent direction preconditioned.
    v0 = -g / precond
    vs, hvs = [], []
    v = v0 / np.linalg.norm(v0)
    for it in range(1, max_iter + 1):
        for u in vs:
            v -= (u @ v) * u
        for u in vs:
            v -= (u @ v) * u
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            break
        v /= nv
        vs.append(v)
        hvs.append(hop(v))
        m = len(vs)
        vmat = np.array(vs)
        hmat = vmat @ np.array(hvs).T
        hmat = 0.5 * (hmat + hmat.T)
        gsub = vmat @ g
        aug = np.zeros((m + 1, m + 1))
        aug[0, 1:] = gsub
        aug[1:, 0] = gsub
        aug[1:, 1:] = hmat
        w, z = np.linalg.eigh(aug)
        lam, zz = w[0], z[:, 0]
        if abs(zz[0]) < 1e-8:
            raise RuntimeError("AH: vanishing leading component")
        y = zz[1:] / zz[0]
        s = vmat.T @ y
        hs = np.array(hvs).T @ y
        # residual of (H - lam) s + g = 0
        r = hs - lam * s + g
        rn = np.linalg.norm(r)
        if rn < tol * gn or it == max_iter:
            pred = g @ s + 0.5 * s @ hs
            return s, pred, it, rn, lam, hs
        d = precond - lam
        d = np.where(np.abs(d) < 1e-8, 1e-8, d)
        v = -r / d
    pred = g @ s + 0.5 * s @ hs
    return s, pred, it, rn, lam, hs


def run_ah(
    cas,
    gtol=1e-7,
    etol=1e-12,
    max_iter=200,
    radius=0.4,
    max_radius=1.0,
    hfloor=0.05,
    tol_rel=0.1,
    forcing="quad",
    max_dav=40,
    max_step=None,
    secant=0,
    scale=False,
):
    """``secant = m > 0``: the Davidson operator is ``H0 + R M^-1 R^T`` with
    ``R = [y_i - H0_i s_i]`` and ``M = sym(R^T S)`` over the last ``m`` steps
    (multi-secant SR1 on top of the exact fixed-RDM Hessian; ``H0_i s_i`` is
    the product the step's own Davidson returned, so no extra products)."""
    c = cas.c0.copy()
    pt = cas.evaluate(c)
    e_old = None
    it = 1
    n_reject = 0
    mem_s, mem_r = [], []
    alpha = 1.0
    while True:
        e, g = pt["e"], pt["g"]
        gn = np.linalg.norm(g)
        if converged(e, e_old, gn, gtol, etol):
            return dict(e=e, it=it, gn=gn, conv=True, rej=n_reject)
        if it >= max_iter:
            return dict(e=e, it=it, gn=gn, conv=False, rej=n_reject)
        pre = np.maximum(np.abs(cas.approx_hdiag(pt)), hfloor)
        tol = min(tol_rel, gn) if forcing == "quad" else tol_rel
        corr = None
        if mem_s:
            smat, rmat = np.array(mem_s).T, np.array(mem_r).T
            mm = rmat.T @ smat
            mm = 0.5 * (mm + mm.T)
            w, u = np.linalg.eigh(mm)
            keep = np.abs(w) > 1e-8 * np.abs(w).max()
            minv = (u[:, keep] / w[keep]) @ u[:, keep].T
            corr = (rmat, minv)

        def hop(v, corr=corr):
            hv = cas.hvp(pt, v)
            if corr is not None:
                hv = hv + corr[0] @ (corr[1] @ (corr[0].T @ v))
            return hv

        def hop0(v, corr=corr):
            return hop(v, corr)

        s, pred, nd, rn, lam, hs = ah_davidson(hop0, g, pre, tol, max_iter=max_dav)
        # H0 s (without the correction) for the secant pair.
        h0s = hs - (corr[0] @ (corr[1] @ (corr[0].T @ s)) if corr is not None else 0.0)
        if scale:
            s, hs, h0s = alpha * s, alpha * hs, alpha * h0s
        sn = np.linalg.norm(s)
        f = radius / sn
        if max_step is not None:
            f = min(f, max_step / np.abs(s).max())
        if f < 1.0:
            s, hs, sn = f * s, f * hs, f * sn
            h0s = f * h0s
            pred = g @ s + 0.5 * s @ hs
        ci_keep = cas.ci0
        while True:
            c_new = cas.rotate(c, s)
            pt_new = cas.evaluate(c_new)
            it += 1
            de = pt_new["e"] - e
            ratio = de / pred if pred < 0 else 0.0
            if de > 1e-10 and it < max_iter:
                # Reject: back to c with half the step (pred scales as s and s^2).
                n_reject += 1
                s, hs, sn = 0.5 * s, 0.5 * hs, 0.5 * sn
                h0s = 0.5 * h0s
                radius = sn
                pred = g @ s + 0.5 * s @ hs
                cas.ci0 = ci_keep
                continue
            break
        if VERBOSE:
            log(
                f"  it {it:3d} E {pt_new['e']:.12f} |g| {gn:.2e} |s| {sn:.3f} "
                f"radius {radius:.3f} ratio {ratio:6.3f} dav {nd} lam {lam:.2e}"
            )
        # Previous ratio as the next step's length factor: the fixed-RDM
        # Hessian misses the CI response, which lowers the curvature.
        alpha = min(max(ratio, 1.0), 2.0) if (scale and f >= 1.0) else 1.0
        if secant:
            r = (pt_new["g"] - g) - h0s
            if abs(r @ s) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
                mem_s.append(s)
                mem_r.append(r)
                mem_s, mem_r = mem_s[-secant:], mem_r[-secant:]
        if ratio < 0.25:
            radius = max(0.5 * sn, 1e-3)
        elif ratio > 0.75 and sn >= 0.99 * radius:
            radius = min(2.0 * radius, max_radius)
        c, e_old, pt = c_new, e, pt_new


def pyscf_counts(cas, start):
    """pyscf.mcscf.CASSCF macro iterations at gen_casscf_refs.py settings."""
    atoms = list(zip(cas.fmol.symbols(), cas.fmol.coords_bohr()))
    pmol = gto.M(atom=atoms, unit="Bohr", basis=cas.basis, verbose=0)
    assert pmol.nao_nr() == cas.nao
    pmol.incore_anyway = True
    mf = scf.RHF(pmol)
    mf.get_hcore = lambda *a: cas.h
    mf.get_ovlp = lambda *a: cas.s
    mf.energy_nuc = lambda *a: cas.enuc
    mf._eri = cas.eri8
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    mc = mcscf.CASSCF(mf, cas.na, cas.nel)
    mc.verbose = 0
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = 300
    mc.fcisolver.conv_tol = 1e-14
    stats = {}

    def cb(envs):
        stats["macro"] = envs["imacro"] + 1
        stats["totmicro"] = envs.get("totmicro")
        stats["totinner"] = envs.get("totinner")

    mc.callback = cb
    mo = cas.c0 if start == "ferric" else None
    mc.kernel(mo)
    return dict(
        e=mc.e_tot,
        conv=mc.converged,
        macro=stats.get("macro"),
        totmicro=stats.get("totmicro"),
        totinner=stats.get("totinner"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="h2o,n2,o2,benzene")
    ap.add_argument("--only", choices=("hvp", "count", "pyscf"))
    ap.add_argument("--variants", default="bfgs,ah")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = a.verbose
    log(f"ferric build {ferric.__build__}")
    names = a.systems.split(",")
    ok = True
    if a.only in (None, "hvp"):
        for name in names:
            if name == "benzene":
                continue
            r = anchor_hvp(name)
            ok &= r["none"][0] < 10 * r["none"][1] + 1e-9
            ok &= r["flip_aa"][0] > 1e3 * (r["none"][1] + 1e-9)
    variants = {
        "bfgs": None,
        "ah": {},
        "ah-cap": dict(max_step=0.2),
        "ah-eta.1": dict(forcing="const", tol_rel=0.1, max_step=0.2),
        "ah-eta.3": dict(forcing="const", tol_rel=0.3, max_step=0.2),
        "ah-eta.1-d8": dict(forcing="const", tol_rel=0.1, max_dav=8, max_step=0.2),
        "ah-q-d8": dict(max_dav=8, max_step=0.2),
        "sr1-1": dict(secant=1, max_step=0.2),
        "scale": dict(scale=True, max_step=0.2),
        "scale-eta.1": dict(scale=True, max_step=0.2, forcing="const", tol_rel=0.1),
        "scale-eta.3": dict(scale=True, max_step=0.2, forcing="const", tol_rel=0.3),
        "sr1-3": dict(secant=3, max_step=0.2),
        "sr1-5": dict(secant=5, max_step=0.2),
        "sr1-3-eta.1": dict(secant=3, max_step=0.2, forcing="const", tol_rel=0.1),
    }
    if a.only in (None, "count"):
        for name in names:
            for opt in a.variants.split(","):
                cas = Cas(name)
                t0 = time.time()
                kw = variants[opt]
                r = run_bfgs(cas) if kw is None else run_ah(cas, **kw)
                log(
                    f"{name:8s} {opt:12s} E {r['e']:.10f} conv {r['conv']} macro {r['it']} "
                    f"evals {cas.n_eval} hvp {cas.n_hvp} |g| {r['gn']:.1e} "
                    f"rejects {r.get('rej', '-')} ({time.time() - t0:.1f} s)"
                )
    if a.only in (None, "pyscf"):
        for name in names:
            cas = Cas(name)
            for start in ("pyscf-rhf", "ferric"):
                r = pyscf_counts(cas, start)
                log(
                    f"{name:8s} pyscf from {start:9s}: E {r['e']:.10f} conv {r['conv']} "
                    f"macro {r['macro']} totmicro {r['totmicro']} totinner {r['totinner']}"
                )
    log("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
