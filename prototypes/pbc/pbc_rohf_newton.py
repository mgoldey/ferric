"""Second-order (trust-region augmented-Hessian) ROHF / ROKS on the INJECTED (periodic) path — Iteration 24.

Everything here is driven by the three things ferric's injected ROHF path (ferric-scf rohf.rs::solve_rohf_injected
with a PeriodicInjection) already has, and nothing molecular:
  * S, h, E_nn, the exact-exchange fraction a and the Madelung shift v_M (kshift) of the stage;
  * a J/K contraction for ANY symmetric AO density  jk(D) -> (J[D], K[D])  (the injected J/K builder);
  * for ROKS, the polarized XC builder  (D_a, D_b) -> (E_xc, V_a, V_b)  (XcBuilder::build_polarized).
The XC response is a central finite difference of that builder (no f_xc kernel needed on the periodic grid).

Parametrization (identical to ferric's rohf_newton / rohf_ah): C <- C U, U = exp(kappa) or the Cayley unitary,
kappa antisymmetric in the MO basis with ONE free parameter per non-redundant pair (p, q), kappa[p, q] = x,
kappa[q, p] = -x, p the "less occupied" index.  Packing order = ferric's: vc (virt x closed, row-major), then
vo (virt x open), then oc (open x closed).  Occupations n_a = 1 on closed+open, n_b = 1 on closed.

Exact first and second derivatives of E(C e^kappa) at kappa = 0 (derivation):
  D_s(kappa) (MO) = e^kappa n_s e^-kappa = n_s + [kappa, n_s] + 1/2 [kappa, [kappa, n_s]] + ...
  E(kappa) = E0 + sum_s tr f_s [kappa, n_s] + 1/2 { sum_s tr f_s [kappa, [kappa, n_s]] + sum_s tr dF_s[kappa] [kappa, n_s] }
  with f_s = C^T F_s C and dF_s[kappa] = C^T (J[dD_a + dD_b] - a (K[dD_s] + v_M S dD_s S) + dV_xc^s) C,
  dD_s = C [kappa, n_s] C^T.  With d/dx tr(G kappa) = G[q, p] - G[p, q]:
    gradient     G1 = sum_s (n_s f_s - f_s n_s)
    Hessian x    G2 = sum_s 1/2 (n_s A_s - A_s n_s) + 1/2 (X_s f_s - f_s X_s) + (n_s dF_s - dF_s n_s),
                 A_s = [f_s, X], X_s = [X, n_s], X = unpack(x).
  Per block the TRUE gradient is 2 (f_a + f_b)_vc, 2 (f_a)_vo, 2 (f_b)_oc, i.e. EXACTLY 2 x ferric's packed
  gradient_blocks in all three blocks (checked by FD in run_rohf_newton.py 'derivs').

`ferric_hvp` mirrors crates/ferric-scf/src/rohf_newton.rs::hessian_matvec (per-spin Fock DIAGONAL only, 2e response
in ferric's packed half-scale): 2 * ferric_hvp is compared against the exact Hessian.

Drivers:
  trah(...)        level-shifted AH + Fletcher trust region (ferric crate::trah semantics: radius0 0.4, rho < 0
                   reject, <= 0.25 shrink 0.7, > 0.75 grow 1.2, alpha bisection on log alpha in [1, 1000]) on the
                   exact gradient/Hessian, with a rejected step re-solved immediately from the SAME point in the
                   SAME Krylov subspace (no J/K spent); stops at max|g_packed| < gtol.
  ferric_ah_mirror(...)  the molecular rohf.rs AH branch as written: DIIS until err_max < ah_trigger (it > 3), then
                   every iteration diagonalize F_eff (aufbau), take the gradient/Hessian from the STALE per-spin Fock
                   in that basis, approximate diag-Fock Hessian, AH (alpha = 1), componentwise clip 0.2, no energy test.

Cost accounting: `Injected.n_jk` counts jk(D) calls (one per-spin contraction returning J and K; a Fock build is 2,
one Hessian matvec is 2), `n_fock` Fock builds, `n_xc` XC-builder calls (ROKS: 1 per Fock, 2 per matvec).
Units Bohr / Hartree."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_uks as U  # noqa: E402


class Injected:
    """What ferric's injected ROHF/ROKS path provides: S, h, E_nn, jk(D), a, v_M, and the polarized XC builder."""

    def __init__(self, S, h, enn, jk, xc="HF", grid=None, kshift=0.0, fxc_step=1e-4):
        self.S, self.h, self.enn, self.jk, self.xc, self.kshift = (
            S,
            h,
            enn,
            jk,
            xc,
            kshift,
        )
        self.hyb = U.hybrid_fraction(xc)
        self.grid = None if str(xc).upper() == "HF" else grid
        self.fxc_step = fxc_step
        self.n_jk = self.n_fock = self.n_xc = self.n_hvp = 0
        self._ref = None

    def counts(self):
        return dict(jk=self.n_jk, fock=self.n_fock, xc=self.n_xc, hvp=self.n_hvp)

    def fock(self, Da, Db):
        self.n_jk += 2
        self.n_fock += 1
        if self.grid is not None:
            self.n_xc += 1
        e, Fa, Fb, _ = U._fock_energy(
            self.S,
            self.h,
            self.jk,
            self.enn,
            Da,
            Db,
            self.grid,
            self.xc,
            self.hyb,
            self.kshift,
        )
        self._ref = (Da, Db)
        return e, Fa, Fb

    def response(self, dDa, dDb):
        """Linear response of (F_a, F_b) to (dD_a, dD_b) at the density of the last fock() call."""
        S, a, ks = self.S, self.hyb, self.kshift
        Ja, Ka = self.jk(dDa)
        Jb, Kb = self.jk(dDb)
        self.n_jk += 2
        J = Ja + Jb
        dFa = J - a * (Ka + ks * S @ dDa @ S)
        dFb = J - a * (Kb + ks * S @ dDb @ S)
        if self.grid is not None:
            Da, Db = self._ref
            m = max(np.abs(dDa).max(), np.abs(dDb).max())
            if m > 0:
                t = self.fxc_step / m
                _, Vpa, Vpb = U.eval_vxc_uks(
                    self.grid, Da + t * dDa, Db + t * dDb, self.xc
                )
                _, Vma, Vmb = U.eval_vxc_uks(
                    self.grid, Da - t * dDa, Db - t * dDb, self.xc
                )
                self.n_xc += 2
                dFa = dFa + (Vpa - Vma) / (2 * t)
                dFb = dFb + (Vpb - Vmb) / (2 * t)
        return dFa, dFb


# ------------------------------------------------------------------------------------------- MO-space helpers
def pairs(n, nd, no):
    na = nd + no
    vc = [(p, q) for p in range(na, n) for q in range(nd)]
    vo = [(p, q) for p in range(na, n) for q in range(nd, na)]
    oc = [(p, q) for p in range(nd, na) for q in range(nd)]
    return vc + vo + oc, (len(vc), len(vo), len(oc))


def occs(n, nd, no):
    na_, nb_ = np.zeros(n), np.zeros(n)
    na_[: nd + no] = 1.0
    nb_[:nd] = 1.0
    return na_, nb_


def unpack(x, P, n):
    K = np.zeros((n, n))
    for v, (p, q) in zip(x, P):
        K[p, q] = v
        K[q, p] = -v
    return K


def extract(G, P):
    return np.array([G[q, p] - G[p, q] for p, q in P])


def dens(C, nd, no):
    Db = C[:, :nd] @ C[:, :nd].T
    return Db + C[:, nd : nd + no] @ C[:, nd : nd + no].T, Db


def cayley(K):
    n = K.shape[0]
    return np.linalg.solve(np.eye(n) - 0.5 * K, np.eye(n) + 0.5 * K)


class Point:
    """Energy, Focks, exact gradient and exact Hessian-vector product at one determinant C."""

    def __init__(self, sysm, C, nd, no):
        self.sys, self.C, self.nd, self.no = sysm, C, nd, no
        n = C.shape[1]
        self.n = n
        self.P, self.blocks = pairs(n, nd, no)
        self.na, self.nb = occs(n, nd, no)
        self.Da, self.Db = dens(C, nd, no)
        self.e, self.Fa, self.Fb = sysm.fock(self.Da, self.Db)
        self.fa, self.fb = C.T @ self.Fa @ C, C.T @ self.Fb @ C
        G1 = sum(
            np.diag(ns) @ f - f @ np.diag(ns)
            for ns, f in ((self.na, self.fa), (self.nb, self.fb))
        )
        self.g = extract(G1, self.P)

    def g_packed(self):
        """ferric's gradient_blocks packing (half the true gradient)."""
        return 0.5 * self.g

    def hvp(self, x):
        self.sys.n_hvp += 1
        C, n = self.C, self.n
        X = unpack(x, self.P, n)
        Xs = [X @ np.diag(ns) - np.diag(ns) @ X for ns in (self.na, self.nb)]
        dFa, dFb = self.sys.response(C @ Xs[0] @ C.T, C @ Xs[1] @ C.T)
        G = np.zeros((n, n))
        for ns, f, Xn, dF in (
            (self.na, self.fa, Xs[0], C.T @ dFa @ C),
            (self.nb, self.fb, Xs[1], C.T @ dFb @ C),
        ):
            N = np.diag(ns)
            A = f @ X - X @ f
            G += 0.5 * (N @ A - A @ N) + 0.5 * (Xn @ f - f @ Xn) + (N @ dF - dF @ N)
        return extract(G, self.P)

    def diag(self):
        """Hessian-diagonal approximation (TRUE scale) for the Davidson preconditioner: 2 x ferric's per-spin gaps."""
        fa, fb = np.diag(self.fa), np.diag(self.fb)
        nd, na = self.nd, self.nd + self.no
        out = []
        for p, q in self.P:
            if p >= na and q < nd:
                out.append(2 * ((fa[p] + fb[p]) - (fa[q] + fb[q])))
            elif p >= na:
                out.append(2 * (fa[p] - fa[q]))
            else:
                out.append(2 * (fb[p] - fb[q]))
        return np.array(out)

    def ferric_hvp(self, x):
        """Mirror of rohf_newton.rs::hessian_matvec (ferric packed scale): 2e response on the three blocks + per-spin
        Fock-DIAGONAL gap terms only."""
        self.sys.n_hvp += 1
        C, n, nd, na = self.C, self.n, self.nd, self.nd + self.no
        ddA, ddB = np.zeros((n, n)), np.zeros((n, n))
        for v, (p, q) in zip(x, self.P):
            if p >= na and q < nd:
                ddA[p, q] = ddA[q, p] = ddB[p, q] = ddB[q, p] = v
            elif p >= na:
                ddA[p, q] = ddA[q, p] = v
            else:
                ddB[p, q] = ddB[q, p] = v
        dFa, dFb = self.sys.response(C @ ddA @ C.T, C @ ddB @ C.T)
        da, db = C.T @ dFa @ C, C.T @ dFb @ C
        fa, fb = np.diag(self.fa), np.diag(self.fb)
        out = []
        for v, (p, q) in zip(x, self.P):
            if p >= na and q < nd:
                out.append(
                    da[p, q] + db[p, q] + ((fa[p] + fb[p]) - (fa[q] + fb[q])) * v
                )
            elif p >= na:
                out.append(da[p, q] + (fa[p] - fa[q]) * v)
            else:
                out.append(db[p, q] + (fb[p] - fb[q]) * v)
        return np.array(out)

    def hessian(self, which="exact"):
        m = len(self.P)
        f = self.hvp if which == "exact" else self.ferric_hvp
        H = np.column_stack([f(np.eye(m)[i]) for i in range(m)])
        return H

    def rotated(self, kappa, use_cayley=True):
        K = unpack(kappa, self.P, self.n)
        if use_cayley:
            return self.C @ cayley(K)
        from scipy.linalg import expm

        return self.C @ expm(K)


# ------------------------------------------------------------------------------------ AH with a shared subspace
class AHSubspace:
    """Krylov/Davidson subspace V (orthonormal) and W = H V for one macro iteration.  The scaled augmented Hessian
    A(alpha) = [[0, alpha g^T], [alpha g, H]] is solved in span{e0} + V for ANY alpha without new matvecs; the
    subspace is expanded (one H matvec per new vector) until the kappa-space residual is below tol."""

    def __init__(self, g, hvp, diag):
        self.g, self.hvp, self.diag = g, hvp, diag
        self.V = np.zeros((len(g), 0))
        self.W = np.zeros((len(g), 0))
        self._add(g / np.linalg.norm(g))

    def _add(self, t):
        for _ in range(2):
            t = t - self.V @ (self.V.T @ t)
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            return False
        t = t / nt
        self.V = np.column_stack([self.V, t])
        self.W = np.column_stack([self.W, self.hvp(t)])
        return True

    def _proj(self, alpha):
        V, W, g = self.V, self.W, self.g
        k = V.shape[1]
        Hs = V.T @ W
        Hs = 0.5 * (Hs + Hs.T)
        A = np.zeros((k + 1, k + 1))
        A[0, 1:] = A[1:, 0] = alpha * (V.T @ g)
        A[1:, 1:] = Hs
        w, Z = np.linalg.eigh(A)
        c0, y = Z[0, 0], Z[1:, 0]
        mu = w[0]
        kappa = V @ y / (alpha * c0)
        r = (W @ y + alpha * g * c0 - mu * (V @ y)) / (alpha * c0)  # (H - mu) kappa + g
        return mu, kappa, r, c0

    def solve(self, alpha, tol, max_vecs=60):
        while True:
            mu, kappa, r, c0 = self._proj(alpha)
            if np.linalg.norm(r) < tol or self.V.shape[1] >= max_vecs:
                return mu, kappa, np.linalg.norm(r)
            den = self.diag - mu
            den = np.where(np.abs(den) < 1e-4, np.sign(den + 1e-300) * 1e-4, den)
            if not self._add(-r / den):
                return mu, kappa, np.linalg.norm(r)


def tr_step(
    sub, radius, tol, alpha_min=1.0, alpha_max=1000.0, step_tol=5e-2, max_shift=20
):
    """crate::trah::solve_trust_region semantics on a shared subspace: alpha = 1 first; if ||kappa|| > radius grow
    alpha x8 until feasible, then bisect on log alpha until ||kappa|| within step_tol of the radius; clamp at
    alpha_max."""
    mu, kap, res = sub.solve(alpha_min, tol)
    best = (alpha_min, mu, kap, res)
    if np.linalg.norm(kap) > radius:
        lo = hi = alpha_min
        feas = None
        n = 1
        while hi < alpha_max and n < max_shift:
            hi = min(hi * 8, alpha_max)
            c = sub.solve(hi, tol)
            n += 1
            if np.linalg.norm(c[1]) <= radius:
                feas = (hi,) + c
                break
            lo = hi
        if feas is None:
            a, mu, kap, res = best
            best = (a, mu, kap * radius / np.linalg.norm(kap), res)
        else:
            while (
                n < max_shift
                and np.linalg.norm(feas[2]) < radius * (1 - step_tol)
                and hi > lo * (1 + 1e-9)
            ):
                mid = np.exp(0.5 * (np.log(lo) + np.log(hi)))
                c = sub.solve(mid, tol)
                n += 1
                if np.linalg.norm(c[1]) <= radius:
                    feas, hi = (mid,) + c, mid
                else:
                    lo = mid
            best = feas
    alpha, mu, kap, res = best
    hk = sub.W @ (sub.V.T @ kap)  # H kappa within the subspace (kappa lies in span V)
    pred = sub.g @ kap + 0.5 * kap @ hk
    return dict(
        kappa=kap,
        mu=mu,
        alpha=alpha,
        pred=pred,
        pred_cheap=0.5 * (sub.g @ kap + mu * kap @ kap),
        res=res,
        norm=np.linalg.norm(kap),
    )


def trah(
    sysm,
    C0,
    nd,
    no,
    gtol=1e-9,
    max_macro=200,
    radius0=0.4,
    radius_min=1e-4,
    radius_max=2.0,
    pred_min=1e-12,
    inner_tol=1e-2,
    trace=False,
    use_cayley=True,
):
    """Trust-region AH on the exact injected-path gradient/Hessian.  Converged when max|g_packed| < gtol
    (g_packed = ferric's gradient_blocks scale).  Returns dict(converged, e, C, macro, hist, counts, rejects)."""
    pt = Point(sysm, np.array(C0), nd, no)
    R = radius0
    hist, rejects = [], 0
    for macro in range(1, max_macro + 1):
        gmax = np.abs(pt.g_packed()).max()
        hist.append(dict(macro=macro, e=pt.e, gmax=gmax, R=R, counts=sysm.counts()))
        if trace:
            print(
                f"    macro {macro:3d} E {pt.e:.12f} max|g| {gmax:.2e} R {R:.3f} jk {sysm.n_jk}",
                flush=True,
            )
        if gmax < gtol:
            return dict(
                converged=True,
                e=pt.e,
                C=pt.C,
                point=pt,
                macro=macro,
                hist=hist,
                counts=sysm.counts(),
                rejects=rejects,
            )
        gn = np.linalg.norm(pt.g)
        sub = AHSubspace(pt.g, pt.hvp, pt.diag())
        tol = max(1e-11, inner_tol * min(1.0, gn) * gn)
        while True:
            st = tr_step(sub, R, tol)
            Cn = pt.rotated(st["kappa"], use_cayley)
            new = Point(sysm, Cn, nd, no)
            act = new.e - pt.e
            if (
                abs(st["pred"]) < pred_min
            ):  # predicted gain below resolution: accept (ferric predicted_min)
                rho = 1.0
            else:
                rho = act / st["pred"]
            if trace:
                print(
                    f"      step |k| {st['norm']:.3e} alpha {st['alpha']:.2f} mu {st['mu']:+.3e} pred "
                    f"{st['pred']:+.3e} act {act:+.3e} rho {rho:+.4f} nvec {sub.V.shape[1]} res {st['res']:.1e}",
                    flush=True,
                )
            if rho < 0.0:
                rejects += 1
                R *= 0.7
                if R < radius_min:
                    return dict(
                        converged=False,
                        e=pt.e,
                        C=pt.C,
                        point=pt,
                        macro=macro,
                        hist=hist,
                        counts=sysm.counts(),
                        rejects=rejects,
                        exit="RadiusCollapsed",
                    )
                continue  # re-solve from the SAME point in the SAME subspace (no new J/K for the solve)
            if rho <= 0.25:
                R *= 0.7
            elif rho > 0.75:
                R = min(R * 1.2, radius_max) if st["norm"] >= R * 0.95 else R
            R = max(R, radius_min)
            pt = new
            break
    return dict(
        converged=False,
        e=pt.e,
        C=pt.C,
        point=pt,
        macro=max_macro,
        hist=hist,
        counts=sysm.counts(),
        rejects=rejects,
        exit="MaxIter",
    )


# ------------------------------------------------------------------------ mirror of the molecular rohf.rs AH path
def ferric_ah_mirror(
    sysm,
    C0,
    nd,
    no,
    ah_trigger=1e-2,
    max_iter=200,
    max_step=0.2,
    gtol=1e-9,
    diis_size=8,
    trace=False,
):
    """rohf.rs AH branch as it is written for the molecular path, driven by the injected Fock/response:
    DIIS (ferric's) until it > 3 and err_max < ah_trigger; from then on every iteration: diagonalize F_eff, aufbau,
    f_mo of the STALE F_a/F_b in that basis, ferric_hvp (diag-Fock) AH at alpha = 1 (Davidson to 1e-7), clip the
    step to max|kappa| <= max_step, rotate, no energy test.  Converged when the fresh gradient at C < gtol."""
    import roks_replica as RR

    S = sysm.S
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s**-0.5) @ V.T
    C = np.array(C0)
    diis = RR.FerricDiis(diis_size)
    hist = []
    n_ah = 0
    for it in range(1, max_iter + 1):
        pt = Point(sysm, C, nd, no)
        gmax = np.abs(pt.g_packed()).max()
        F = RR.roothaan(pt.Fa, pt.Fb, pt.Da, pt.Db, S)
        ga = RR.mo_gradient(C, pt.Fa, pt.Fb, nd, no)
        err = S @ C @ (ga - ga.T) @ C.T @ S
        err_max = np.abs(err).max()
        hist.append((it, pt.e, gmax, err_max))
        if trace:
            print(
                f"    it {it:3d} E {pt.e:.12f} max|g| {gmax:.2e} err {err_max:.2e}",
                flush=True,
            )
        if gmax < gtol:
            return dict(
                converged=True,
                e=pt.e,
                C=C,
                it=it,
                hist=hist,
                n_ah=n_ah,
                counts=sysm.counts(),
            )
        if it > 3 and err_max < ah_trigger:
            eps, Cn = np.linalg.eigh(X @ F @ X)
            Cn = X @ Cn
            q = Point.__new__(
                Point
            )  # stale-Fock point in the Roothaan basis (no new Fock build)
            q.sys, q.C, q.nd, q.no, q.n = sysm, Cn, nd, no, Cn.shape[1]
            q.P, q.blocks = pairs(q.n, nd, no)
            q.na, q.nb = occs(q.n, nd, no)
            q.Fa, q.Fb, q.e = pt.Fa, pt.Fb, pt.e
            q.fa, q.fb = Cn.T @ pt.Fa @ Cn, Cn.T @ pt.Fb @ Cn
            sysm._ref = (pt.Da, pt.Db)
            gp = np.array(
                [
                    (q.fa[p, qq] + q.fb[p, qq])
                    if (p >= nd + no and qq < nd)
                    else (q.fa[p, qq] if p >= nd + no else q.fb[p, qq])
                    for p, qq in q.P
                ]
            )
            if np.abs(gp).max() < 1e-7:
                C = Cn
            else:
                dg = np.array([d / 2 for d in q.diag()])
                sub = AHSubspace(gp, q.ferric_hvp, dg)
                _, kap, _ = sub.solve(1.0, 1e-7, max_vecs=50)
                km = np.abs(kap).max()
                if km > max_step:
                    kap = kap * max_step / km
                C = Cn @ cayley(unpack(kap, q.P, q.n))
            n_ah += 1
            continue
        Fn, _ = diis.step(F, err)
        eps, Cn = np.linalg.eigh(X @ Fn @ X)
        C = X @ Cn
    return dict(
        converged=False,
        e=hist[-1][1],
        C=C,
        it=max_iter,
        hist=hist,
        n_ah=n_ah,
        counts=sysm.counts(),
    )


# ------------------------------------------------------------------------------------------------- analysis
def pyscf_fill(Cn, eps, Fa, nd, no):
    """pyscf.scf.rohf._fill_rohf_occ: closed = the nd lowest ROOTHAAN eigenvalues, open = the no lowest ALPHA
    diagonal energies mo_ea = c^T F_a c among the rest; returns Cn reordered closed | open | virtual (eps order)."""
    order = np.argsort(eps, kind="stable")
    closed = list(order[:nd])
    rest = list(order[nd:])
    ea = np.einsum("mi,mn,ni->i", Cn, Fa, Cn)
    op = sorted(rest, key=lambda p: ea[p])[:no]
    virt = [p for p in rest if p not in op]
    return Cn[:, sorted(closed) + sorted(op) + virt]


def state_report(sysm, C, nd, no, hessian=True):
    """Energy, max|g_packed|, Roothaan-label aufbau pattern, occupation-aware per-spin gaps, exact-Hessian spectrum
    (lowest eigenvalues) and 2*ferric-approx vs exact Hessian difference at C."""
    import roks_replica as RR

    pt = Point(sysm, C, nd, no)
    F = RR.roothaan(pt.Fa, pt.Fb, pt.Da, pt.Db, sysm.S)
    s, V = np.linalg.eigh(sysm.S)
    X = V @ np.diag(s**-0.5) @ V.T
    eps, Cr = np.linalg.eigh(X @ F @ X)
    Cr = X @ Cr
    S = sysm.S
    wcl = np.sum((C[:, :nd].T @ S @ Cr) ** 2, axis=0)
    wop = np.sum((C[:, nd : nd + no].T @ S @ Cr) ** 2, axis=0)
    lab = "".join(np.where(wcl > 0.5, "D", np.where(wop > 0.5, "S", "V")))
    out = dict(
        e=pt.e,
        gmax=np.abs(pt.g_packed()).max(),
        labels=lab,
        aufbau=lab == "D" * nd + "S" * no + "V" * (len(eps) - nd - no),
        gap_a=U.occ_gap(pt.Fa, pt.Da, S),
        gap_b=U.occ_gap(pt.Fb, pt.Db, S),
    )
    if hessian:
        H = pt.hessian("exact")
        # ferric's approximate Hessian is used in the Roothaan eigenbasis (rohf.rs diagonalizes F_eff before the AH
        # step), so compare it there: same state, canonical orbitals (PySCF fill picks the columns).
        Cc = pyscf_fill(Cr, eps, pt.Fa, nd, no)
        pc = Point(sysm, Cc, nd, no)
        Hc = pc.hessian("exact")
        Hf = pc.hessian("ferric")
        out["canon_state_same"] = np.abs(pc.e - pt.e)
        out["asym"] = np.abs(H - H.T).max()
        w = np.linalg.eigvalsh(0.5 * (H + H.T))
        out["hess_eigs"] = w
        out["hess_min"] = w[0]
        wf = np.linalg.eigvalsh(0.5 * (2 * Hf + 2 * Hf.T))
        out["ferric_min"] = wf[0]
        out["ferric_vs_exact"] = np.abs(2 * Hf - Hc).max()
        out["canon_min"] = np.linalg.eigvalsh(0.5 * (Hc + Hc.T))[0]
        out["ferric_asym"] = np.abs(Hf - Hf.T).max()
    return out
