#!/usr/bin/env python3
"""Analytic nuclear gradient of COSX (seminumerical) exchange, plain and overlap-fitted.

Derivation: `book/src/14-derivatives/01-hf-gradient.md#hf-gradient-cosx`; the COSX
energy is `book/src/07-hartree-fock/05-link-cosx-cfmm.md#link-cosx-cfmm-link-cosx`.
Code: crates/ferric-scf/src/cosx_gradient.rs (cosx_exchange_gradient_bilinear_with_q,
chunk_gradient, snum_derivative, fitted_exchange_response_with_q) and
crates/ferric-scf/src/gradient.rs (rhf_gradient_cosx_with_q, uhf_gradient_cosx,
rohf_gradient_cosx).

Model: two "atoms" in 3D (charges 2 and 1) carrying five normalized s Gaussians,
with closed-form overlap, kinetic, nuclear-attraction, (mn|ls) and point-potential
integrals A^g_mn = int chi_m chi_n / |r - r_g|. An atom-centred grid: fixed radial
x angular offsets u_g that ride their home atom, r_g = R_h(g) + u_g, and weights
w_g = v_g p_h(g)(r_g) with a smooth softmax partition p_A = exp(-k d_A^2) /
sum_B exp(-k d_B^2), so the weights move with every atom. The COSX exchange is
K~(M) = sum_g w_g phi_g (A^g M phi_g)^T, K = sym(K~(D)) (fit off) or
K_f = sym(Q K~(D)), Q = S S_num^-1, S_num = sum_g w_g phi_g phi_g^T (fit on),
exactly the builder of cosx_k.rs. RHF (2 electrons) and UHF (3 electrons) SCFs
are re-solved at displaced geometries.

Independent route: fourth-order central differences of (i) the grid sum
sum_g w_g phi_g^T B A^g Y Q phi_g at FIXED matrices (geometry rebuilt: points,
weights, AO values, A^g, S, S_num) and (ii) the RE-SOLVED SCF energy. The
analytic side uses the three-term COSX derivative (AO values, A^g with the
charge-position part on the home atom, weight response), the dQ term, the
Z-vector, and the code's W. The skeleton derivatives of h, (mn|ls) and E_nn are
taken by central differences at fixed matrices (they are not under test here;
section hf-gradient-gradient checks them); S' and every grid term are analytic.

Checks (each label carries "(claim X)"):
  X1  Fit off, fixed D: (a) AO + (b) integrals + (c) weights = FD of sum_g w_g e_g;
      each term alone is O(0.1), and dropping the charge-position part of (b)
      or the weight response misses (negative controls).
  X2  Fit off, RHF: g = E_nn' + tr[D h'] - tr[W S'] + J'(D,D)/2 - dT(D,D)/4
      equals FD of the re-solved COSX SCF energy; rows sum to zero.
  X3  sympy: tr[D K(D)] = sum_g w_g phi^T D A D phi is a quadratic form in D with
      dE/dD = 2 K~(D)^T, whose symmetric part is 2 K(D): fit-off COSX is variational.
  X4  sympy + numeric: K_f is not self-adjoint, tr[Y K_f(B)] = tr[B L(Y)] with
      L(Y) = sym(K~(Y Q)), and dE/dD = F + Delta, Delta = (K_f(D) - L(D))/4 (RHF),
      with tr[D Delta] = 0 (so for ONE occupied orbital Delta_oo vanishes; the toy
      therefore has two, or the W mutation below could not be seen).
  X5  Fit on, fixed (Y, B): the explicit derivative with dQ = dS S_num^-1 -
      Q dS_num S_num^-1 equals FD of T(Y,B) = tr[Y Q K~(B)]; without dQ it misses.
  X6  Fit on, RHF: the Z-vector Lagrangian gradient equals FD of the re-solved
      fitted energy; the explicit derivative alone (no response) misses.
  X7  Fit on, UHF: per-spin Z-vector (p = 2, c_L = 1) gradient equals FD; fit-off
      UHF too.
  X8  Exactness anchor: Q = I through the fitted pipeline gives Delta = 0, z = 0
      and the fit-off gradient.

Mutations (each turns the script red; exact one-occurrence substitutions):
  charge-position block of dA^g dropped from the home atom      (X1, X2 fail)
  weight response (c) dropped                                   (X1, X2 fail)
  Z-vector right-hand side -p Delta -> -p Delta / 2             (X6, X7 fail)
  dQ loses its S_num part                                       (X5, X6 fail)
  occupied block of W loses Delta_oo                            (X6, X7 fail)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import scipy.linalg as sla
import sympy as sp
from scipy.special import erf

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def sym(m):
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------- the toy system
ZNUC = np.array([3.0, 1.0])
R0 = np.array([[0.0, 0.0, 0.0], [0.35, -0.25, 1.55]])
EXPS = np.array([1.6, 0.45, 0.18, 1.1, 0.3])
ATOM_OF = np.array([0, 0, 0, 1, 1])
NBF, NATM = len(EXPS), len(ZNUC)
NORM = (2.0 * EXPS / np.pi) ** 0.75
KAPPA = 0.7  # softmax partition stiffness
RHF_OCC, UHF_OCC = (2,), (2, 1)  # 4 electrons closed shell; 3 electrons open shell


def _grid_offsets():
    """Fixed per-atom radial x angular offsets u_g and quadrature weights v_g."""
    rad = np.array([0.25, 0.6, 1.1, 1.8, 2.8, 4.2])
    drad = np.array([0.3, 0.42, 0.6, 0.85, 1.2, 1.7])
    octa = np.vstack([np.eye(3), -np.eye(3)])
    cube = np.array(
        [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    ) / np.sqrt(3)
    ang = np.vstack([octa, cube])
    angw = np.r_[np.full(6, 4 * np.pi / 15), np.full(8, 4 * np.pi * 3 / 40)]
    rng = np.random.default_rng(7)
    out_u, out_v, out_h = [], [], []
    for a in range(NATM):
        rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        for r, dr in zip(rad, drad):
            for d, wa in zip(ang @ rot.T, angw):
                out_u.append(r * d)
                out_v.append(wa * r * r * dr)
                out_h.append(a)
    return np.array(out_u), np.array(out_v), np.array(out_h)


U_G, V_G, HOME = _grid_offsets()


def boys01(t):
    """F0(t), F1(t) for t >= 0 (Taylor series below 1e-2)."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-2
    ts = np.where(small, 1.0, t)
    f0 = 0.5 * np.sqrt(np.pi / ts) * erf(np.sqrt(ts))
    f1 = (f0 - np.exp(-ts)) / (2 * ts)
    tt = np.where(small, t, 0.0)
    f0s = 1 - tt / 3 + tt**2 / 10 - tt**3 / 42 + tt**4 / 216
    f1s = 1 / 3 - tt / 5 + tt**2 / 14 - tt**3 / 54 + tt**4 / 264
    return np.where(small, f0s, f0), np.where(small, f1s, f1)


def pair_data(r):
    """p, mu, P, K, centres for every AO pair (all arrays (n, n, ...))."""
    c = r[ATOM_OF]
    a = EXPS[:, None]
    b = EXPS[None, :]
    p = a + b
    mu = a * b / p
    ab = c[:, None, :] - c[None, :, :]
    kab = np.exp(-mu * np.sum(ab**2, axis=-1))
    pc = (a[..., None] * c[:, None, :] + b[..., None] * c[None, :, :]) / p[..., None]
    nn = NORM[:, None] * NORM[None, :]
    return c, a, b, p, mu, ab, kab, pc, nn


def overlap(r):
    _, _, _, p, _, _, kab, _, nn = pair_data(r)
    return nn * (np.pi / p) ** 1.5 * kab


def overlap_deriv(r):
    """dS/dR_(X,c) as (natm, 3, n, n), analytic."""
    _, _, _, p, mu, ab, kab, _, nn = pair_data(r)
    s = nn * (np.pi / p) ** 1.5 * kab
    dsa = -2 * mu[..., None] * ab * s[..., None]  # d/d(centre of m)
    out = np.zeros((NATM, 3, NBF, NBF))
    for x in range(NATM):
        on_m = (ATOM_OF == x)[:, None, None]
        on_n = (ATOM_OF == x)[None, :, None]
        out[x] = np.moveaxis(dsa * on_m - dsa * on_n, -1, 0)
    return out


def core_h(r):
    _, _, _, p, mu, ab, kab, pc, nn = pair_data(r)
    s = nn * (np.pi / p) ** 1.5 * kab
    t = mu * (3 - 2 * mu * np.sum(ab**2, axis=-1)) * s
    v = np.zeros_like(t)
    for zc, rc in zip(ZNUC, r):
        f0, _ = boys01(p * np.sum((pc - rc) ** 2, axis=-1))
        v -= zc * nn * (2 * np.pi / p) * kab * f0
    return t + v


def eri(r):
    _, _, _, p, _, _, kab, pc, nn = pair_data(r)
    pp = p[:, :, None, None]
    qq = p[None, None, :, :]
    rho = pp * qq / (pp + qq)
    dpq = pc[:, :, None, None, :] - pc[None, None, :, :, :]
    f0, _ = boys01(rho * np.sum(dpq**2, axis=-1))
    pref = 2 * np.pi**2.5 / (pp * qq * np.sqrt(pp + qq))
    kk = (nn * kab)[:, :, None, None] * (nn * kab)[None, None, :, :]
    return pref * kk * f0


def e_nuc(r):
    d = np.linalg.norm(r[0] - r[1])
    return ZNUC[0] * ZNUC[1] / d


def grid(r):
    """Points, weights and dw/dR (npts, natm, 3) of the riding, partitioned grid."""
    pts = r[HOME] + U_G
    d2 = np.sum((pts[:, None, :] - r[None, :, :]) ** 2, axis=-1)  # (g, B)
    s = np.exp(-KAPPA * d2)
    zsum = s.sum(axis=1)
    p_home = s[np.arange(len(pts)), HOME] / zsum
    w = V_G * p_home
    # d d2_(g,B)/dR_X = 2 (r_g - R_B) (delta_(X,h) - delta_(X,B))
    dw = np.zeros((len(pts), NATM, 3))
    diff = pts[:, None, :] - r[None, :, :]
    for x in range(NATM):
        fac = (HOME == x)[:, None].astype(float) - (np.arange(NATM) == x)[None, :]
        dd2 = 2 * diff * fac[..., None]  # (g, B, 3)
        ds = -KAPPA * s[..., None] * dd2
        dz = ds.sum(axis=1)
        dsh = ds[np.arange(len(pts)), HOME]
        dw[:, x, :] = V_G[:, None] * (
            dsh / zsum[:, None] - p_home[:, None] * dz / zsum[:, None]
        )
    return pts, w, dw


def ao_on_grid(r, pts):
    c = r[ATOM_OF]
    diff = pts[:, None, :] - c[None, :, :]
    phi = NORM * np.exp(-EXPS * np.sum(diff**2, axis=-1))
    gphi = -2 * EXPS[None, :, None] * diff * phi[..., None]  # grad_r chi
    return phi, gphi


def a_on_grid(r, pts, deriv=False):
    """A^g (g, n, n); with deriv also dA/d(centre of m) and dA/d(point)."""
    _, a, _, p, mu, ab, kab, pc, nn = pair_data(r)
    pr = pc[None] - pts[:, None, None, :]
    f0, f1 = boys01(p[None] * np.sum(pr**2, axis=-1))
    pref = nn * (2 * np.pi / p) * kab
    amat = pref[None] * f0
    if not deriv:
        return amat
    d_m = pref[None, ..., None] * (
        -2 * mu[None, ..., None] * ab[None] * f0[..., None]
        - 2 * a[None, ..., None] * pr * f1[..., None]
    )
    d_pt = pref[None, ..., None] * 2 * p[None, ..., None] * pr * f1[..., None]
    return amat, d_m, d_pt


class Geo:
    """Every geometry-dependent piece the COSX energy needs at nuclear positions r."""

    def __init__(self, r, deriv=False):
        self.r = r
        self.pts, self.w, self.dw = grid(r)
        self.phi, self.gphi = ao_on_grid(r, self.pts)
        if deriv:
            self.a, self.da_m, self.da_pt = a_on_grid(r, self.pts, deriv=True)
        else:
            self.a = a_on_grid(r, self.pts)
        self.s = overlap(r)
        self.snum = np.einsum("g,gm,gn->mn", self.w, self.phi, self.phi)
        self.h = core_h(r)
        self.g = eri(r)

    def q(self, fit):
        return self.s @ np.linalg.inv(self.snum) if fit else np.eye(NBF)

    def kt(self, m):
        """K~(M) = sum_g w_g phi_g (A^g M phi_g)^T, the builder on a general M."""
        amphi = np.einsum("gmn,gn->gm", self.a, self.phi @ m.T)
        return np.einsum("g,gm,gn->mn", self.w, self.phi, amphi)

    def kx(self, d, q):
        return sym(q @ self.kt(d))

    def lop(self, y, q):
        return sym(self.kt(y @ q))

    def j(self, d):
        return np.einsum("mnls,ls->mn", self.g, d)


def t_value(geo, y, b, fit):
    return np.trace(y @ geo.q(fit) @ geo.kt(b))


# ------------------------------------------------------- the analytic COSX terms
def dt_explicit(geo, terms, fit, parts=("a", "b", "c"), with_dq=True, charge_part=True):
    """Explicit d/dR of sum_k c_k T(Y_k, B_k) at fixed matrices, (natm, 3)."""
    q = geo.q(fit)
    out = np.zeros((NATM, 3))
    w, phi, gphi = geo.w, geo.phi, geo.gphi
    for y, b, c in terms:
        m = y @ q
        f = phi @ b.T  # F_g = B phi_g
        h = phi @ m.T  # H_g = Y Q phi_g
        af = np.einsum("gmn,gn->gm", geo.a, f)
        ah = np.einsum("gmn,gn->gm", geo.a, h)
        e = np.sum(f * ah, axis=1)
        if "a" in parts:  # (a) AO values: the point drags phi (+), the centre (-)
            dedphi = ah @ b + af @ m
            v = (w[:, None] * dedphi)[..., None] * gphi  # (g, m, 3)
            for x in range(NATM):
                out[x] += c * v[HOME == x].sum(axis=(0, 1))
                out[x] -= c * v[:, ATOM_OF == x].sum(axis=(0, 1))
        if "b" in parts:  # (b) dA^g: bra centre, ket centre, charge position
            fh = w[:, None, None] * f[:, :, None] * h[:, None, :]
            fh = fh + fh.transpose(0, 2, 1)  # both orientations of A symmetric
            vm = np.einsum("gmn,gmnc->gmc", fh, geo.da_m)
            vpt = np.einsum("gmn,gmnc->gc", 0.5 * fh, geo.da_pt)
            for x in range(NATM):
                out[x] += c * vm[:, ATOM_OF == x].sum(axis=(0, 1))
                if charge_part:
                    out[x] += c * vpt[HOME == x].sum(axis=0)
        if "c" in parts:  # (c) weight response
            out += c * np.einsum("g,gxc->xc", e, geo.dw)
    if fit and with_dq:
        sninv = np.linalg.inv(geo.snum)
        mtot = sum(c * geo.kt(b) @ y for y, b, c in terms)
        p1 = sninv @ mtot
        p2 = p1 @ (sninv @ geo.s).T
        ds = overlap_deriv(geo.r)
        dsn = snum_deriv(geo)
        out += np.einsum("xcmn,nm->xc", ds, p1)
        out -= np.einsum("xcmn,nm->xc", dsn, p2)
    return out


def snum_deriv(geo):
    """dS_num/dR_(X,c) on the moving grid: weights + AO values (natm, 3, n, n)."""
    w, phi, gphi = geo.w, geo.phi, geo.gphi
    out = np.einsum("gxc,gm,gn->xcmn", geo.dw, phi, phi)
    for x in range(NATM):
        dphi = gphi * ((HOME == x)[:, None] * 1.0 - (ATOM_OF == x)[None, :])[..., None]
        t = np.einsum("g,gmc,gn->cmn", w, dphi, phi)
        out[x] += t + t.transpose(0, 2, 1)
    return out


# ------------------------------------------------------------------ SCF solvers
def solve(r, fit, nocc, guess=None, q_identity=False):
    """RHF (nocc = (n,)) or UHF (nocc = (na, nb)) with COSX exchange on the toy grid."""
    geo = Geo(r)
    q = np.eye(NBF) if q_identity else geo.q(fit)
    s = geo.s
    if guess is None:
        _, c = sla.eigh(geo.h, s)
        dens = [c[:, :n] @ c[:, :n].T for n in nocc]
    else:
        dens = [d.copy() for d in guess]
    restricted = len(nocc) == 1
    hist_f, hist_e = [], []
    for _ in range(400):
        if restricted:
            d = 2 * dens[0]
            focks = [geo.h + geo.j(d) - 0.5 * geo.kx(d, q)]
        else:
            dt = dens[0] + dens[1]
            focks = [geo.h + geo.j(dt) - geo.kx(ds, q) for ds in dens]
        errs = [f @ ds @ s - s @ ds @ f for f, ds in zip(focks, dens)]
        err = max(np.max(np.abs(e)) for e in errs)
        if err < 1e-13:
            break
        hist_f.append(focks)
        hist_e.append(np.concatenate([e.ravel() for e in errs]))
        hist_f, hist_e = hist_f[-8:], hist_e[-8:]
        nh = len(hist_e)
        bmat = -np.ones((nh + 1, nh + 1))
        bmat[nh, nh] = 0
        for i in range(nh):
            for k in range(nh):
                bmat[i, k] = hist_e[i] @ hist_e[k]
        rhs = np.zeros(nh + 1)
        rhs[nh] = -1
        coef = np.linalg.lstsq(bmat, rhs, rcond=None)[0][:nh]
        focks = [
            sum(cf * hf[s_] for cf, hf in zip(coef, hist_f)) for s_ in range(len(nocc))
        ]
        dens = []
        for f, n in zip(focks, nocc):
            _, c = sla.eigh(f, s)
            dens.append(c[:, :n] @ c[:, :n].T)
    else:
        raise RuntimeError(f"toy SCF did not converge (err {err:.1e})")
    orbs = [sla.eigh(f, s) for f in focks]
    if restricted:
        d = 2 * dens[0]
        e = (
            e_nuc(r)
            + np.sum(d * geo.h)
            + 0.5 * np.sum(d * geo.j(d))
            - 0.25 * np.sum(d * geo.kx(d, q))
        )
    else:
        dt = dens[0] + dens[1]
        e = e_nuc(r) + np.sum(dt * geo.h) + 0.5 * np.sum(dt * geo.j(dt))
        e -= 0.5 * sum(np.sum(ds * geo.kx(ds, q)) for ds in dens)
    return {"e": e, "dens": dens, "orbs": orbs, "nocc": nocc, "geo": geo}


def fd(fun, r, h=1e-3):
    """Fourth-order central difference of a scalar fun(r) for every (atom, axis)."""
    out = np.zeros((NATM, 3))
    for x in range(NATM):
        for c in range(3):
            v = {}
            for t in (-2, -1, 1, 2):
                rr = r.copy()
                rr[x, c] += t * h
                v[t] = fun(rr)
            out[x, c] = (8 * (v[1] - v[-1]) - (v[2] - v[-2])) / (12 * h)
    return out


def skeleton(r, dh, j_terms):
    """E_nn + tr[Dh h] + sum c tr[Y J(B)] at FIXED matrices (differenced, not under test)."""
    h = core_h(r)
    g = eri(r)
    val = e_nuc(r) + np.sum(dh * h)
    for y, b, c in j_terms:
        val += c * np.sum(y * np.einsum("mnls,ls->mn", g, b))
    return val


def pulay(r, w):
    return -np.einsum("xcmn,mn->xc", overlap_deriv(r), w)


# ------------------------------------------------------------------ gradients
def grad_plain(scf, r):
    """Fit-off gradient: RHF or UHF, no response (the energy is variational)."""
    geo = Geo(r, deriv=True)
    dens = scf["dens"]
    if len(dens) == 1:
        d = 2 * dens[0]
        eps, c = scf["orbs"][0]
        n = scf["nocc"][0]
        w = 2 * c[:, :n] @ np.diag(eps[:n]) @ c[:, :n].T
        g = fd(lambda rr: skeleton(rr, d, [(d, d, 0.5)]), r)
        return g + pulay(r, w) + dt_explicit(geo, [(d, d, -0.25)], fit=False)
    dt = dens[0] + dens[1]
    w = sum(
        c[:, :n] @ np.diag(eps[:n]) @ c[:, :n].T
        for (eps, c), n in zip(scf["orbs"], scf["nocc"])
    )
    g = fd(lambda rr: skeleton(rr, dt, [(dt, dt, 0.5)]), r)
    terms = [(ds, ds, -0.5) for ds in dens]
    return g + pulay(r, w) + dt_explicit(geo, terms, fit=False)


def zvector(geo, scf, q):
    """Fitted-COSX Z-vector (RHF: p=4, c_L=1/2, Delta=(K_f-L)/4; UHF: p=2, c_L=1)."""
    restricted = len(scf["dens"]) == 1
    pz, cl, cd, wocc = (4.0, 0.5, 0.25, 2.0) if restricted else (2.0, 1.0, 0.5, 1.0)
    blocks = []
    for (eps, c), n, ds in zip(scf["orbs"], scf["nocc"], scf["dens"]):
        d = 2 * ds if restricted else ds
        delta = cd * (geo.kx(d, q) - geo.lop(d, q))
        blocks.append((c[:, :n], c[:, n:], eps[:n], eps[n:], delta))
    sizes = [co.shape[1] * cv.shape[1] for co, cv, _, _, _ in blocks]
    nvec = sum(sizes)

    def unpack(v):
        out, o = [], 0
        for co, cv, _, _, _ in blocks:
            k = cv.shape[1] * co.shape[1]
            out.append(v[o : o + k].reshape(cv.shape[1], co.shape[1]))
            o += k
        return out

    def zsym(zz):
        return [sym(cv @ z @ co.T) for (co, cv, _, _, _), z in zip(blocks, zz)]

    def response(zs):
        jz = geo.j(sum(zs))
        return [jz - cl * geo.lop(z, q) for z in zs]

    def apply(v):
        zz = unpack(v)
        rr = response(zsym(zz))
        out = []
        for (co, cv, eo, ev, _), z, rm in zip(blocks, zz, rr):
            out.append(((ev[:, None] - eo[None, :]) * z + pz * cv.T @ rm @ co).ravel())
        return np.concatenate(out)

    amat = np.column_stack([apply(e) for e in np.eye(nvec)])
    rhs = np.concatenate(
        [(-pz * cv.T @ delta @ co).ravel() for co, cv, _, _, delta in blocks]
    )
    zvec = np.linalg.solve(amat, rhs)
    zz = unpack(zvec)
    zs = zsym(zz)
    rr = response(zs)
    w = np.zeros((NBF, NBF))
    for (co, cv, eo, _, delta), rm, z in zip(blocks, rr, zz):
        occ = co.T @ (delta + rm) @ co + np.diag(eo)
        w += wocc * co @ occ @ co.T
        w += sym(cv @ (z * eo[None, :]) @ co.T)
    deltas = [b[4] for b in blocks]
    return zs, w, zvec, deltas


def grad_fitted(scf, r, q_identity=False, response=True):
    """Fitted-COSX Lagrangian gradient (RHF or UHF); q_identity is the anchor."""
    geo = Geo(r, deriv=True)
    fit = not q_identity
    q = np.eye(NBF) if q_identity else geo.q(True)
    restricted = len(scf["dens"]) == 1
    dens = [2 * scf["dens"][0]] if restricted else scf["dens"]
    dt = sum(dens)
    zs, w, zvec, deltas = zvector(geo, scf, q)
    if not response:
        zs = [np.zeros_like(z) for z in zs]
        w = sum(
            (2 if restricted else 1) * c[:, :n] @ np.diag(eps[:n]) @ c[:, :n].T
            for (eps, c), n in zip(scf["orbs"], scf["nocc"])
        )
    zt = sum(zs)
    g = fd(lambda rr: skeleton(rr, dt + zt, [(dt, dt, 0.5), (zt, dt, 1.0)]), r)
    g += pulay(r, w)
    if restricted:
        terms = [(dt, dt, -0.25), (zs[0], dt, -0.5)]
    else:
        terms = [(d, d, -0.5) for d in dens] + [(z, d, -1.0) for z, d in zip(zs, dens)]
    g += dt_explicit(geo, terms, fit=fit)
    return g, zvec, deltas


# ------------------------------------------------------------------ the checks
def sympy_checks():
    """X3, X4 in symbols: two AOs, two grid points, general Q."""
    n = 2
    dsym = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"d{min(i, j)}{max(i, j)}"))
    dgen = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"g{i}{j}"))
    ysym = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"y{min(i, j)}{max(i, j)}"))
    bsym = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"b{min(i, j)}{max(i, j)}"))
    qgen = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"q{i}{j}"))
    pts = []
    for g in range(2):
        w = sp.Symbol(f"w{g}", positive=True)
        phi = sp.Matrix(n, 1, lambda i, _, g=g: sp.Symbol(f"p{g}{i}"))
        a = sp.Matrix(n, n, lambda i, j, g=g: sp.Symbol(f"a{g}{min(i, j)}{max(i, j)}"))
        pts.append((w, phi, a))

    def kt(m):
        return sum((w * phi * (a * m * phi).T for w, phi, a in pts), sp.zeros(n, n))

    def symm(m):
        return (m + m.T) / 2

    e_gen = sum((w * (phi.T * dgen.T * a * dgen * phi)[0] for w, phi, a in pts), 0)
    grad = sp.Matrix(n, n, lambda i, j: sp.diff(e_gen, dgen[i, j]))
    ok_grad = sp.expand(grad - 2 * kt(dgen).T) == sp.zeros(n, n)
    e_sym = sum((w * (phi.T * dsym * a * dsym * phi)[0] for w, phi, a in pts), 0)
    ok_trace = sp.expand(e_sym - (dsym * symm(kt(dsym))).trace()) == 0
    t = sp.Symbol("t")
    ok_quad = (
        sp.expand(e_sym.subs({s: t * s for s in dsym.free_symbols}) - t**2 * e_sym) == 0
    )
    check(
        "X3 tr[D K(D)] is a quadratic form, dE/dD = 2 K~(D)^T, sym part 2 K(D) (claim X3)",
        ok_grad and ok_trace and ok_quad,
        f"grad {ok_grad} trace {ok_trace} homogeneous {ok_quad}",
    )

    def kf(b):
        return symm(qgen * kt(b))

    def lop(y):
        return symm(kt(y * qgen))

    adj = sp.expand((ysym * kf(bsym)).trace() - (bsym * lop(ysym)).trace()) == 0
    not_self = sp.expand((ysym * kf(bsym)).trace() - (bsym * kf(ysym)).trace()) != 0
    q_one = (
        sp.expand(
            ((ysym * kf(bsym)).trace() - (bsym * kf(ysym)).trace()).subs(
                {qgen[i, j]: int(i == j) for i in range(n) for j in range(n)}
            )
        )
        == 0
    )
    check(
        "X4a tr[Y K_f(B)] = tr[B L(Y)], K_f not self-adjoint, self-adjoint at Q = I (claim X4)",
        adj and not_self and q_one,
        f"adjoint {adj} not-self-adjoint {not_self} Q=I {q_one}",
    )


def main():
    sympy_checks()
    r = R0.copy()
    geo = Geo(r, deriv=True)
    rng = np.random.default_rng(3)
    dfix = sym(rng.standard_normal((NBF, NBF))) * 0.3 + np.eye(NBF) * 0.2

    # X1: fit-off three-term derivative at fixed D.
    an = dt_explicit(geo, [(dfix, dfix, 1.0)], fit=False)
    num = fd(lambda rr: t_value(Geo(rr), dfix, dfix, False), r)
    err = np.max(np.abs(an - num))
    sizes = [
        np.max(np.abs(dt_explicit(geo, [(dfix, dfix, 1.0)], False, parts=(p,))))
        for p in "abc"
    ]
    miss_c = np.max(
        np.abs(dt_explicit(geo, [(dfix, dfix, 1.0)], False, parts=("a", "b")) - num)
    )
    miss_q = np.max(
        np.abs(dt_explicit(geo, [(dfix, dfix, 1.0)], False, charge_part=False) - num)
    )
    check(
        "X1 fit off: (a)+(b)+(c) = FD of sum_g w_g e_g at fixed D (claim X1)",
        err < 1e-8
        and min(sizes) > 1e-2
        and min(miss_c, miss_q) > 1e3 * max(err, 1e-11),
        f"err {err:.1e} |a|,|b|,|c| {sizes[0]:.2f},{sizes[1]:.2f},{sizes[2]:.2f} "
        f"no-weights {miss_c:.1e} no-charge-position {miss_q:.1e}",
    )

    # X2: fit-off RHF gradient vs FD of the re-solved SCF energy.
    ref = solve(r, fit=False, nocc=RHF_OCC)
    g_plain = grad_plain(ref, r)
    fd_plain = fd(lambda rr: solve(rr, False, RHF_OCC, guess=ref["dens"])["e"], r)
    err2 = np.max(np.abs(g_plain - fd_plain))
    trans = np.max(np.abs(g_plain.sum(axis=0)))
    check(
        "X2 fit off RHF: gradient = FD of the re-solved COSX SCF energy (claim X2)",
        err2 < 1e-8 and trans < 1e-8,
        f"err {err2:.1e} |sum over atoms| {trans:.1e} |g| {np.max(np.abs(g_plain)):.2f}",
    )

    # X4b: dE/dD = F + Delta for the fitted energy (directional derivative).
    q = geo.q(True)
    d0 = 2 * solve(r, fit=True, nocc=RHF_OCC)["dens"][0]
    dd = sym(rng.standard_normal((NBF, NBF)))

    def e_of(d):
        return (
            np.sum(d * geo.h)
            + 0.5 * np.sum(d * geo.j(d))
            - 0.25 * np.sum(d * geo.kx(d, q))
        )

    hh = 1e-4
    num_dir = (
        8 * (e_of(d0 + hh * dd) - e_of(d0 - hh * dd))
        - (e_of(d0 + 2 * hh * dd) - e_of(d0 - 2 * hh * dd))
    ) / (12 * hh)
    fock = geo.h + geo.j(d0) - 0.5 * geo.kx(d0, q)
    delta = 0.25 * (geo.kx(d0, q) - geo.lop(d0, q))
    an_dir = np.sum(dd * (fock + delta))
    err4 = abs(num_dir - an_dir)
    miss4 = abs(num_dir - np.sum(dd * fock))
    adj_num = abs(np.trace(dfix @ geo.kx(dd, q)) - np.trace(dd @ geo.lop(dfix, q)))
    tr_d_delta = abs(np.sum(d0 * delta))  # tr[D Delta] = 0 by the adjoint identity
    check(
        "X4b fitted: dE/dD = F + Delta, Delta = (K_f - L)(D)/4; tr[Y K_f(B)] = tr[B L(Y)] (claim X4)",
        err4 < 1e-9 and miss4 > 1e3 * err4 and adj_num < 1e-12 and tr_d_delta < 1e-13,
        f"err {err4:.1e} F-only miss {miss4:.1e} adjoint residual {adj_num:.1e} "
        f"|tr[D Delta]| {tr_d_delta:.1e}",
    )

    # X5: fitted explicit derivative of T(Y, B) at fixed matrices, dQ included.
    yfix = sym(rng.standard_normal((NBF, NBF))) * 0.3
    an5 = dt_explicit(geo, [(yfix, dfix, 1.0)], fit=True)
    num5 = fd(lambda rr: t_value(Geo(rr), yfix, dfix, True), r)
    err5 = np.max(np.abs(an5 - num5))
    miss5 = np.max(
        np.abs(dt_explicit(geo, [(yfix, dfix, 1.0)], True, with_dq=False) - num5)
    )
    check(
        "X5 fit on: explicit dT(Y,B) with dQ = FD of tr[Y Q K~(B)] at fixed Y, B (claim X5)",
        err5 < 1e-8 and miss5 > 1e3 * max(err5, 1e-11),
        f"err {err5:.1e} without dQ {miss5:.1e}",
    )

    # X6: fitted RHF Lagrangian gradient vs FD of the re-solved fitted energy.
    reff = solve(r, fit=True, nocc=RHF_OCC)
    g_fit, _, _ = grad_fitted(reff, r)
    g_nor, _, _ = grad_fitted(reff, r, response=False)
    fd_fit = fd(lambda rr: solve(rr, True, RHF_OCC, guess=reff["dens"])["e"], r)
    err6 = np.max(np.abs(g_fit - fd_fit))
    miss6 = np.max(np.abs(g_nor - fd_fit))
    check(
        "X6 fit on RHF: Z-vector Lagrangian gradient = FD; no response misses (claim X6)",
        err6 < 1e-8 and miss6 > 1e2 * max(err6, 1e-10),
        f"err {err6:.1e} without response {miss6:.1e}",
    )

    # X7: UHF, fitted and fit-off.
    refu = solve(r, fit=True, nocc=UHF_OCC)
    g_ufit, _, _ = grad_fitted(refu, r)
    g_unor, _, _ = grad_fitted(refu, r, response=False)
    fd_ufit = fd(lambda rr: solve(rr, True, UHF_OCC, guess=refu["dens"])["e"], r)
    refp = solve(r, fit=False, nocc=UHF_OCC)
    g_up = grad_plain(refp, r)
    fd_up = fd(lambda rr: solve(rr, False, UHF_OCC, guess=refp["dens"])["e"], r)
    err7 = np.max(np.abs(g_ufit - fd_ufit))
    err7p = np.max(np.abs(g_up - fd_up))
    miss7 = np.max(np.abs(g_unor - fd_ufit))
    check(
        "X7 UHF: per-spin Z-vector gradient = FD (fit on), fit-off gradient = FD (claim X7)",
        err7 < 1e-8 and err7p < 1e-8 and miss7 > 1e2 * max(err7, 1e-10),
        f"fit-on err {err7:.1e} (no response {miss7:.1e}) fit-off err {err7p:.1e}",
    )

    # X8: Q = I anchor through the fitted pipeline.
    g_id, zvec, deltas = grad_fitted(ref, r, q_identity=True)
    dz = max(np.max(np.abs(zvec)), max(np.max(np.abs(d)) for d in deltas))
    err8 = np.max(np.abs(g_id - g_plain))
    check(
        "X8 anchor Q = I: Delta = 0, z = 0, fitted pipeline = fit-off gradient (claim X8)",
        dz < 1e-15 and err8 < 1e-12,
        f"max|Delta|,|z| {dz:.1e} |g_Q=I - g_plain| {err8:.1e}",
    )

    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all COSX gradient identities verified")


if __name__ == "__main__":
    main()
