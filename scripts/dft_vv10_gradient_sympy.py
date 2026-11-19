#!/usr/bin/env python3
"""VV10 nuclear gradient: what ferric computes, and what it drops.

Derivation: `book/src/14-derivatives/03-xc-gradient.md#xc-gradient-vv10`
(energy and potentials: `book/src/09-kohn-sham/05-vv10.md#vv10-energy-potentials`).
Code: crates/ferric-dft/src/gradient.rs::vv10_gradient_from_density (per-point
v_rho, v_sigma from crates/ferric-dft/src/vv10.rs::compute_vv10_potentials, then
the GGA AO-derivative assembly gga_gradient_from_potentials), called from
crates/ferric-scf/src/ks_gradient.rs::add_xc_gradient on the total density.

The VV10 energy on an atom-centred grid is a double sum
  E = sum_i W_i rho_i [beta + f_i / 2],  f_i = sum_p W_p rho_p Phi_ip,
  Phi_ip = -3 / (2 g_i g_p (g_i + g_p)),  g_x = omega0_x R_ip^2 + kappa_x.
Its exact derivative with respect to a nuclear coordinate R_{A,alpha} has four
parts:
  G_AO   sum_i W_i (v_rho,i drho_i/dR + v_sigma,i dsigma_i/dR) at fixed points
         (the code's term; v_rho, v_sigma are the vv10.rs potentials, which
         already carry the double-sum factor 2);
  G_home sum_{i: home(i)=A} W_i (v_rho,i d_alpha rho_i + v_sigma,i d_alpha sigma_i)
         (the point rides its atom: the integrand moves under it);
  G_ker  sum_{i,p} W_i rho_i W_p rho_p dPhi_ip/d(R^2) (r_i - r_p)_alpha
         (delta_{home(i)A} - delta_{home(p)A})   (the pair distance changes);
  G_wt   sum_i rho_i (beta + f_i) dW_i/dR_{A,alpha}   (partition weights).
ferric computes G_AO only. G_home + G_ker + G_wt is the derivative of the
quadrature rule at a frozen integrand; it vanishes when the rule is exact.

Model: a two-atom 3-D density from five real Gaussians (s, s, p_z on A; s, s on
B), D = 2 (c1 c1^T + c2 c2^T) with c1 > 0 on the s functions so rho > 0
everywhere; ferric's VV10 constants (b = 6.0, C = 0.01, wB97X-V). Each atom
carries a product grid (Gauss-Legendre radial on [0, 8] Bohr, Gauss-Legendre
in cos theta, uniform phi) whose points ride the atom, and a smooth two-atom
partition P_A = q_A / (q_A + q_B), q = exp(-gamma |r - R|^2).

Independent route: central finite differences of the energy, with the grid
held fixed in space (for G_AO) and with the grid riding the atoms (for the
full gradient), and sympy for the two new partials on a three-point grid.

Checks (each label names its claim: weights, kernel, potentials, code, full,
invariance, anchor, limit; the book boxes cite claims code and full):
  S1  sympy: dE/dW_i = rho_i (beta + f_i), not rho_i (beta + f_i / 2).
  S2  sympy: the explicit dependence of E on a point's position is
      sum_p W_i rho_i W_p rho_p 2 dPhi_ip/d(R^2) (x_i - x_p), with
      dPhi/d(R^2) = (3/2)/(g_i g_p g_t) [omega_i (1/g_i + 1/g_t) + omega_p (1/g_p + 1/g_t)].
  P1  the code's v_rho, v_sigma times W are the derivatives of E (FD per point).
  G1  G_AO (the code's assembly) equals the FD of E with the grid fixed in space.
  G2  G_AO + G_home + G_ker + G_wt equals the FD of the moving-grid energy.
  G3  the full gradient sums to zero over atoms; G_AO alone does not.
  G4  negative controls: dropping G_ker, or using rho (beta + f/2) in G_wt,
      misses G2 by far more than G2's agreement.
  G5  anchor: on one atom the moving-grid energy does not depend on R_A, so
      G_AO = -G_home exactly (the dropped part is the whole code gradient).
  G6  the dropped part shrinks under grid refinement, G_AO converges, and
      the net force sum_A G_AO goes to zero (exact-quadrature limit).

Mutations (each turns the script red; exact one-occurrence substitutions):
  v_sigma built from u_i instead of w_i in the code-form potentials (G1 fails)
  the kernel term G_ker halved                                          (G2 fails)
  the weight response built from rho (beta + f/2)                      (G2 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys
from itertools import pairwise

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- constants
B_VV, C_VV = 6.0, 0.01  # wB97X-V
PI43 = 4.0 * np.pi / 3.0
K_VV = B_VV * 1.5 * np.pi * (9.0 * np.pi) ** (-1.0 / 6.0)
BETA = (3.0 / B_VV**2) ** 0.75 / 32.0
GAMMA = 0.3  # partition sharpness
RMAX = 8.0


# ------------------------------------------------------------- sympy claims
def sympy_claims():
    n = 3
    W = sp.symbols("W0:3", positive=True)
    rho = sp.symbols("rho0:3", positive=True)
    beta = sp.Symbol("beta", positive=True)
    # symmetric kernel matrix, self pairs included
    P = [[sp.Symbol(f"Phi{min(i, p)}{max(i, p)}") for p in range(n)] for i in range(n)]
    f = [sum(W[p] * rho[p] * P[i][p] for p in range(n)) for i in range(n)]
    energy = sp.expand(sum(W[i] * rho[i] * (beta + f[i] / 2) for i in range(n)))

    ok1 = all(
        sp.expand(sp.diff(energy, W[i]) - rho[i] * (beta + f[i])) == 0 for i in range(n)
    )
    bad1 = sp.expand(sp.diff(energy, W[0]) - rho[0] * (beta + f[0] / 2)) != 0
    check(
        "S1 dE/dW_i = rho_i (beta + f_i) (claim weights)",
        ok1 and bad1,
        "(and the rho (beta + f/2) form is wrong)",
    )

    # each off-diagonal pair enters E with total weight W_i rho_i W_p rho_p
    ok2a = all(
        sp.expand(sp.diff(energy, P[i][p]) - W[i] * rho[i] * W[p] * rho[p]) == 0
        for i in range(n)
        for p in range(n)
        if i < p
    )
    # the kernel's R^2 derivative through g_i = omega_i R^2 + kappa_i, g_p
    r2, oi, op, ki, kp = sp.symbols("R2 omega_i omega_p kappa_i kappa_p", positive=True)
    gi, gp = oi * r2 + ki, op * r2 + kp
    gt = gi + gp
    phi = -sp.Rational(3, 2) / (gi * gp * gt)
    dphi = (
        sp.Rational(3, 2)
        / (gi * gp * gt)
        * (oi * (1 / gi + 1 / gt) + op * (1 / gp + 1 / gt))
    )
    ok2b = sp.simplify(sp.diff(phi, r2) - dphi) == 0
    check(
        "S2 dE/dx_i = sum_p W_i rho_i W_p rho_p 2 dPhi/dR^2 (x_i - x_p) (claim kernel)",
        ok2a and ok2b,
        "(pair weight W_i rho_i W_p rho_p, dPhi/dR^2 closed form)",
    )


# ------------------------------------------------------------------ model
# (atom, exponent, axis); axis -1 is an s function, 0..2 a p function
BASIS = [(0, 1.1, -1), (0, 0.32, -1), (0, 0.6, 2), (1, 0.9, -1), (1, 0.28, -1)]
NBF = len(BASIS)
BF_ATOM = np.array([b[0] for b in BASIS])
C1 = np.array([0.7, 0.5, 0.0, 0.6, 0.4])
C2 = np.array([0.1, -0.2, 0.5, 0.15, -0.1])
DMAT = 2.0 * (np.outer(C1, C1) + np.outer(C2, C2))
R0 = np.array([[0.0, 0.0, 0.0], [0.45, 0.3, 1.6]])


def aos(pts, centers):
    """chi (nbf, n), dchi (3, nbf, n), ddchi (3, 3, nbf, n) at pts."""
    n = len(pts)
    chi = np.zeros((NBF, n))
    dchi = np.zeros((3, NBF, n))
    ddchi = np.zeros((3, 3, NBF, n))
    eye = np.eye(3)
    for mu, (at, a, k) in enumerate(BASIS):
        d = (pts - centers[at]).T  # (3, n)
        e = np.exp(-a * np.sum(d * d, axis=0))
        if k < 0:
            chi[mu] = e
            for i in range(3):
                dchi[i, mu] = -2 * a * d[i] * e
                for j in range(3):
                    ddchi[i, j, mu] = (4 * a * a * d[i] * d[j] - 2 * a * eye[i, j]) * e
        else:
            chi[mu] = d[k] * e
            for i in range(3):
                dchi[i, mu] = (eye[i, k] - 2 * a * d[i] * d[k]) * e
                for j in range(3):
                    ddchi[i, j, mu] = (
                        -2 * a * eye[i, k] * d[j]
                        - 2 * a * eye[j, k] * d[i]
                        + d[k] * (4 * a * a * d[i] * d[j] - 2 * a * eye[i, j])
                    ) * e
    return chi, dchi, ddchi


def density(chi, dchi, ddchi):
    m = DMAT @ chi
    rho = np.sum(chi * m, axis=0)
    grad = np.array([2 * np.sum(dchi[b] * m, axis=0) for b in range(3)])
    hess = np.zeros((3, 3, chi.shape[1]))
    for a in range(3):
        for b in range(3):
            hess[a, b] = 2 * (
                np.sum(ddchi[a, b] * m, axis=0)
                + np.sum(dchi[a] * (DMAT @ dchi[b]), axis=0)
            )
    return rho, grad, hess


def atom_offsets(level):
    nr, nt, nph = level
    x, wx = np.polynomial.legendre.leggauss(nr)
    r = RMAX * (1 + x) / 2
    wr = RMAX / 2 * wx * r * r
    ct, wt = np.polynomial.legendre.leggauss(nt)
    ph = 2 * np.pi * (np.arange(nph) + 0.5) / nph
    st = np.sqrt(1 - ct * ct)
    dirs = np.array(
        [[s * np.cos(p), s * np.sin(p), c] for c, s in zip(ct, st) for p in ph]
    )
    wang = np.repeat(wt, nph) * (2 * np.pi / nph)
    off = (r[:, None, None] * dirs[None]).reshape(-1, 3)
    w = (wr[:, None] * wang[None]).reshape(-1)
    return off, w


class Grid:
    """Points ride their home atom; weight = w0 * P_home(r) (smooth partition)."""

    def __init__(self, level, natoms):
        off, w = atom_offsets(level)
        self.off = np.concatenate([off] * natoms)
        self.w0 = np.concatenate([w] * natoms)
        self.home = np.repeat(np.arange(natoms), len(w))
        self.natoms = natoms

    def points(self, R):
        return R[self.home] + self.off

    def weights(self, R):
        pts = self.points(R)
        if self.natoms == 1:
            return self.w0.copy(), np.zeros((len(pts), 1, 3))
        d2 = np.array([np.sum((pts - R[a]) ** 2, axis=1) for a in range(2)])
        s = GAMMA * (d2[1] - d2[0])  # P_A = logistic(s)
        pa = 1 / (1 + np.exp(-s))
        p_home = np.where(self.home == 0, pa, 1 - pa)
        # ds/dR_C at a point riding home h: r = R_h + off
        dsdr = np.zeros((len(pts), 2, 3))
        for c in range(2):
            dr = (self.home == c).astype(float)[:, None]  # dr/dR_c = 1 if home
            # s = gamma (|r - R_B|^2 - |r - R_A|^2)
            dsdr[:, c] = (
                2
                * GAMMA
                * ((pts - R[1]) * (dr - (c == 1)) - (pts - R[0]) * (dr - (c == 0)))
            )
        dpa = (pa * (1 - pa))[:, None, None] * dsdr
        dhome = np.where((self.home == 0)[:, None, None], dpa, -dpa)
        return self.w0 * p_home, self.w0[:, None, None] * dhome


# ------------------------------------------------------------------- VV10
def vv10(W, xyz, rho, sigma, home=None, natoms=1, chunk=1500):
    """Energy, code-form potentials (vrho, vsig), f, and the kernel term."""
    w0sq_tmp = C_VV * (sigma / (rho * rho)) ** 2
    om = np.sqrt(w0sq_tmp + PI43 * rho)
    kp = K_VV * rho ** (1.0 / 6.0)
    dk_dr = kp / 6.0
    dw0_dr = (0.5 * PI43 * rho - 2.0 * w0sq_tmp) / om
    dw0_dg = C_VV * sigma / (rho**3 * om)
    wr = W * rho
    n = len(W)
    f = np.zeros(n)
    u = np.zeros(n)
    w = np.zeros(n)
    ker = np.zeros((natoms, 3))
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        dv = xyz[sl, None, :] - xyz[None, :, :]
        r2 = np.sum(dv * dv, axis=2)
        gi = r2 * om[sl, None] + kp[sl, None]
        gp = r2 * om[None, :] + kp[None, :]
        gt = gi + gp
        t = wr[None, :] / (gi * gp * gt)
        f[sl] = -1.5 * t.sum(axis=1)
        tu = t * (1 / gi + 1 / gt)
        u[sl] = tu.sum(axis=1)
        w[sl] = (tu * r2).sum(axis=1)
        if home is not None:
            # dphi = W_p rho_p dPhi_ip/d(R^2); pref = W_i rho_i W_p rho_p dPhi/d(R^2)
            dphi = 1.5 * (om[sl, None] * tu + om[None, :] * t * (1 / gp + 1 / gt))
            pref = wr[sl, None] * dphi
            # sum_ip pref_ip (r_i - r_p) (delta_{home(i)a} - delta_{home(p)a})
            for a in range(natoms):
                on_i = home[sl] == a
                on_p = (home == a).astype(float)
                row = pref[on_i]
                ker[a] += row.sum(axis=1) @ xyz[sl][on_i] - (row.sum(axis=0)) @ xyz
                ker[a] -= (pref @ on_p) @ xyz[sl] - (on_p * pref.sum(axis=0)) @ xyz
    energy = np.sum(wr * (BETA + 0.5 * f))
    vrho = BETA + f + 1.5 * (u * dk_dr + w * dw0_dr)
    vsig = 1.5 * w * dw0_dg
    return energy, vrho, vsig, f, ker


def energy_at(R_ao, R_grid, grid):
    xyz = grid.points(R_grid)
    W, _ = grid.weights(R_grid)
    rho, grad, _ = density(*aos(xyz, R_ao))
    return vv10(W, xyz, rho, np.sum(grad * grad, axis=0))[0]


def gradient_terms(R, grid):
    """G_AO (code form), G_home, G_ker, G_wt at geometry R."""
    nat = grid.natoms
    xyz = grid.points(R)
    W, dW = grid.weights(R)
    chi, dchi, ddchi = aos(xyz, R)
    rho, grad, hess = density(chi, dchi, ddchi)
    sigma = np.sum(grad * grad, axis=0)
    _, vrho, vsig, f, ker = vv10(W, xyz, rho, sigma, grid.home, nat)

    # G_AO: gga_gradient_from_potentials (t_rho = W v_rho, c_b = 2 W v_sigma d_b rho)
    t_rho = W * vrho
    c = 2 * W * vsig * grad
    m = DMAT @ chi
    g_ao = np.zeros((nat, 3))
    for ax in range(3):
        part = np.sum(m * t_rho * dchi[ax], axis=1)
        for b in range(3):
            part += np.sum(dchi[ax] * (DMAT @ dchi[b]) * c[b], axis=1)
            part += np.sum(ddchi[ax, b] * m * c[b], axis=1)
        for mu in range(NBF):
            g_ao[BF_ATOM[mu], ax] -= 2 * part[mu]

    # G_home: the integrand under a point that rides its atom
    dsig = 2 * np.einsum("bg,abg->ag", grad, hess)
    loc = W * (vrho * grad + vsig * dsig)  # (3, n)
    g_home = np.array([loc[:, grid.home == a].sum(axis=1) for a in range(nat)])

    # G_wt: dE/dW_i = rho_i (beta + f_i)
    g_wt = np.einsum("g,gcx->cx", rho * (BETA + f), dW)
    return g_ao, g_home, ker, g_wt, (W, xyz, rho, sigma, vrho, vsig, f)


def fd(fun, R, h):
    out = np.zeros_like(R)
    for a in range(R.shape[0]):
        for x in range(3):
            rp, rm = R.copy(), R.copy()
            rp[a, x] += h
            rm[a, x] -= h
            out[a, x] = (fun(rp) - fun(rm)) / (2 * h)
    return out


def main():
    global BASIS, NBF, BF_ATOM, DMAT
    sympy_claims()

    grid = Grid((6, 4, 8), 2)
    g_ao, g_home, g_ker, g_wt, (W, xyz, rho, sigma, vrho, vsig, f) = gradient_terms(
        R0, grid
    )
    scale = np.abs(g_ao).max()

    # P1: per-point potentials are dE/drho_i / W_i, dE/dsigma_i / W_i
    worst = 0.0
    for i in np.argsort(W * rho)[-6::2]:
        for which, v in ((0, vrho), (1, vsig)):
            h = 1e-5 * (rho[i] if which == 0 else sigma[i])
            ep, em = [], []
            for sgn, store in ((1, ep), (-1, em)):
                r_, s_ = rho.copy(), sigma.copy()
                (r_ if which == 0 else s_)[i] += sgn * h
                store.append(vv10(W, xyz, r_, s_)[0])
            num = (ep[0] - em[0]) / (2 * h)
            worst = max(worst, abs(num - W[i] * v[i]) / max(abs(num), 1e-300))
    check(
        "P1 W v_rho, W v_sigma = dE/drho, dE/dsigma (claim potentials)",
        worst < 1e-5,
        f"max rel {worst:.1e}",
    )

    # G1: code term = FD with the grid fixed in space
    h = 1e-4
    fd_fixed = fd(lambda R: energy_at(R, R0, grid), R0, h)
    e1 = np.abs(g_ao - fd_fixed).max()
    check(
        "G1 G_AO = FD at fixed grid (claim code)",
        e1 < 1e-7 * scale,
        f"max|diff| {e1:.1e} (|G| {scale:.1e})",
    )

    # G2: full gradient = FD of the moving-grid energy
    fd_moving = fd(lambda R: energy_at(R, R, grid), R0, h)
    full = g_ao + g_home + g_ker + g_wt
    e2 = np.abs(full - fd_moving).max()
    check(
        "G2 G_AO + G_home + G_ker + G_wt = FD of moving-grid energy (claim full)",
        e2 < 1e-7 * scale,
        f"max|diff| {e2:.1e}",
    )

    # G3: translation invariance
    net_full = np.abs(full.sum(axis=0)).max()
    net_ao = np.abs(g_ao.sum(axis=0)).max()
    check(
        "G3 full gradient sums to zero, G_AO does not (claim invariance)",
        net_full < 1e-12 * scale and net_ao > 1e3 * net_full,
        f"net full {net_full:.1e}, net G_AO {net_ao:.1e}",
    )

    # G4: negative controls
    miss_ker = np.abs(g_ao + g_home + g_wt - fd_moving).max()
    g_wt_half = np.einsum("g,gcx->cx", rho * (BETA + 0.5 * f), grid.weights(R0)[1])
    miss_wt = np.abs(g_ao + g_home + g_ker + g_wt_half - fd_moving).max()
    check(
        "G4 dropping G_ker or using rho (beta + f/2) misses (negative controls)",
        min(miss_ker, miss_wt) > 100 * max(e2, 1e-14),
        f"no G_ker {miss_ker:.1e}, half-f weight {miss_wt:.1e}, vs {e2:.1e}",
    )

    # G5: one atom, all AOs moved with it: E_moving is R-independent
    one = Grid((6, 4, 8), 1)
    keep = BF_ATOM == 0
    saved = (BASIS, NBF, BF_ATOM, DMAT)
    BASIS = [b for b in BASIS if b[0] == 0]
    NBF, BF_ATOM, DMAT = len(BASIS), BF_ATOM[keep], DMAT[np.ix_(keep, keep)]
    r1 = R0[:1]
    a_ao, a_home, a_ker, a_wt, _ = gradient_terms(r1, one)
    a_fd = fd(lambda R: energy_at(R, R, one), r1, h)
    rel = np.abs(a_ao + a_home).max() / np.abs(a_ao).max()
    check(
        "G5 one atom: G_AO = -G_home, full = 0 (anchor)",
        rel < 1e-12
        and np.abs(a_ker).max() < 1e-12 * np.abs(a_ao).max()
        and np.abs(a_wt).max() == 0
        and np.abs(a_fd).max() < 1e-9 * np.abs(a_ao).max(),
        f"|G_AO + G_home|/|G_AO| {rel:.1e}, |G_AO| {np.abs(a_ao).max():.1e}, "
        f"FD {np.abs(a_fd).max():.1e}, |G_ker| {np.abs(a_ker).max():.1e}",
    )
    BASIS, NBF, BF_ATOM, DMAT = saved

    # G6: refinement
    rows = []
    for lev in ((6, 4, 8), (8, 6, 12), (12, 8, 16)):
        gr = Grid(lev, 2)
        t_ao, t_home, t_ker, t_wt, _ = gradient_terms(R0, gr)
        dropped = np.abs(t_home + t_ker + t_wt).max()
        rows.append((len(gr.w0), dropped, np.abs(t_ao.sum(axis=0)).max()))
    print("    npts   |dropped|   |net G_AO|")
    for n, dmax, net in rows:
        print(f"    {n:5d}   {dmax:.2e}    {net:.2e}")
    d = [r[1] for r in rows]
    nets = [r[2] for r in rows]
    shrink = all(a > b for a, b in pairwise(d)) and all(
        a > b for a, b in pairwise(nets)
    )
    check(
        "G6 dropped part and net force shrink with the grid (claim limit)",
        shrink and d[-1] < 0.1 * d[0] and nets[-1] < 0.1 * nets[0],
        f"dropped {d[0]:.1e} -> {d[-1]:.1e}, net {nets[0]:.1e} -> {nets[-1]:.1e}",
    )

    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all VV10 gradient identities verified")


if __name__ == "__main__":
    main()
