#!/usr/bin/env python3
"""Sympy/numpy checks of the many-body dispersion (MBD) code.

Theory: `book/src/25-dispersion/03-mbd.md#mbd`.
Code: crates/ferric-rpa/src/dispersion/mbd_rsscs.rs (MBD@rsSCS: t_bare, t_gg,
rsscs_screen, rsscs_forward, rsscs_backward, t_bare_vjp, t_gg_vjp,
fermi_partials, mbd_rsscs_gradient), dispersion/mbd_scf.rs
(orthonormality_term, the lattice response of mbd_rsscs_impl) and
dispersion/mbd.rs (dipole_coupling_tensor, mbd_screen, the full-range SCS of
the C6 path). Book: book/src/25-dispersion/03-mbd.md. Every check prints "FAIL [<name>]"
when it fails, and the script exits nonzero if any did.

Notation (atomic units). Per atom A: alpha_A(iu) = alpha_A / (1 + (u/w_A)^2),
w_A = 4 C6_A / (3 alpha_A^2), Gaussian width
sigma_A = (sqrt(2/pi) alpha_A(iu) / 3)^(1/3), sigma_AB = sqrt(sigma_A^2 + sigma_B^2).
Dipole tensors: T_bare = (delta R^2 - 3 R R^T)/R^5,
T_GG = (erf z - th) T_bare + 2 z^2 th R R^T / R^5, z = R/sigma_AB,
th = 2 z e^(-z^2)/sqrt(pi). Screening: C = diag(1/alpha(iu)) + (1 - f) T_GG,
alpha^SCS_A = (1/3) Tr sum_B (C^-1)_AB. Energy:
H = diag(w^2) + w_A w_B sqrt(alpha_A alpha_B) f T_bare,
E = 1/2 sum_p sqrt(lambda_p) - 3/2 sum_A w_A.

  1  t-bare-is-hessian (sympy): T_bare = -grad grad (1/R).
  2  t-gg-is-hessian (sympy): the coded T_GG = -grad grad (erf(R/s)/R), and
     mbd.rs's (zeta, eta) form of dipole_coupling_tensor is the same tensor.
  3  sigma-self-energy (sympy): with sigma_A as coded, the R -> 0 limit of
     -grad grad erf(R/sigma_AA)/R, sigma_AA = sqrt(2) sigma_A, is
     (1/alpha_A) * identity: the diagonal 1/alpha of C is the self-interaction
     of the atom's Gaussian dipole density.
  4  scs-trivial-limit (numpy): with no coupling the screening returns
     alpha^SCS = alpha at every node, and the code's C6 sum on the libMBD grid
     (15 Gauss-Legendre nodes, u = 0.6 (1+x)/(1-x)) reproduces the TS C6
     (measured 1e-8 relative; bar 1e-6), so omega^rsSCS = omega and
     R_vdW^rsSCS = R_vdW.
  5  scs-dimer-closed-form (numpy): for a homonuclear dimer on the z axis the
     row-block contraction gives alpha^SCS = (1/3) sum_d alpha/(1 + alpha t_d),
     t = (1 - f) (T_GG,xx, T_GG,yy, T_GG,zz), the dimer's screened response.
  6  qho-vs-frequency (numpy, independent route): the coupled-oscillator
     energy 1/2 sum sqrt(lambda) - 3/2 sum w equals
     (1/2 pi) int_0^inf ln det(1 + A(iu) T) du, A(iu) = diag(alpha(iu)),
     T = f T_bare, on a random 4-atom geometry (2000-point quadrature).
  7  qho-second-order-london (sympy): for two atoms the coupling-squared term
     of the oscillator energy is -f^2 C6_AB / R^6 with the London
     C6_AB = (3/2) alpha_A alpha_B w_A w_B / (w_A + w_B), the Casimir-Polder
     C6 of two single poles (scripts/rpa_c6_polarizability_sympy.py, check
     london-integral).
  8  energy-adjoint (numpy): dE/dH = 1/4 H^(-1/2), the first step of
     rsscs_backward (G = 1/4 V diag(lambda^-1/2) V^T), against a central
     finite difference of E in every element of a symmetric perturbation.

  9  vjp-partials (sympy): t_bare_vjp and t_gg_vjp (the vector-Jacobian
     products of rsscs_backward) equal sympy's gradient of sum M_ij T_ij in
     d and sigma at random points and random M, to 1e-12.
 10  gradient-vs-fd (numpy, independent route): rsscs_backward transcribed
     step by step (adjoint seed, H, omega~/R~, the per-node screening
     adjoint C^-bar = -B^T B-dot B^T, the TS map) equals a Richardson
     difference of the transcribed forward energy in every coordinate,
     alpha_0, C6 and R_vdW. Bar: 50x the round-off floor
     eps S / (h max|g|), S = 1/2 sum sqrt(lambda) ~ 5 >> |E|.
 11  gradient-translation: the explicit force sums to zero over atoms.
 12  ratio-chain-rule (mbd_rsscs_gradient): dE/dr_A =
     a_free dE/dalpha_0 + 2 r C6_free dE/dC6 + 1/3 r^(-2/3) R_free dE/dR_vdW
     against a Richardson difference in r_A.
 13  orthonormality-term (mbd_scf.rs orthonormality_term): with the occupied
     orbitals Lowdin-orthonormalised in a moving metric S(x), d/dx Tr[V D]
     equals -1/2 Tr[V D S' D] (closed shell), -sum_s Tr[V D_s S' D_s] (UKS,
     each spin separately) and -Tr[S' W_Q], W_Q = D_a V D_a + P_c V P_c
     + 1/2 (P_c V P_o + P_o V P_c) (ROKS, closed and open as one set).
 14  lattice-term (mbd_scf.rs lattice response): on a toy Hirshfeld volume
     (fixed density matrix over atom-centred Gaussians, a lattice that moves
     with the centroid), the total derivative equals the fixed-lattice term
     minus 1/N of its sum over atoms, with the lattice term 7% of the total.

Mutations (each turns the named check red):
  - the -3/2 sum w reference energy as -1 sum w                  [qho-vs-frequency]
  - T_GG's 2 z^2 th R R^T term without its factor 2              [t-gg-is-hessian]
  - the width prefactor sqrt(2/pi)/3 as 1/3                      [sigma-self-energy]
  - the contraction over the diagonal block B = A only           [scs-dimer-closed-form]
  - the adjoint 1/4 lambda^-1/2 as 1/2 lambda^-1/2               [energy-adjoint]
  - t_gg_vjp's dg2/dsigma (3 - 2 z^2) as (3 + 2 z^2)            [vjp-partials]
  - step 4's C6 adjoint 6/pi as 3/pi                             [gradient-vs-fd]
  - step 3's R~ -> alpha~ adjoint R~/(3 alpha~) as R~/(2 alpha~) [gradient-vs-fd]
  - the u = 0 node's alpha~_0 adjoint dropped                    [gradient-vs-fd]
  - the ratio chain rule's r^(-2/3) as r^(-1/3)                  [ratio-chain-rule]
  - the closed-shell -1/2 Tr[V D S' D] as -1 Tr[...]             [orthonormality-term]
  - the ROKS closed-open cross term of W_Q dropped               [orthonormality-term]
  - the lattice term dropped                                     [lattice-term]

Run: python3 scripts/mbd_coupled_dipole_sympy.py  (needs sympy, numpy)
"""

import sys
from math import erf

import numpy as np
import sympy as sp

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# ------------------------------------------------- code transcriptions
FRAC = sp.sqrt(2 / sp.pi) / 3  # mbd_rsscs.rs rsscs_screen `frac`
FRAC_F = float(FRAC)
SQRT_PI = float(sp.sqrt(sp.pi))


def t_bare(d):
    """mbd_rsscs.rs t_bare: (delta R^2 - 3 R R^T)/R^5."""
    r = np.linalg.norm(d)
    return (np.eye(3) * r * r - 3.0 * np.outer(d, d)) / r**5


def t_gg(d, sigma_ab):
    """mbd_rsscs.rs t_gg."""
    r = np.linalg.norm(d)
    zeta = r / sigma_ab
    theta = 2.0 * zeta / SQRT_PI * np.exp(-zeta * zeta)
    erf_theta = erf(zeta) - theta
    extra = 2.0 * zeta * zeta * theta / r**5
    return erf_theta * t_bare(d) + extra * np.outer(d, d)


def fermi(r, s, a):
    return 1.0 / (1.0 + np.exp(-a * (r / s - 1.0)))


def gl_map(n, scale):
    """quadrature.rs gauss_legendre_nodes: u = L (1+x)/(1-x)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return scale * (1 + x) / (1 - x), w * 2 * scale / (1 - x) ** 2


def mbd_freq_grid(n=15):
    """mbd_rsscs.rs mbd_freq_grid: a u = 0 node of weight 0, then n nodes."""
    u, w = gl_map(n, 0.6)
    return np.concatenate([[0.0], u[::-1]]), np.concatenate([[0.0], w[::-1]])


def rsscs_screen(pos, a0, c6, rvdw, beta, fa, freqs, couple=True):
    """mbd_rsscs.rs rsscs_screen: alpha^rsSCS(iu_k) as [k][A]."""
    n = len(pos)
    om = 4.0 / 3.0 * c6 / a0**2
    out = []
    for u in freqs:
        alpha = a0 / (1.0 + (u / om) ** 2)
        sigma = np.cbrt(FRAC_F * alpha)
        c = np.zeros((3 * n, 3 * n))
        for a in range(n):
            c[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] = np.eye(3) / alpha[a]
            for b in range(a):
                if not couple:
                    continue
                d = pos[a] - pos[b]
                r = np.linalg.norm(d)
                sr = 1.0 - fermi(r, beta * (rvdw[a] + rvdw[b]), fa)
                t = sr * t_gg(d, np.hypot(sigma[a], sigma[b]))
                c[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = t
                c[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = t.T
        cinv = np.linalg.inv(c)
        a_rs = np.array(
            [
                sum(cinv[3 * a + i, 3 * b + i] for b in range(n) for i in range(3))
                / 3.0
                for a in range(n)
            ]
        )
        out.append(a_rs)
    return np.array(out)


def qho_matrix(pos, a0, om, rvdw, beta, fa, scale=1.0):
    """mbd_rsscs.rs rsscs_forward: H = diag(w^2) + w w sqrt(a a) f T_bare."""
    n = len(pos)
    h = np.zeros((3 * n, 3 * n))
    for a in range(n):
        h[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] = np.eye(3) * om[a] ** 2
        for b in range(a):
            d = pos[a] - pos[b]
            r = np.linalg.norm(d)
            f = fermi(r, beta * (rvdw[a] + rvdw[b]), fa)
            v = scale * om[a] * om[b] * np.sqrt(a0[a] * a0[b]) * f * t_bare(d)
            h[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = v
            h[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = v.T
    return h


def qho_energy(h, om):
    lam = np.linalg.eigvalsh(h)
    return 0.5 * np.sum(np.sqrt(lam)) - 1.5 * om.sum()


def energy_adjoint(h):
    """rsscs_backward step 1: G = dE/dH = 1/4 V diag(lambda^-1/2) V^T."""
    lam, v = np.linalg.eigh(h)
    return (v * (0.25 / np.sqrt(lam))) @ v.T


# ---------------------------------------------------------- 1-3 symbolic
def symbolic():
    x, y, z = sp.symbols("x y z", real=True)
    s = sp.symbols("s", positive=True)
    rv = sp.Matrix([x, y, z])
    r = sp.sqrt(x * x + y * y + z * z)

    def hess(fun):
        return sp.Matrix(3, 3, lambda i, j: -sp.diff(fun, rv[i], rv[j]))

    tb_code = (sp.eye(3) * r**2 - 3 * rv * rv.T) / r**5
    check(
        "t-bare-is-hessian",
        sp.simplify(hess(1 / r) - tb_code) == sp.zeros(3, 3),
        "T_bare = -grad grad 1/R",
    )

    zeta = r / s
    ref = hess(sp.erf(r / s) / r)
    # mbd.rs dipole_coupling_tensor: zeta*(delta - 3nn)/r^3 + eta*3nn/r^3.
    nn = rv * rv.T / r**2
    zeta_mbd = sp.erf(zeta) - 2 * zeta / sp.sqrt(sp.pi) * sp.exp(-(zeta**2))
    eta_mbd = 4 * zeta**3 / (3 * sp.sqrt(sp.pi)) * sp.exp(-(zeta**2))
    tgg_mbd = zeta_mbd * (sp.eye(3) - 3 * nn) / r**3 + eta_mbd * 3 * nn / r**3
    pts = [
        tuple(sp.Rational(v) for v in p)
        for p in (
            ("0.7", "-0.3", "1.1", "0.9"),
            ("2", "0.5", "-0.4", "1.7"),
            ("-0.2", "1.3", "0.6", "0.4"),
        )
    ]
    worst = 0.0
    worst_mbd = 0.0
    for px, py, pz, ps in pts:
        sub = {x: px, y: py, z: pz, s: ps}
        exact = np.array(ref.subs(sub).evalf(30), dtype=float)
        coded = t_gg(np.array([px, py, pz], dtype=float), float(ps))
        diff_mbd = (tgg_mbd - ref).subs(sub).evalf(30)
        scale = np.max(np.abs(exact))
        worst = max(worst, np.max(np.abs(coded - exact)) / scale)
        worst_mbd = max(worst_mbd, float(max(abs(v) for v in diff_mbd)) / scale)
    check(
        "t-gg-is-hessian",
        worst < 1e-13 and worst_mbd < 1e-25,
        f"rel {worst:.1e} (mbd_rsscs t_gg, f64), {worst_mbd:.1e} (mbd.rs zeta/eta form)",
    )

    alpha = sp.symbols("alpha", positive=True)
    sigma_a = (FRAC * alpha) ** sp.Rational(1, 3)
    s_aa = sp.sqrt(2) * sigma_a
    rr = sp.symbols("R", positive=True)
    # zz element on the z axis: -d2/dz2 erf(z/s)/z at z -> 0.
    zz = -sp.diff(sp.erf(rr / s_aa) / rr, rr, 2)
    lim = sp.limit(zz, rr, 0)
    check(
        "sigma-self-energy",
        sp.simplify(lim - 1 / alpha) == 0,
        f"lim T_GG,zz = {sp.simplify(lim)}",
    )


# -------------------------------------------------------- 4-8 numerical
rng = np.random.default_rng(11)


def numeric():
    n = 4
    pos = rng.uniform(-3.0, 3.0, (n, 3))
    pos[1] += [4.0, 0.0, 0.0]
    pos[2] += [0.0, 4.5, 0.0]
    pos[3] += [0.0, 0.0, 5.0]
    a0 = rng.uniform(4.0, 12.0, n)
    c6 = rng.uniform(5.0, 40.0, n)
    rvdw = rng.uniform(2.5, 3.8, n)
    beta, fa = 0.83, 6.0
    om = 4.0 / 3.0 * c6 / a0**2

    freqs, weights = mbd_freq_grid()
    plain = rsscs_screen(pos, a0, c6, rvdw, beta, fa, freqs, couple=False)
    alpha_free = a0[None, :] / (1.0 + (freqs[:, None] / om[None, :]) ** 2)
    c6_rs = 3.0 / np.pi * np.einsum("k,ka,ka->a", weights, plain, plain)
    om_rs = 4.0 / 3.0 * c6_rs / plain[0] ** 2
    rel_c6 = np.max(np.abs(c6_rs / c6 - 1.0))
    check(
        "scs-trivial-limit",
        np.max(np.abs(plain - alpha_free)) < 1e-12 * a0.max()
        and rel_c6 < 1e-6
        and np.max(np.abs(om_rs / om - 1.0)) < 1e-6
        and np.max(np.abs(plain[0] / a0 - 1.0)) < 1e-14,
        f"alpha exact, C6 {rel_c6:.1e} rel on the 15-node grid",
    )

    # Homonuclear dimer on z.
    ad, cd, rd = 9.0, 30.0, 3.2
    dz = np.array([0.0, 0.0, 5.5])
    pos2 = np.array([[0.0, 0.0, 0.0], dz])
    got = rsscs_screen(
        pos2,
        np.array([ad, ad]),
        np.array([cd, cd]),
        np.array([rd, rd]),
        beta,
        fa,
        freqs,
    )
    want = []
    omd = 4.0 / 3.0 * cd / ad**2
    for u in freqs:
        al = ad / (1.0 + (u / omd) ** 2)
        sg = np.cbrt(FRAC_F * al)
        t = (1.0 - fermi(5.5, beta * 2 * rd, fa)) * np.diag(t_gg(dz, np.hypot(sg, sg)))
        want.append(np.mean(al / (1.0 + al * t)))
    want = np.array(want)
    err = np.max(np.abs(got[:, 0] - want) / want)
    check(
        "scs-dimer-closed-form",
        err < 1e-13 and np.max(np.abs(got[:, 1] - want) / want) < 1e-13,
        f"rel {err:.1e}",
    )

    # Oscillator energy vs the frequency integral of ln det(1 + A T).
    h = qho_matrix(pos, a0, om, rvdw, beta, fa)
    e_qho = qho_energy(h, om)
    tmat = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for b in range(a):
            d = pos[a] - pos[b]
            f = fermi(np.linalg.norm(d), beta * (rvdw[a] + rvdw[b]), fa)
            tmat[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = f * t_bare(d)
            tmat[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = f * t_bare(d).T
    uq, wq = gl_map(2000, 0.5)
    e_freq = 0.0
    for u, w in zip(uq, wq):
        al = np.repeat(a0 / (1.0 + (u / om) ** 2), 3)
        sign, logdet = np.linalg.slogdet(np.eye(3 * n) + al[:, None] * tmat)
        assert sign > 0
        e_freq += w * logdet
    e_freq /= 2.0 * np.pi
    check(
        "qho-vs-frequency",
        abs(e_qho - e_freq) < 1e-10 * abs(e_qho) and abs(e_qho) > 1e-5,
        f"oscillators {e_qho:.12e}, frequency integral {e_freq:.12e}",
    )

    # dE/dH = 1/4 H^(-1/2).
    g = energy_adjoint(h)
    pert = rng.standard_normal(h.shape)
    pert = 0.5 * (pert + pert.T)
    step = 1e-6

    def e_of(m):
        return 0.5 * np.sum(np.sqrt(np.linalg.eigvalsh(m)))

    fd = (e_of(h + step * pert) - e_of(h - step * pert)) / (2 * step)
    an = np.sum(g * pert)
    check(
        "energy-adjoint",
        abs(fd - an) < 1e-7 * abs(an),
        f"<G, dH> {an:.12e}, FD {fd:.12e}",
    )


def london_second_order():
    wa, wb, aa, ab, f, r, lam = sp.symbols(
        "w_A w_B alpha_A alpha_B f R lambda", positive=True
    )
    # Dimer on z: T_bare = diag(1, 1, -2)/R^3; each direction is a 2x2 block.
    total = 0
    for t in (1, 1, -2):
        g = lam * wa * wb * sp.sqrt(aa * ab) * f * t / r**3
        # For a 2x2 positive-definite M: sqrt(l1) + sqrt(l2)
        # = sqrt(tr M + 2 sqrt(det M)), free of eigenvalue ordering.
        tr, det = wa**2 + wb**2, wa**2 * wb**2 - g**2
        total += sp.Rational(1, 2) * sp.sqrt(tr + 2 * sp.sqrt(det))
    total -= wa + wb  # 3/2 (w_A + w_B) split over the three directions
    second = sp.series(total, lam, 0, 3).removeO().coeff(lam, 2)
    london = sp.Rational(3, 2) * aa * ab * wa * wb / (wa + wb)
    check(
        "qho-second-order-london",
        sp.simplify(sp.factor(second) + f**2 * london / r**6) == 0,
        f"{sp.factor(sp.simplify(second))}",
    )


# ------------------------------------------- 9-14 gradient (rsscs_backward, mbd_scf.rs)
def fermi_partials(r, s, a):
    """mbd_rsscs.rs fermi_partials: (df/dR, df/dS)."""
    e = np.exp(-a * (r / s - 1.0))
    fp = e / ((1.0 + e) * (1.0 + e)) if np.isfinite(e) else 0.0
    return fp * a / s, -fp * a * r / (s * s)


def t_bare_vjp(d, r, m):
    """mbd_rsscs.rs t_bare_vjp: (L, dL/dd, q, M d) for L = sum M_ij T_bare,ij."""
    ms = 0.5 * (m + m.T)
    tr = np.trace(ms)
    md = ms @ d
    q = d @ md
    r2 = r * r
    r5 = r2 * r2 * r
    r7 = r5 * r2
    lval = tr / (r2 * r) - 3.0 * q / r5
    g = -3.0 * tr * d / r5 - 6.0 * md / r5 + 15.0 * q * d / r7
    return lval, g, q, md


def t_gg_vjp(d, r, sigma, m):
    """mbd_rsscs.rs t_gg_vjp: (dL/dd, dL/dsigma) for L = sum M_ij T_GG,ij."""
    lb, dlb, q, md = t_bare_vjp(d, r, m)
    zeta = r / sigma
    theta = 2.0 * zeta / SQRT_PI * np.exp(-zeta * zeta)
    g1 = erf(zeta) - theta
    r5 = r**5
    z2t = zeta * zeta * theta
    g2 = 2.0 * z2t / r5
    dg1_dr = 2.0 * z2t / r
    dg1_ds = -2.0 * z2t / sigma
    dg2_dr = -4.0 * z2t * (1.0 + zeta * zeta) / (r5 * r)
    dg2_ds = -2.0 * z2t * (3.0 - 2.0 * zeta * zeta) / (sigma * r5)
    radial = (lb * dg1_dr + q * dg2_dr) / r
    gd = g1 * dlb + 2.0 * g2 * md + radial * d
    return gd, lb * dg1_ds + q * dg2_ds


def rsscs_forward(pos, a0, c6, rvdw, beta, fa, n_freq=15):
    """mbd_rsscs.rs rsscs_forward with keep = true: (E, tape)."""
    n = len(pos)
    freqs, weights = mbd_freq_grid(n_freq)
    om = 4.0 / 3.0 * c6 / a0**2
    a_dyn, binv = [], []
    for u in freqs:
        alpha = a0 / (1.0 + (u / om) ** 2)
        sigma = np.cbrt(FRAC_F * alpha)
        c = np.zeros((3 * n, 3 * n))
        for a in range(n):
            c[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] = np.eye(3) / alpha[a]
            for b in range(a):
                d = pos[a] - pos[b]
                r = np.linalg.norm(d)
                sr = 1.0 - fermi(r, beta * (rvdw[a] + rvdw[b]), fa)
                t = sr * t_gg(d, np.hypot(sigma[a], sigma[b]))
                c[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = t
                c[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = t.T
        cinv = np.linalg.inv(c)
        a_dyn.append(
            np.array(
                [
                    sum(
                        cinv[3 * a + i, 3 * bb + i] for bb in range(n) for i in range(3)
                    )
                    / 3.0
                    for a in range(n)
                ]
            )
        )
        binv.append(cinv)
    a0t = a_dyn[0]
    c6t = np.array(
        [
            3.0 / np.pi * sum(w * ak[a] ** 2 for w, ak in zip(weights, a_dyn))
            for a in range(n)
        ]
    )
    rt = rvdw * np.cbrt(a0t / a0)
    omt = 4.0 / 3.0 * c6t / a0t**2
    h = qho_matrix(pos, a0t, omt, rt, beta, fa)
    lam, vec = np.linalg.eigh(h)
    e = 0.5 * np.sum(np.sqrt(lam)) - 1.5 * omt.sum()
    tape = {
        "freqs": freqs,
        "weights": weights,
        "a_dyn": a_dyn,
        "binv": binv,
        "lam": lam,
        "vec": vec,
        "a0t": a0t,
        "c6t": c6t,
        "rt": rt,
        "omt": omt,
    }
    return e, tape


def rsscs_backward(pos, a0, c6, rvdw, beta, fa, tp):
    """mbd_rsscs.rs rsscs_backward, steps 1-5, in the Rust loop order."""
    n = len(pos)
    d_pos = np.zeros((n, 3))
    d_a0, d_c6, d_r = np.zeros(n), np.zeros(n), np.zeros(n)
    # Step 1: G = 1/4 V diag(lambda^-1/2) V^T.
    g = (tp["vec"] * (0.25 / np.sqrt(tp["lam"]))) @ tp["vec"].T
    # Step 2: adjoint through H.
    a0t, rt, omt = tp["a0t"], tp["rt"], tp["omt"]
    om_bar = np.full(n, -1.5)
    a0t_bar, rt_bar = np.zeros(n), np.zeros(n)
    for a in range(n):
        om_bar[a] += 2.0 * omt[a] * sum(g[3 * a + i, 3 * a + i] for i in range(3))
        for b in range(a):
            d = pos[a] - pos[b]
            r = np.linalg.norm(d)
            s_ = beta * (rt[a] + rt[b])
            f = fermi(r, s_, fa)
            sq = np.sqrt(a0t[a] * a0t[b])
            pref = omt[a] * omt[b] * sq * f
            tb = t_bare(d)
            vbar = (
                g[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
                + g[3 * b : 3 * b + 3, 3 * a : 3 * a + 3].T
            )
            pbar = np.sum(vbar * tb)
            _, gd, _, _ = t_bare_vjp(d, r, pref * vbar)
            om_bar[a] += pbar * omt[b] * sq * f
            om_bar[b] += pbar * omt[a] * sq * f
            half = 0.5 * pbar * omt[a] * omt[b] * sq * f
            a0t_bar[a] += half / a0t[a]
            a0t_bar[b] += half / a0t[b]
            fbar = pbar * omt[a] * omt[b] * sq
            df_dr, df_ds = fermi_partials(r, s_, fa)
            rt_bar[a] += beta * fbar * df_ds
            rt_bar[b] += beta * fbar * df_ds
            gx = gd + fbar * df_dr * d / r
            d_pos[a] += gx
            d_pos[b] -= gx
    # Step 3: omega~ and R~.
    c6t_bar = om_bar * 4.0 / (3.0 * a0t * a0t)
    a0t_bar -= om_bar * 2.0 * omt / a0t
    a0t_bar += rt_bar * rt / (3.0 * a0t)
    d_r += rt_bar * rt / rvdw
    d_a0 -= rt_bar * rt / (3.0 * a0)
    # Step 4: per frequency node.
    om_ts = 4.0 / 3.0 * c6 / a0**2
    om_ts_bar = np.zeros(n)
    for k, u in enumerate(tp["freqs"]):
        bm, ak = tp["binv"][k], tp["a_dyn"][k]
        abar = c6t_bar * (6.0 / np.pi) * tp["weights"][k] * ak
        if k == 0:
            abar = abar + a0t_bar
        w3 = np.zeros((3 * n, 3))
        z3 = np.zeros((3 * n, 3))
        for bb in range(n):
            for i in range(3):
                w3[:, i] += bm[:, 3 * bb + i]
                z3[:, i] += bm[3 * bb + i, :] * abar[bb] / 3.0
        cbar = -(z3 @ w3.T)
        alpha = a0 / (1.0 + (u / om_ts) ** 2)
        sigma = np.cbrt(FRAC_F * alpha)
        alpha_bar, sigma_bar = np.zeros(n), np.zeros(n)
        for a in range(n):
            alpha_bar[a] -= (
                sum(cbar[3 * a + i, 3 * a + i] for i in range(3)) / alpha[a] ** 2
            )
            for b in range(a):
                d = pos[a] - pos[b]
                r = np.linalg.norm(d)
                s_ = beta * (rvdw[a] + rvdw[b])
                sr = 1.0 - fermi(r, s_, fa)
                sab = np.hypot(sigma[a], sigma[b])
                t = t_gg(d, sab)
                vbar = (
                    cbar[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
                    + cbar[3 * b : 3 * b + 3, 3 * a : 3 * a + 3].T
                )
                srbar = np.sum(vbar * t)
                gd, gsig = t_gg_vjp(d, r, sab, sr * vbar)
                fbar = -srbar
                df_dr, df_ds = fermi_partials(r, s_, fa)
                d_r[a] += beta * fbar * df_ds
                d_r[b] += beta * fbar * df_ds
                gx = gd + fbar * df_dr * d / r
                d_pos[a] += gx
                d_pos[b] -= gx
                sigma_bar[a] += gsig * sigma[a] / sab
                sigma_bar[b] += gsig * sigma[b] / sab
        alpha_bar += sigma_bar * sigma / (3.0 * alpha)
        d_a0 += alpha_bar * alpha / a0
        om_ts_bar += alpha_bar * 2.0 * alpha * alpha * u * u / (a0 * om_ts**3)
    # Step 5: omega = 4/3 C6/alpha0^2.
    d_c6 += om_ts_bar * 4.0 / (3.0 * a0 * a0)
    d_a0 -= om_ts_bar * 2.0 * om_ts / a0
    return d_pos, d_a0, d_c6, d_r


def richardson(f, h):
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(2 * h) - f(-2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


def vjp_symbolic():
    """t_gg_vjp / t_bare_vjp against sympy's gradient of L = sum M_ij T_ij."""
    x = sp.Matrix(sp.symbols("d0:3", real=True))
    sg = sp.Symbol("sigma", positive=True)
    rr = sp.sqrt(x.dot(x))
    z = rr / sg
    th = 2 * z / sp.sqrt(sp.pi) * sp.exp(-(z**2))
    tb = (sp.eye(3) * rr**2 - 3 * x * x.T) / rr**5
    tgg = (sp.erf(z) - th) * tb + 2 * z**2 * th * x * x.T / rr**5
    worst = 0.0
    for _ in range(3):
        d = rng.uniform(-2.0, 2.0, 3)
        sig = rng.uniform(0.8, 2.0)
        m = rng.standard_normal((3, 3))
        msym = sp.Matrix(m)
        sub = {x[0]: d[0], x[1]: d[1], x[2]: d[2], sg: sig}
        for tens, coded in ((tb, "bare"), (tgg, "gg")):
            lval = sum(msym[i, j] * tens[i, j] for i in range(3) for j in range(3))
            gd = np.array([float(sp.diff(lval, xi).subs(sub).evalf(25)) for xi in x])
            r = np.linalg.norm(d)
            if coded == "bare":
                got_d = t_bare_vjp(d, r, m)[1]
                err = np.max(np.abs(got_d - gd)) / np.max(np.abs(gd))
            else:
                gs = float(sp.diff(lval, sg).subs(sub).evalf(25))
                got_d, got_s = t_gg_vjp(d, r, sig, m)
                err = max(
                    np.max(np.abs(got_d - gd)) / np.max(np.abs(gd)),
                    abs(got_s - gs) / abs(gs),
                )
            worst = max(worst, err)
    check(
        "vjp-partials",
        worst < 1e-12,
        f"t_bare_vjp, t_gg_vjp vs sympy gradient: rel {worst:.1e}",
    )


def gradient_numeric():
    n = 4
    pos = rng.uniform(-3.0, 3.0, (n, 3))
    pos[1] += [4.0, 0.0, 0.0]
    pos[2] += [0.0, 4.5, 0.0]
    pos[3] += [0.0, 0.0, 5.0]
    a0 = rng.uniform(4.0, 12.0, n)
    c6 = rng.uniform(5.0, 40.0, n)
    rvdw = rng.uniform(2.5, 3.8, n)
    beta, fa = 0.83, 6.0
    e0, tp = rsscs_forward(pos, a0, c6, rvdw, beta, fa)
    d_pos, d_a0, d_c6, d_r = rsscs_backward(pos, a0, c6, rvdw, beta, fa, tp)

    def e_of(**kw):
        p = {"pos": pos, "a0": a0, "c6": c6, "rvdw": rvdw}
        p.update(kw)
        return rsscs_forward(p["pos"], p["a0"], p["c6"], p["rvdw"], beta, fa)[0]

    worst = {}
    for name, base, an, h in (
        ("positions", pos, d_pos, 1e-3),
        ("alpha_0", a0, d_a0, 1e-3),
        ("C6", c6, d_c6, 1e-3),
        ("R_vdW", rvdw, d_r, 1e-3),
    ):
        fd = np.zeros_like(base)
        for idx in np.ndindex(base.shape):

            def f(t, idx=idx, name=name, base=base):
                x = base.copy()
                x[idx] += t
                key = {
                    "positions": "pos",
                    "alpha_0": "a0",
                    "C6": "c6",
                    "R_vdW": "rvdw",
                }[name]
                return e_of(**{key: x})

            fd[idx] = richardson(f, h * max(1.0, abs(base[idx])))
        # Round-off floor of the difference: E = S - (3/2) sum w with
        # S = 1/2 sum sqrt(lambda) ~ 5 >> |E|, so FD carries eps S / h.
        floor = (
            np.finfo(float).eps
            * 0.5
            * np.sum(np.sqrt(tp["lam"]))
            / (h * max(1.0, np.max(np.abs(base))) * np.max(np.abs(fd)))
        )
        worst[name] = (np.max(np.abs(an - fd)) / np.max(np.abs(fd)), floor)
    check(
        "gradient-vs-fd",
        all(e < 50.0 * fl for e, fl in worst.values()) and abs(e0) > 1e-5,
        ", ".join(f"{k} {e:.1e} (floor {fl:.0e})" for k, (e, fl) in worst.items()),
    )
    # Translation invariance of the explicit term (sum over atoms is zero).
    check(
        "gradient-translation",
        np.max(np.abs(d_pos.sum(axis=0))) < 1e-12 * np.max(np.abs(d_pos)) * n,
        f"|sum_A dE/dR_A| {np.max(np.abs(d_pos.sum(axis=0))):.1e}",
    )

    # mbd_rsscs_gradient: dE/dr = a_free dE/da0 + 2 r C6_free dE/dC6 + 1/3 r^-2/3 R_free dE/dR.
    a_free, c_free, r_free = a0 * 1.3, c6 * 1.7, rvdw * 1.1
    ratio = rng.uniform(0.6, 0.95, n)

    def params(rr):
        return rr * a_free, rr * rr * c_free, np.cbrt(rr) * r_free

    pa, pc, pr = params(ratio)
    _, tp2 = rsscs_forward(pos, pa, pc, pr, beta, fa)
    _, ga, gc, gr = rsscs_backward(pos, pa, pc, pr, beta, fa, tp2)
    d_ratio = (
        a_free * ga
        + 2.0 * ratio * c_free * gc
        + r_free / (3.0 * np.cbrt(ratio) ** 2) * gr
    )
    fd = np.zeros(n)
    for i in range(n):

        def f(t, i=i):
            rr = ratio.copy()
            rr[i] += t
            return rsscs_forward(pos, *params(rr), beta, fa)[0]

        fd[i] = richardson(f, 1e-3)
    err = np.max(np.abs(d_ratio - fd)) / np.max(np.abs(fd))
    check("ratio-chain-rule", err < 1e-7, f"rel {err:.1e}")


def orthonormality_terms():
    """mbd_scf.rs orthonormality_term: the derivative of Tr[V D] with the
    occupied orbitals Lowdin-orthonormalised in the moving metric S(x)."""
    nb, nc, no = 7, 2, 1
    a = rng.standard_normal((nb, nb))
    s0 = a @ a.T / nb + np.eye(nb)
    s1 = rng.standard_normal((nb, nb))
    s1 = 0.1 * (s1 + s1.T)
    v = rng.standard_normal((nb, nb))
    v = v + v.T
    w, q = np.linalg.eigh(s0)
    c0 = (q / np.sqrt(w)) @ q.T @ np.linalg.qr(rng.standard_normal((nb, nb)))[0]
    cocc = c0[:, : nc + no]  # orthonormal in s0

    def lowdin(c, s):
        m = c.T @ s @ c
        ww, qq = np.linalg.eigh(m)
        return c @ (qq / np.sqrt(ww)) @ qq.T

    def dens(x):
        c = lowdin(cocc, s0 + x * s1)
        pc, po = c[:, :nc] @ c[:, :nc].T, c[:, nc:] @ c[:, nc:].T
        return pc, po

    pc, po = dens(0.0)
    errs = {}
    # Closed shell: D = 2 (P_c + P_o) (all doubly occupied), term -1/2 Tr[V D S^x D].
    d = 2.0 * (pc + po)
    fd = richardson(lambda x: np.sum(v * 2.0 * sum(dens(x))), 1e-3)
    errs["closed"] = abs(fd - (-0.5 * np.sum(s1 * (d @ v @ d))))

    # UKS: D_s idempotent per spin, term -sum_s Tr[V D_s S^x D_s]; alpha = P_c + P_o
    # and beta = P_c orthonormalised SEPARATELY.
    def dens_u(x):
        ca = lowdin(cocc, s0 + x * s1)
        cb = lowdin(cocc[:, :nc], s0 + x * s1)
        return ca @ ca.T + cb @ cb.T

    da, db = pc + po, pc
    fd = richardson(lambda x: np.sum(v * dens_u(x)), 1e-3)
    errs["uks"] = abs(fd - (-np.sum(s1 * (da @ v @ da + db @ v @ db))))
    # ROKS: closed and open orthonormalised as ONE set; D_alpha + D_beta = 2 P_c + P_o.
    cross = pc @ v @ po
    w_q = da @ v @ da + pc @ v @ pc + 0.5 * (cross + cross.T)
    fd = richardson(lambda x: np.sum(v * (2.0 * dens(x)[0] + dens(x)[1])), 1e-3)
    errs["roks"] = abs(fd - (-np.sum(s1 * w_q)))
    check(
        "orthonormality-term",
        max(errs.values()) < 1e-9,
        ", ".join(f"{k} {e:.1e}" for k, e in errs.items()),
    )


def lattice_term():
    """mbd_scf.rs lattice response: the volume lattice is anchored to the atom
    centroid, so dv/dR_B = dv/dR_B|lattice - (1/N) sum_C dv/dR_C|lattice."""
    n = 3
    pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.3, -0.2], [-0.4, 1.6, 0.5]])
    coef = rng.uniform(0.5, 1.5, (n, n))
    coef = coef + coef.T  # a fixed "D" over one s Gaussian per atom
    expo = np.array([0.9, 0.6, 0.7])
    g1 = np.linspace(-3.0, 3.0, 15)
    base = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)

    def volumes(p, lattice_shift):
        pts = base + lattice_shift
        phi = np.exp(
            -expo[None, :] * np.sum((pts[:, None, :] - p[None, :, :]) ** 2, -1)
        )
        rho = np.einsum("gm,mn,gn->g", phi, coef, phi)
        pro = np.exp(-1.2 * np.sqrt(np.sum((pts[:, None, :] - p[None, :, :]) ** 2, -1)))
        wgt = pro / pro.sum(axis=1, keepdims=True)
        r3 = np.sum((pts[:, None, :] - p[None, :, :]) ** 2, -1) ** 1.5
        return np.einsum("ga,g,ga->a", wgt, rho, r3)

    cvec = rng.uniform(0.5, 1.5, n)  # c_A = dE/dv_A

    def e_full(p):
        return cvec @ volumes(p, p.mean(axis=0))

    shift0 = pos.mean(axis=0)
    full, fixed = np.zeros((n, 3)), np.zeros((n, 3))
    for b, x in np.ndindex(n, 3):

        def f_full(t, b=b, x=x):
            p = pos.copy()
            p[b, x] += t
            return e_full(p)

        def f_fixed(t, b=b, x=x):
            p = pos.copy()
            p[b, x] += t
            return cvec @ volumes(p, shift0)

        full[b, x] = richardson(f_full, 1e-3)
        fixed[b, x] = richardson(f_fixed, 1e-3)
    lattice = np.tile(-fixed.sum(axis=0) / n, (n, 1))
    err = np.max(np.abs(full - (fixed + lattice))) / np.max(np.abs(full))
    size = np.max(np.abs(lattice)) / np.max(np.abs(full))
    check(
        "lattice-term",
        err < 1e-8 and size > 1e-4,
        f"rel {err:.1e}; lattice term {size:.1e} of the gradient",
    )


symbolic()
numeric()
london_second_order()
vjp_symbolic()
gradient_numeric()
orthonormality_terms()
lattice_term()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall MBD identities hold")
