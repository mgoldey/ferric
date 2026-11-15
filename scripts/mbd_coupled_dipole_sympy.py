#!/usr/bin/env python3
"""Sympy/numpy checks of the many-body dispersion (MBD) code.

Code: crates/ferric-rpa/src/dispersion/mbd_rsscs.rs (MBD@rsSCS: t_bare, t_gg,
rsscs_screen, rsscs_forward, rsscs_backward) and dispersion/mbd.rs
(dipole_coupling_tensor, mbd_screen, the full-range SCS of the C6 path).
Book: book/src/25-dispersion/03-mbd.md. Every check prints "FAIL [<name>]"
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

Mutations (each turns the named check red):
  - the -3/2 sum w reference energy as -1 sum w                  [qho-vs-frequency]
  - T_GG's 2 z^2 th R R^T term without its factor 2              [t-gg-is-hessian]
  - the width prefactor sqrt(2/pi)/3 as 1/3                      [sigma-self-energy]
  - the contraction over the diagonal block B = A only           [scs-dimer-closed-form]
  - the adjoint 1/4 lambda^-1/2 as 1/2 lambda^-1/2               [energy-adjoint]

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


symbolic()
numeric()
london_second_order()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall MBD identities hold")
