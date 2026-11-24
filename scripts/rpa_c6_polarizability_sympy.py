#!/usr/bin/env python3
"""Sympy/numpy checks of the RPA polarizability and Casimir-Polder C6 code.

Theory and notation: `book/src/21-polarizabilities/01-polarizabilities-from-dielectric-modes.md#polarizabilities-from-dielectric-modes-dielectric-form` and its "Per-atom polarizability" part. Every check prints
"FAIL [<name>]" when it fails, and the script exits nonzero if any did.

  1  Single pole (sympy, dispersion/mbd.rs ts_atom_params): with
     a(iw) = a0/(1 + (w/w0)^2), (3/pi) int_0^inf a_A a_B dw
     = (3/2) a_A a_B w_A w_B/(w_A + w_B) (London); at A = B it is
     (3/4) a^2 w, so w_A = (4/3) C6_A/a_A^2.
  2  Quadrature (dispersion.rs casimir_polder_c6 on quadrature.rs
     gauss_legendre_nodes): the coded sum (3/pi) sum_k w_k a_A a_B over the
     default 20-point u0 = 0.5 grid reproduces the London value of check 1
     (measured 3e-14 relative, bar 1e-8), and converges below 1e-10 at 200
     points. The anisotropic C6 is element by element; the isotropic one uses
     (1/3) Tr alpha.
  3  Response kernel (properties/per_atom.rs closed_per_atom_alpha): the
     coded alpha_dj = 4 m_d.(g o mu_j) - 16 (B (g o m_d))^T eps^-1 (B (g o mu_j)),
     eps = 1 + 4 B diag(g) B^T, g = e/(w^2 + e^2) (or 1/e static), equals the
     independent sum over the closed-shell dRPA (Casida) excitations,
     alpha(iw) = sum_n f_n/(Omega_n^2 + w^2) with Omega^2 = eig of
     Delta^1/2 (Delta + 4K) Delta^1/2 and f_n = 4 (m^T Delta^1/2 Z_n)(Z_n^T Delta^1/2 mu).
     The static form 1/e equals the w -> 0 limit.
  4  C6 from the kernel on the grid equals the analytic pole double sum
     (3/2) sum_mn f^A_m f^B_n/(Omega_m Omega_n (Omega_m + Omega_n)) (200 points).
  5  Charge-transfer sum rule (per_atom.rs charge_transfer_remainder): for
     partition weights with sum_A w_A = 1 (so sum_A <i|w_A|a> = 0) and
     m^A = <i|(r - R_A) w_A|a>, alpha_mol - sum_A sym(alpha^A) =
     sym <sum_A R_A w_A|R|mu>, where sym is the code's symmetrize_tensor.

Mutations (each turns the named check red):
  - PREF, the Casimir-Polder prefactor, 3/pi -> 3/(2 pi)      [c6-quadrature-converged]
  - kernel screening term 16 -> 4                             [kernel-vs-casida]
  - TS pole omega_A without its factor 4/3                    [ts-pole-reproduces-c6]
  - alpha_CT with the first atom left out of sum_A           [ct-sum-rule]

Run: python3 scripts/rpa_c6_polarizability_sympy.py  (needs sympy, numpy)
"""

import sys

import numpy as np
import sympy as sp

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


PREF = 3.0 / np.pi  # dispersion.rs casimir_polder_c6


def symbolic():
    w, aa, ab, wa, wb = sp.symbols("omega a_A a_B omega_A omega_B", positive=True)
    integ = sp.integrate(
        aa * ab / ((1 + (w / wa) ** 2) * (1 + (w / wb) ** 2)), (w, 0, sp.oo)
    )
    c6 = sp.simplify(3 / sp.pi * integ)
    check(
        "london-integral",
        sp.simplify(c6 - sp.Rational(3, 2) * aa * ab * wa * wb / (wa + wb)) == 0,
        f"{c6}",
    )
    self_c6 = sp.simplify(c6.subs({ab: aa, wb: wa}))
    check(
        "ts-pole-identity",
        sp.simplify(self_c6 - sp.Rational(3, 4) * aa**2 * wa) == 0,
        f"C6_AA = {self_c6}",
    )


def gl_grid(n, u0=0.5):
    """quadrature.rs gauss_legendre_nodes."""
    x, wg = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + x) / (1 - x), wg * 2 * u0 / (1 - x) ** 2


def casimir_polder_c6(per_atom, weights):
    """dispersion.rs casimir_polder_c6: per_atom[A][k] is a 3x3 tensor."""
    iso = [[np.trace(t) / 3.0 for t in row] for row in per_atom]
    n = len(per_atom)
    c_iso = np.zeros((n, n))
    c_aniso = np.zeros((n, n, 3, 3))
    for a in range(n):
        for b in range(n):
            c_iso[a, b] = PREF * sum(
                wk * iso[a][k] * iso[b][k] for k, wk in enumerate(weights)
            )
            for k, wk in enumerate(weights):
                c_aniso[a, b] += PREF * wk * per_atom[a][k] * per_atom[b][k]
    return c_iso, c_aniso


def ts_atom_params(alpha_free, c6_free, ratio):
    """dispersion/mbd.rs ts_atom_params."""
    alpha_eff = max(ratio * alpha_free, 1e-8)
    c6_eff = ratio * ratio * c6_free
    omega_a = (4.0 / 3.0) * c6_eff / (alpha_eff * alpha_eff)
    return alpha_eff, omega_a, c6_eff


def quadrature_checks():
    # Free-atom references of the order of C and H (a.u.), volume ratios < 1.
    atoms = [ts_atom_params(12.0, 46.6, 0.85), ts_atom_params(4.5, 6.5, 0.7)]
    shape = [np.diag([1.2, 0.9, 0.9]), np.eye(3)]

    def per_atom_on(freqs):
        return [
            [a / (1 + (f / w0) ** 2) * sh for f in freqs]
            for (a, w0, _), sh in zip(atoms, shape)
        ]

    def london(i, j):
        (a1, w1, _), (a2, w2, _) = atoms[i], atoms[j]
        return 1.5 * a1 * a2 * w1 * w2 / (w1 + w2)

    f200, w200 = gl_grid(200)
    c_iso, c_aniso = casimir_polder_c6(per_atom_on(f200), w200)
    err = max(
        abs(c_iso[i, j] - london(i, j)) / london(i, j)
        for i in range(2)
        for j in range(2)
    )
    check("c6-quadrature-converged", err < 1e-10, f"200-point rel err {err:.1e}")
    check(
        "ts-pole-reproduces-c6",
        abs(c_iso[0, 0] - atoms[0][2]) < 1e-9 * atoms[0][2],
        f"C6_AA {c_iso[0, 0]:.10f} vs r^2 C6_free {atoms[0][2]:.10f}",
    )
    f20, w20 = gl_grid(20)
    c20, _ = casimir_polder_c6(per_atom_on(f20), w20)
    err20 = max(
        abs(c20[i, j] - london(i, j)) / london(i, j) for i in range(2) for j in range(2)
    )
    check("c6-default-grid", err20 < 1e-8, f"20-point u0 = 0.5 rel err {err20:.1e}")
    elem = c_aniso[0, 1][0, 0]
    expect = 1.2 * 1.0 * london(0, 1)
    check(
        "c6-aniso-elementwise",
        abs(elem - expect) < 1e-9 * expect,
        "C6_xx = shape_xx^A shape_xx^B C6_iso",
    )


# --------------------------------------------------------------- response
rng = np.random.default_rng(5)
no, nv, naux, natom = 2, 4, 6, 3
eo = np.array([-1.1, -0.7])
ev = np.array([0.3, 0.6, 1.1, 1.8])
e_ia = (ev[None, :] - eo[:, None]).ravel()
nov = no * nv
b_ov = 0.3 * rng.standard_normal((naux, nov))
# Partition: sum_A (w_A)_ia = 0, sum_A (r w_A)_ia = mu_ia (dipole direction d).
mu = 0.5 * rng.standard_normal((3, nov))
w_part = rng.standard_normal((natom, nov))
w_part[-1] = -w_part[:-1].sum(axis=0)
rw = rng.standard_normal((natom, 3, nov))
rw[-1] = mu - rw[:-1].sum(axis=0)
r_atoms = rng.standard_normal((natom, 3))
bra = np.array(
    [[rw[a, d] - r_atoms[a, d] * w_part[a] for d in range(3)] for a in range(natom)]
)


def closed_per_atom_alpha(bra_list, mu_vec, omega):
    """per_atom.rs closed_per_atom_alpha (omega None = Frequency::Static)."""
    g = 1.0 / e_ia if omega is None else e_ia / (omega * omega + e_ia * e_ia)
    b_scaled = b_ov * np.sqrt(4.0 * g)
    eps_mat = b_scaled @ b_scaled.T + np.eye(naux)
    mu_g = [mu_vec[d] * g for d in range(3)]
    y_mol = [np.linalg.solve(eps_mat, b_ov @ mu_g[d]) for d in range(3)]
    out = []
    for atom in bra_list:
        tensor = np.zeros((3, 3))
        for d in range(3):
            w_ai = b_ov @ (atom[d] * g)
            for j in range(3):
                bare = atom[d] @ mu_g[j]
                coupled = w_ai @ y_mol[j]
                tensor[d][j] = 4.0 * bare - 16.0 * coupled
        out.append(0.5 * (tensor + tensor.T))  # symmetrize_tensor
    return out


def casida_alpha(bra_vec, ket_vec, omega):
    """Independent route: sum over dRPA excitations."""
    kmat = b_ov.T @ b_ov
    sq = np.sqrt(e_ia)
    m = sq[:, None] * (np.diag(e_ia) + 4.0 * kmat) * sq[None, :]
    om2, z = np.linalg.eigh(m)
    bl = (bra_vec * sq) @ z
    kt = (ket_vec * sq) @ z
    f = 4.0 * bl * kt
    return np.sum(f / (om2 + omega**2)), f, np.sqrt(om2)


def response_checks():
    worst = 0.0
    for omega in [0.0, 0.4, 1.7]:
        ker = closed_per_atom_alpha(bra, mu, omega)
        for a in range(natom):
            for d in range(3):
                for j in range(3):
                    ref = 0.5 * (
                        casida_alpha(bra[a][d], mu[j], omega)[0]
                        + casida_alpha(bra[a][j], mu[d], omega)[0]
                    )
                    worst = max(worst, abs(ker[a][d, j] - ref))
    check("kernel-vs-casida", worst < 1e-12, f"max |d| = {worst:.1e}")
    st = closed_per_atom_alpha(bra, mu, None)
    lim = closed_per_atom_alpha(bra, mu, 1e-9)
    d_st = max(np.max(np.abs(s - t)) for s, t in zip(st, lim))
    check("static-limit", d_st < 1e-12, f"|alpha(Static) - alpha(1e-9)| = {d_st:.1e}")

    # 4. C6 between atom 0 and atom 1 (xx element) on the grid vs pole sum.
    f200, w200 = gl_grid(200)
    per = [[closed_per_atom_alpha(bra, mu, f)[a] for f in f200] for a in range(2)]
    _, c_an = casimir_polder_c6(per, w200)
    _, fa, om = casida_alpha(bra[0][0], mu[0], 0.0)
    _, fb, _ = casida_alpha(bra[1][0], mu[0], 0.0)
    pole = 1.5 * np.sum(
        np.outer(fa, fb) / (np.outer(om, om) * (om[:, None] + om[None, :]))
    )
    check(
        "c6-pole-sum",
        abs(c_an[0, 1][0, 0] - pole) < 1e-9 * abs(pole),
        f"{c_an[0, 1][0, 0]:.12f} vs {pole:.12f}",
    )

    # 5. Charge-transfer sum rule.
    for omega in [None, 0.8]:
        per_atom = closed_per_atom_alpha(bra, mu, omega)
        mol = closed_per_atom_alpha([mu], mu, omega)[0]
        ct = mol - sum(per_atom)  # charge_transfer_remainder
        rw_sum = np.array(
            [sum(r_atoms[a, d] * w_part[a] for a in range(natom)) for d in range(3)]
        )
        ct_ref = closed_per_atom_alpha([rw_sum], mu, omega)[0]
        d_ct = np.max(np.abs(ct - ct_ref))
        check(
            "ct-sum-rule",
            d_ct < 1e-12,
            f"omega={omega}: |alpha_CT - sym<sum R_A w_A|R|mu>| = {d_ct:.1e}, |alpha_CT| = {np.max(np.abs(ct)):.2f}",
        )


symbolic()
quadrature_checks()
response_checks()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall polarizability / C6 identities hold")
