#!/usr/bin/env python3
"""Sympy/numpy checks of the identities ferric-pcm's IEF-PCM code relies on.

Theory and notation: crates/ferric-pcm/theory.md. Each claim names the Rust
function it checks, asserts, and carries a negative control (a deliberately
wrong variant that must NOT satisfy the identity). Exits nonzero on the first
mismatch.

  1  Sphere kernels (matrices.rs build_s_d_kind diagonal, :137 and :169): on a
     sphere of radius R the double-layer kernel with the source-point normal,
     (r - r').n' / |r - r'|^3, equals -1/(2R) times the single-layer kernel
     1/|r - r'|. The code's D_ii = -S_ii/(2R) is this identity applied to
     the self-term.
  2  Funk-Hecke spectra: on the sphere S Y_lm = 4 pi R/(2l+1) Y_lm and
     D Y_lm = -2 pi/(2l+1) Y_lm (D acting on a surface density with dS').
  3  IEF on a sphere (matrices.rs build_k_r): with f = (eps-1)/(eps+1),
     K = S - f/(2 pi) D A S and R = -f (I - D A/(2 pi)), channel l gives
     q_l = -k_l v_l/S_l with k_l = (eps-1)(l+1)/((l+1) eps + l), Kirkwood's
     dielectric-sphere reaction field for every l. l = 0 is Gauss's law
     (total induced charge -(eps-1)/eps Q) and the Born energy
     -(eps-1)/(2 eps R) Q^2. Controls: f = (eps-1)/eps, and the COSMO factors
     (eps-1)/eps (C-PCM) and (eps-1)/(eps+1/2) (ferric_scf::cosmo), are not
     Kirkwood for every l.
  4  Conductor limit (build_k_r at f = 1): K^-1 R = -S^-1 exactly for any
     S, D, A (K = (I - DA/2pi) S, R = -(I - DA/2pi)): IEF reduces to the
     conductor-like equation S q = -v of C-PCM/COSMO at eps -> inf. Control:
     f != 1 does not.
  5  Symmetrized solve (solver.rs solve_pcm_charges): q = 1/2 [K^-1 R v +
     R^T (K^T)^-1 v] = 1/2 (Q + Q^T) v has the same energy 1/2 q.v as the
     unsymmetrized Q v; it equals Q v itself exactly when D A S is symmetric
     (the discrete Calderon identity S A D^T = D A S of the exact continuum).
     Controls: the reversed order (K^T)^-1 R^T v changes the energy; a generic
     D gives q_sym != Q v.
  6  Fock contribution (potential.rs build_reaction_field_operator, lib.rs
     pcm_step): with v_k(D) = v_nuc,k + sum_mn D_mn B^k_mn,
     B^k = <m| -1/|r - r_k| |n>, the gradient of E = 1/2 v^T Q_sym v is
     dE/dD_mn = sum_k q_k B^k_mn = V_pcm. Control: q_raw = Q v is not the
     gradient when Q is not symmetric.
  7  Discrete sphere (numpy; replicates build_cavity on one atom,
     build_s_d_kind GaussianSmeared, build_k_r, solve_pcm_charges): a unit
     charge at d = 0, R/4, R/2 inside a Lebedev sphere converges to the
     Kirkwood energy and the Gauss-law charge as the order goes 50 -> 110 ->
     302, with |E_302 - E_exact| <= |E_302 - E_110|. The Rust twin is
     crates/ferric-pcm/tests/ief_limits.rs.

Run: python3 scripts/pcm_ief_sympy.py  (needs sympy, numpy, scipy >= 1.15)
"""

import sys

import numpy as np
import sympy as sp

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def zero(expr):
    return sp.simplify(expr) == 0


# ---------------------------------------------------------------- claim 1
R, c = sp.symbols("R c", positive=True)
th, ph, th2, ph2 = sp.symbols("theta phi theta2 phi2", real=True)


def unit(t, p):
    return sp.Matrix([sp.sin(t) * sp.cos(p), sp.sin(t) * sp.sin(p), sp.cos(t)])


n1, n2 = unit(th, ph), unit(th2, ph2)
r1, r2 = R * n1, R * n2
dvec = r1 - r2
dist2 = sp.simplify(dvec.dot(dvec))
# Express through cos(gamma) = n1.n2 =: c; |r1 - r2|^2 = 2 R^2 (1 - c).
cosg = sp.simplify(n1.dot(n2))
check("1 |r-r'|^2 = 2R^2(1 - n.n') on the sphere", zero(dist2 - 2 * R**2 * (1 - cosg)))
d_num = sp.simplify(dvec.dot(n2))  # (r - r').n' , source-point normal
check("1 (r-r').n' = -R(1 - n.n')", zero(d_num + R * (1 - cosg)))
s_kern = 1 / sp.sqrt(2 * R**2 * (1 - c))
d_kern = -R * (1 - c) / (2 * R**2 * (1 - c)) ** sp.Rational(3, 2)
check("1 D kernel = -S kernel/(2R)", zero(sp.simplify(d_kern + s_kern / (2 * R))))
# Control: the field-point normal n gives (r - r').n = +R(1 - n.n'), the
# opposite sign; D^T (field normal) would be +S/(2R).
check("1 control: field-point normal flips the sign", zero(sp.simplify(dvec.dot(n1) - R * (1 - cosg))))

# ---------------------------------------------------------------- claim 2
# Funk-Hecke: int K(n.n') Y_l(n') dOmega' = 2 pi int_{-1}^{1} K(c) P_l(c) dc Y_l(n).
for ell in range(0, 7):
    P = sp.legendre(ell, c)
    s_l = sp.simplify(2 * sp.pi * R**2 * sp.integrate(P / sp.sqrt(2 * R**2 * (1 - c)), (c, -1, 1)))
    d_l = sp.simplify(-s_l / (2 * R))
    ok = zero(s_l - 4 * sp.pi * R / (2 * ell + 1)) and zero(d_l + 2 * sp.pi / (2 * ell + 1))
    check(f"2 S_l = 4piR/(2l+1), D_l = -2pi/(2l+1) at l={ell}", ok, f"S_l={s_l}")

# ---------------------------------------------------------------- claim 3
eps, l = sp.symbols("epsilon l", positive=True)
f = (eps - 1) / (eps + 1)
S_l = 4 * sp.pi * R / (2 * l + 1)
D_l = -2 * sp.pi / (2 * l + 1)  # D A acting on a density: the D_l of claim 2
K_l = S_l - f / (2 * sp.pi) * D_l * S_l
R_l = -f * (1 - D_l / (2 * sp.pi))
k_l = sp.simplify(-R_l / K_l * S_l)  # q_l = -k_l v_l / S_l
kirkwood = (eps - 1) * (l + 1) / ((l + 1) * eps + l)
check("3 IEF channel factor = Kirkwood (l+1)(eps-1)/((l+1)eps+l), all l", zero(k_l - kirkwood))
check("3 l=0: Gauss law total charge -(eps-1)/eps", zero(k_l.subs(l, 0) - (eps - 1) / eps))
# Born: v = Q/R uniform -> q_0 = -k_0 v/S_0 per unit-sphere density; total
# charge 4 pi R^2 sigma = -k_0 Q; potential at the centre = -k_0 Q/R.
Q = sp.symbols("Q", real=True)
E_born = sp.Rational(1, 2) * Q * (-k_l.subs(l, 0) * Q / R)
check("3 Born energy -(eps-1)/(2 eps R) Q^2", zero(E_born + (eps - 1) / (2 * eps * R) * Q**2))
f_wrong = (eps - 1) / eps
k_wrong = sp.simplify(
    -(-f_wrong * (1 - D_l / (2 * sp.pi))) / (S_l - f_wrong / (2 * sp.pi) * D_l * S_l) * S_l
)
check("3 control: f=(eps-1)/eps is not Kirkwood", not zero(k_wrong - kirkwood))
for name, fc in (("C-PCM (eps-1)/eps", (eps - 1) / eps), ("COSMO (eps-1)/(eps+1/2)", (eps - 1) / (eps + sp.Rational(1, 2)))):
    at0 = zero(fc - kirkwood.subs(l, 0))
    at1 = zero(fc - kirkwood.subs(l, 1))
    check(f"3 control: {name} is not Kirkwood at every l", not (at0 and at1), f"l=0 match={at0}, l=1 match={at1}")


# ---------------------------------------------------------------- claims 4-6
def rand_rat(rng, shape, lo=-9, hi=9, den=7):
    return sp.Matrix(*shape, lambda i, j: sp.Rational(int(rng.integers(lo, hi + 1)), den))


rng = np.random.default_rng(20261105)
n = 4
M = rand_rat(rng, (n, n))
S = M * M.T + 3 * sp.eye(n)  # symmetric positive definite
Dm = rand_rat(rng, (n, n))  # not symmetric
A = sp.diag(*[sp.Rational(int(x), 5) for x in rng.integers(1, 9, n)])
v = rand_rat(rng, (n, 1))
two_pi = 2 * sp.pi
I = sp.eye(n)


def k_r(fv, Dmat):
    DA = Dmat * A
    return S - fv / two_pi * DA * S, -fv * (I - DA / two_pi)


K1, R1 = k_r(1, Dm)
check("4 f=1: K^-1 R = -S^-1 exactly", zero(sp.simplify(K1.inv() * R1 + S.inv()).norm()))
fr = sp.Rational(77, 79)
Kf, Rf = k_r(fr, Dm)
check("4 control: f<1 is not the conductor equation", not zero(sp.simplify(Kf.inv() * Rf + S.inv()).norm()))

Qm = Kf.inv() * Rf
q_raw = Qm * v
q_sym = (q_raw + Rf.T * (Kf.T).inv() * v) / 2
q_bad = (q_raw + (Kf.T).inv() * Rf.T * v) / 2
e_raw = (q_raw.T * v)[0] / 2
check("5 E(q_sym) = E(Q v)", zero((q_sym.T * v)[0] / 2 - e_raw))
check("5 control: reversed order (K^T)^-1 R^T v changes E", not zero((q_bad.T * v)[0] / 2 - e_raw))
check("5 control: generic D gives q_sym != Q v", not zero(sp.simplify(q_sym - q_raw).norm()))
# Calderon-consistent D: choose D A S = N symmetric, i.e. D = N S^-1 A^-1.
Nm = rand_rat(rng, (n, n))
Nm = Nm + Nm.T
Dc = Nm * S.inv() * A.inv()
Kc, Rc = k_r(fr, Dc)
qc_raw = Kc.inv() * Rc * v
qc_sym = (qc_raw + Rc.T * (Kc.T).inv() * v) / 2
check("5 D A S symmetric => q_sym = Q v exactly", zero(sp.simplify(qc_sym - qc_raw).norm()))

# Claim 6: two basis functions, symbolic symmetric density.
nb = 2
d00, d01, d11 = sp.symbols("d00 d01 d11", real=True)
Dden = sp.Matrix([[d00, d01], [d01, d11]])
Bk = [rand_rat(rng, (nb, nb)) for _ in range(n)]
Bk = [b + b.T for b in Bk]  # one-electron integrals are symmetric in (m, n)
vnuc = rand_rat(rng, (n, 1))
vD = sp.Matrix([vnuc[k] + sum(Dden[a, b] * Bk[k][a, b] for a in range(nb) for b in range(nb)) for k in range(n)])
Qsym = (Qm + Qm.T) / 2
E = sp.expand((vD.T * Qsym * vD)[0] / 2)
qv = Qsym * vD
qv_raw = Qm * vD
# dE/dD_mn for the independent symmetric entries: d00, d11 directly; d01
# appears at (0,1) and (1,0), so dE/dd01 = 2 V_01.
for sym, (a, b), mult in ((d00, (0, 0), 1), (d11, (1, 1), 1), (d01, (0, 1), 2)):
    V_ab = sum(qv[k] * Bk[k][a, b] for k in range(n))
    V_raw = sum(qv_raw[k] * Bk[k][a, b] for k in range(n))
    check(f"6 dE/dD[{a}{b}] = sum_k q_k B^k (V_pcm)", zero(sp.diff(E, sym) - mult * V_ab))
    if (a, b) == (0, 0):
        check("6 control: q_raw is not the gradient", not zero(sp.diff(E, sym) - mult * V_raw))

# ---------------------------------------------------------------- claim 7
try:
    from scipy.integrate import lebedev_rule
    from scipy.special import erf
except ImportError as exc:  # pragma: no cover
    check("7 scipy.integrate.lebedev_rule available (scipy >= 1.15)", False, str(exc))
    lebedev_rule = None

XI = {50: 4.89250673295, 110: 4.90101060987, 302: 4.90498088169}  # matrices.rs gaussian_xi_table
DEG = {50: 11, 110: 17, 302: 29}


def sphere(npts, rad):
    x, w = lebedev_rule(DEG[npts])  # weights sum to 4 pi (PySCF convention)
    nrm = x.T
    assert len(w) == npts
    # cavity.rs: area = 4 pi R^2 w_ferric = R^2 w_4pi; charge_exp = XI/(R sqrt(w_4pi))
    return rad * nrm, nrm, w * rad * rad, XI[npts] / (rad * np.sqrt(w))


def s_d_gaussian(pos, nrm, area, xi, rad):
    rij = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(rij, axis=2)
    np.fill_diagonal(r, 1.0)
    nr = np.einsum("ijk,jk->ij", rij, nrm)  # (r_i - r_j).n_j, source normal
    xij = xi[:, None] * xi[None, :] / np.sqrt(xi[:, None] ** 2 + xi[None, :] ** 2)
    xr = xij * r
    S = erf(xr) / r
    D = S * nr / r**2 - 2 / np.sqrt(np.pi) * xr * np.exp(-(xr**2)) * nr / r**3
    np.fill_diagonal(S, xi * np.sqrt(2 / np.pi))  # switch_fun = 1 on a lone sphere
    np.fill_diagonal(D, -xi * np.sqrt(2 / np.pi) / (2 * rad))
    return S, D


def ief_charges(S, D, area, ev):
    fv = (ev - 1) / (ev + 1)
    DA = D * area[None, :]
    K = S - fv / (2 * np.pi) * DA @ S
    Rm = -fv * (np.eye(len(area)) - DA / (2 * np.pi))
    return Rm, K


def kirkwood_energy(ev, rad, d):
    ls = np.arange(400)
    return -0.5 / rad * np.sum((ls + 1) * (ev - 1) / ((ls + 1) * ev + ls) * (d / rad) ** (2 * ls))


if lebedev_rule is not None:
    rad = 4.0
    u = np.array([0.3, -0.5, 0.81])
    u /= np.linalg.norm(u)
    for ev in (78.4, 4.0):
        for frac in (0.0, 0.25, 0.5):
            centre = frac * rad * u
            e_ex = kirkwood_energy(ev, rad, frac * rad)
            res = []
            for npts in (50, 110, 302):
                pos, nrm, area, xi = sphere(npts, rad)
                vv = 1 / np.linalg.norm(pos - centre, axis=1)
                S7, D7 = s_d_gaussian(pos, nrm, area, xi, rad)
                Rm, K = ief_charges(S7, D7, area, ev)
                q = 0.5 * (np.linalg.solve(K, Rm @ vv) + Rm.T @ np.linalg.solve(K.T, vv))
                res.append((0.5 * q @ vv, q.sum()))
            err = [abs(e - e_ex) for e, _ in res]
            est_e = abs(res[2][0] - res[1][0])
            est_q = abs(res[2][1] - res[1][1])
            qg = -(ev - 1) / ev
            ok = err[2] < err[1] < err[0] and err[2] <= est_e and abs(res[2][1] - qg) <= est_q
            check(
                f"7 sphere eps={ev} d/R={frac}: Kirkwood + Gauss law within discretization estimate",
                ok,
                f"rel err 302 {err[2] / abs(e_ex):.2e} (est {est_e / abs(e_ex):.2e}), "
                f"Q302 {res[2][1]:.9f} vs {qg:.9f}",
            )
    # Control: the IEF factor replaced by C-PCM's (eps-1)/eps misses Born by
    # far more than the discretization estimate.
    pos, nrm, area, xi = sphere(302, rad)
    vv = 1 / np.linalg.norm(pos, axis=1)
    S7, D7 = s_d_gaussian(pos, nrm, area, xi, rad)
    fw = (78.4 - 1) / 78.4
    DA = D7 * area[None, :]
    K = S7 - fw / (2 * np.pi) * DA @ S7
    Rm = -fw * (np.eye(len(area)) - DA / (2 * np.pi))
    q = np.linalg.solve(K, Rm @ vv)
    rel = abs(0.5 * q @ vv - kirkwood_energy(78.4, rad, 0.0)) / abs(kirkwood_energy(78.4, rad, 0.0))
    check("7 control: f=(eps-1)/eps misses Born by > 1e-4", rel > 1e-4, f"rel {rel:.2e}")

if FAILED:
    print(f"\n{len(FAILED)} claim(s) FAILED: {FAILED}")
    sys.exit(1)
print("\nall PCM claims pass")
