#!/usr/bin/env python3
"""Harmonic IR intensities as ferric computes them.

Derivation: crates/ferric-scf/theory.md, "Infrared intensities".
Code: crates/ferric-scf/src/infrared.rs (AU_TO_KM_PER_MOL, ir_intensities,
the field route dipole_derivatives_field) and the normal-mode back-transform
x = M^(-1/2) u of crates/ferric-scf/src/frequencies.rs, transcribed below.

Checks:
  C1  AU_TO_KM_PER_MOL = N_A e^2 / (12 eps0 c^2 m_e) / 1000 from CODATA 2018
      (relative 1e-12); per e^2/u it is 974.880 km/mol and per (D/A)^2/u
      42.2561 km/mol, the values quoted by Gaussian and PySCF.
  S1  (sympy) the 0 -> 1 band of a harmonic mode: from the integrated cross
      section int sigma d(omega) = pi omega |<0|mu|1>|^2 / (3 eps0 hbar c) and
      |<0|Q|1>|^2 = hbar / (2 omega), the molar intensity integrated over
      wavenumber is N_A |dmu/dQ|^2 / (12 eps0 c^2): omega and hbar cancel, so
      the result is the classical one.
  S2  (sympy) at temperature T the net absorption of the whole ladder,
      sum_n (p_n - p_{n+1}) (n + 1) with Boltzmann p_n, is 1: the harmonic
      intensity does not depend on T (high-T limit included).
  S3  (sympy) the field route: for E(R, F) = E0(R) - mu(R) . F - F.alpha(R).F/2,
      -d/dF_a dE/dR_i = dmu_a/dR_i at F = 0, and the central difference
      -(g(+h) - g(-h)) / 2h is exact for this quadratic-in-F model.
  N1  Bent triatomic model (pair springs, bond-charge dipole with a nonlinear
      bond dependence): intensities by the transcribed contraction
      P . x_k equal those from differencing mu along each mode,
      mu(R0 + x_k s) (relative 1e-6, the O(s^2) stencil error).
  N2  Diatomic point charges +-q: A = C q^2 / mu_red (mu_red in m_e)
      (relative 1e-12).
  N3  The transcribed field route on the model of S3 with a cubic field term
      added recovers dmu/dR to its O(h^2) truncation, beta |x| h^2 / 3.

Mutations (each turns the script red; exact one-occurrence substitutions):
  mass weighting dropped: x = u instead of u / sqrt(m)           (N1, N2 fail)
  constant built with m_u instead of m_e                         (C1 fails)
  sign of the field route flipped                                (N3 fails)
  isotropic sum over a: only the z component kept                (N1 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# CODATA 2018 (independent of the code's constant).
N_A = 6.02214076e23
E_C = 1.602176634e-19
EPS0 = 8.8541878128e-12
C_L = 299792458.0
M_E = 9.1093837015e-31
M_U = 1.66053906660e-27
ANG = 1e-10
DEBYE = 1e-21 / C_L

# Transcribed from infrared.rs / frequencies.rs.
AU_TO_KM_PER_MOL = 1_777_097.731_306
AMU_TO_ELECTRON_MASS = 1.0 / 5.485_799_090_65e-4


def normal_modes(hess, masses_amu):
    """Vibrational modes x_k = M^(-1/2) u_k (rows), translations/rotations
    dropped by eigenvalue (the model Hessian is exactly invariant)."""
    mass_au = np.repeat(np.asarray(masses_amu) * AMU_TO_ELECTRON_MASS, 3)
    inv = 1.0 / np.sqrt(mass_au)
    w, u = np.linalg.eigh(hess * np.outer(inv, inv))
    keep = np.abs(w) > 1e-10 * np.abs(w).max()
    return w[keep], (u[:, keep] * inv[:, None]).T


def ir_intensities(apt, modes):
    dmu = apt @ modes.T
    return AU_TO_KM_PER_MOL * np.sum(dmu**2, axis=0)


def field_route(grad_at_field, h):
    """P[a, i] = -(g_i(+h e_a) - g_i(-h e_a)) / (2h)."""
    rows = []
    for a in range(3):
        ep = np.zeros(3)
        ep[a] = h
        gp, gm = grad_at_field(ep), grad_at_field(-ep)
        rows.append(-(gp - gm) / (2.0 * h))
    return np.array(rows)


# ---------------------------------------------------------------------------
# C1 constant
c_si = N_A * E_C**2 / (12 * EPS0 * C_L**2 * M_E) / 1000.0
check("C1 AU_TO_KM_PER_MOL from CODATA 2018", abs(AU_TO_KM_PER_MOL / c_si - 1) < 1e-12, f"{c_si!r}")
per_amu = AU_TO_KM_PER_MOL / AMU_TO_ELECTRON_MASS
check("C1 974.880 km/mol per e^2/u", abs(per_amu - 974.880) < 1e-3, f"{per_amu:.4f}")
d_per_ang = (DEBYE / ANG) ** 2 / M_U * N_A / (12 * EPS0 * C_L**2) / 1000.0
check("C1 42.2561 km/mol per (D/A)^2/u", abs(d_per_ang - 42.2561) < 1e-3, f"{d_per_ang:.4f}")

# ---------------------------------------------------------------------------
# S1 harmonic 0 -> 1 band
w, hbar, eps0, c, NA, dmu = sp.symbols("omega hbar epsilon0 c N_A mu_Q", positive=True)
d01_sq = dmu**2 * hbar / (2 * w)  # |<0|mu|1>|^2 along the mode
sigma_omega = sp.pi * w * d01_sq / (3 * eps0 * hbar * c)  # int sigma d omega
# d(omega) = 2 pi c d(nu~): int sigma d(nu~) = int sigma d(omega) / (2 pi c)
a_molar = NA * sigma_omega / (2 * sp.pi * c)
check("S1 A = N_A |dmu/dQ|^2 / (12 eps0 c^2)", sp.simplify(a_molar - NA * dmu**2 / (12 * eps0 * c**2)) == 0)
check("S1 hbar and omega cancel (classical)", not ({hbar, w} & a_molar.free_symbols))

# S2 temperature independence of the net ladder absorption
# y = exp(-h nu / k T) in (0, 1); p_n = (1 - y) y^n, and the n -> n+1 band
# strength is (n + 1) times the 0 -> 1 strength.
y = sp.symbols("y", positive=True)
n = sp.symbols("n", integer=True, nonnegative=True)
p_n = (1 - y) * y**n
net = sp.summation(sp.expand((p_n - p_n.subs(n, n + 1)) * (n + 1)), (n, 0, sp.oo))
net = sp.piecewise_fold(net)
if isinstance(net, sp.Piecewise):  # the |y| < 1 branch
    net = next(e for e, cond in net.args if cond.subs(y, sp.Rational(1, 2)) == True)  # noqa: E712
check("S2 sum_n (p_n - p_n+1)(n+1) = 1", sp.simplify(net) == 1, f"{sp.simplify(net)}")

# S3 field route on a quadratic-in-F model
R, F, hh = sp.symbols("R F h", real=True)
E0 = sp.Function("E0")(R)
mu = sp.Function("mu")(R)
al = sp.Function("alpha")(R)
E = E0 - mu * F - al * F**2 / 2
g = sp.diff(E, R)
apt_exact = -sp.diff(g, F).subs(F, 0)
check("S3 -d2E/dFdR = dmu/dR", sp.simplify(apt_exact - sp.diff(mu, R)) == 0)
cd = -(g.subs(F, hh) - g.subs(F, -hh)) / (2 * hh)
check("S3 central difference exact for quadratic E(F)", sp.simplify(cd - sp.diff(mu, R)) == 0)

# ---------------------------------------------------------------------------
# N1 bent triatomic model
masses = [15.999, 1.008, 1.008]
r0 = np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])  # Bohr
pairs = [(0, 1, 0.5), (0, 2, 0.5), (1, 2, 0.08)]


def spring_hessian(x):
    n3 = 3 * len(x)
    hmat = np.zeros((n3, n3))
    for a, b, k in pairs:
        d = x[a] - x[b]
        e = d / np.linalg.norm(d)
        blk = k * np.outer(e, e)
        for i, j, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            hmat[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += s * blk
    return hmat


def dipole(x):
    """Bond-charge model: each O-H bond carries q(r) = 0.3 + 0.2 (r - 1.8)^2 + 0.1 (r-1.8)."""
    mu_v = np.zeros(3)
    for hidx in (1, 2):
        d = x[hidx] - x[0]
        rr = np.linalg.norm(d)
        q = 0.3 + 0.1 * (rr - 1.8) + 0.2 * (rr - 1.8) ** 2
        mu_v += q * d
    return mu_v


def apt_fd(x, s=1e-5):
    p = np.zeros((3, x.size))
    for i in range(x.size):
        xp, xm = x.copy().ravel(), x.copy().ravel()
        xp[i] += s
        xm[i] -= s
        p[:, i] = (dipole(xp.reshape(-1, 3)) - dipole(xm.reshape(-1, 3))) / (2 * s)
    return p


_, modes = normal_modes(spring_hessian(r0), masses)
a_contract = ir_intensities(apt_fd(r0), modes)
a_mode = []
for xk in modes:
    s = 1e-3
    dm = (dipole(r0 + s * xk.reshape(-1, 3)) - dipole(r0 - s * xk.reshape(-1, 3))) / (2 * s)
    a_mode.append(AU_TO_KM_PER_MOL * np.dot(dm, dm))
a_mode = np.array(a_mode)
check(
    "N1 contraction P.x_k = difference along the mode (claim N1)",
    len(modes) == 3 and np.allclose(a_contract, a_mode, rtol=1e-6, atol=0),
    f"{a_contract} vs {a_mode}",
)

# N2 point-charge diatomic
m2 = [1.008, 18.998]
x2 = np.array([[0, 0, 0], [0, 0, 1.74]], float)
k2 = 0.6
h2 = np.zeros((6, 6))
h2[2, 2] = h2[5, 5] = k2
h2[2, 5] = h2[5, 2] = -k2
_, modes2 = normal_modes(h2, m2)
q = 0.4
apt2 = np.zeros((3, 6))
for b in range(3):
    apt2[b, b] = q
    apt2[b, 3 + b] = -q
mu_red = m2[0] * m2[1] / (m2[0] + m2[1]) * AMU_TO_ELECTRON_MASS
a2 = ir_intensities(apt2, modes2)
exact2 = AU_TO_KM_PER_MOL * q * q / mu_red
check("N2 diatomic A = C q^2 / mu_red", len(a2) == 1 and abs(a2[0] / exact2 - 1) < 1e-12, f"{a2} vs {exact2}")

# N3 field route with a cubic field term
alpha = np.diag([2.0, 3.0, 4.0])
beta = 5.0


def grad_at_field(f):
    """dE/dR of E = E0 - mu.F - F.alpha(R).F/2 - beta |R|^2 F_z^3 / 6 on the model."""

    def energy(xflat):
        xm = xflat.reshape(-1, 3)
        scale = 1.0 + 0.1 * np.sum(xm**2)
        return (
            -dipole(xm) @ f
            - 0.5 * scale * f @ alpha @ f
            - beta * np.sum(xm**2) * f[2] ** 3 / 6.0
        )

    xflat = r0.ravel()
    gr = np.zeros(xflat.size)
    for i in range(xflat.size):
        xp, xm_ = xflat.copy(), xflat.copy()
        xp[i] += 1e-5
        xm_[i] -= 1e-5
        gr[i] = (energy(xp) - energy(xm_)) / 2e-5
    return gr


H_FIELD = 1e-3
p_field = field_route(grad_at_field, H_FIELD)
p_ref = apt_fd(r0)
# Truncation: the cubic term puts -(beta/3) x_i h^2 into the difference; the
# inner gradient and dipole stencils add ~1e-9.
n3_bound = beta * np.abs(r0).max() * H_FIELD**2 / 3.0 + 1e-8
n3_diff = np.abs(p_field - p_ref).max()
check("N3 field route = dmu/dR", n3_diff <= n3_bound, f"max diff {n3_diff:.2e} (bound {n3_bound:.2e})")

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nall checks passed")
