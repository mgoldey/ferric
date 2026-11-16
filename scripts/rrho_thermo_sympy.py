#!/usr/bin/env python3
"""RRHO thermochemistry as ferric computes it.

Derivation: crates/ferric-scf/theory.md, "RRHO thermochemistry".
Code: crates/ferric-scf/src/thermo.rs (translational, rotational, vibrational,
rrho_thermochemistry_with_masses and the CODATA constants), transcribed below.

Checks:
  K1  R_EH = k_B / E_h and the second radiation constant c2 = 100 h c / k_B
      (cm K) from CODATA 2018 (relative 1e-12).
  S1  (sympy) translation: from q_V = (2 pi m k T / h^2)^(3/2) V,
      E = k T^2 d ln q_V / dT = 3/2 k T, H = E + k T, and
      S = k ln(q_V e / N) + E / T equals the code's k (5/2 + ln q) with
      q = (2 pi m k T / h^2)^(3/2) k T / P (V / N = k T / P).
  S2  (sympy) rotation: linear q = k T / (sigma h B) gives E = k T,
      S = k (1 + ln q); nonlinear q ~ T^(3/2) gives E = 3/2 k T,
      S = k (3/2 + ln q).
  S3  (sympy) vibration: q = e^(-x/2) / (1 - e^-x), x = theta / T, gives
      E = k theta / 2 + k T x / (e^x - 1), S = k (x / (e^x - 1) - ln(1 - e^-x)),
      Cv = k x^2 e^x / (e^x - 1)^2, the code's forms.
  L1  (sympy) limits: x -> 0 (high T, classical): E - ZPE -> k T,
      Cv -> k, S - k (1 - ln x) -> 0; x -> oo (low T): E -> ZPE, S -> 0.
  N1  Independent numerical route: ln Q(T) of water (sigma = 2) summed from
      the levels (vibrational ladders summed to convergence; translation and
      rotation from their partition functions), E and S by central
      differences of ln Q in T (five-point stencil, step 1e-3 T, truncation
      ~1e-13 relative), against the transcribed closed forms (relative 1e-10).
  N2  Rigid-rotor level sum q = sum_J (2J + 1) exp(-J(J+1) h B / k T) / sigma
      against the closed form k T / (sigma h B): the difference is the
      Euler-Maclaurin series 1/3 + y/15 + 4y^2/315, y = h B / k T (relative
      1e-4 on the difference, the size of the next term) at HF's B, 298.15 K.
  N3  Argon at 298.15 K and 1 bar: S = 154.845 J/(mol K) (NIST-JANAF) within
      0.01.

Mutations (each turns the script red; exact one-occurrence substitutions):
  symmetry number ignored in the nonlinear rotor                 (N1 fails)
  translational entropy constant 5/2 -> 3/2                      (S1, N1 fail)
  vibrational entropy: - ln(1 - e^-x) -> + ln(1 - e^-x)          (S3, N1 fail)
  c2 built from h instead of h c                                 (K1 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import math
import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# CODATA 2018, independent of the code's constants.
K_B = 1.380649e-23
H_PL = 6.62607015e-34
C_L = 299792458.0
E_H = 4.3597447222071e-18
M_U = 1.66053906660e-27
N_A = 6.02214076e23
BOHR_M = 0.52917721092e-10  # ferric's units::BOHR_RADIUS_ANGSTROM

# ---------------------------------------------------------------------------
# Transcribed from thermo.rs.
BOLTZMANN_J_PER_K = 1.380_649e-23
PLANCK_J_S = 6.626_070_15e-34
LIGHT_SPEED_M_PER_S = 299_792_458.0
HARTREE_J = 4.359_744_722_207_1e-18
ATOMIC_MASS_KG = 1.660_539_066_60e-27
R_EH = BOLTZMANN_J_PER_K / HARTREE_J
C2 = 100.0 * PLANCK_J_S * LIGHT_SPEED_M_PER_S / BOLTZMANN_J_PER_K


def translational(mass_amu, t, p):
    m = mass_amu * ATOMIC_MASS_KG
    kt = BOLTZMANN_J_PER_K * t
    q = (2.0 * math.pi * m * kt / (PLANCK_J_S * PLANCK_J_S)) ** 1.5 * kt / p
    return dict(e=1.5 * R_EH * t, h=2.5 * R_EH * t, s=R_EH * (2.5 + math.log(q)))


def rotational_nonlinear(b_hz, sigma, t):
    kt_over_h = BOLTZMANN_J_PER_K * t / PLANCK_J_S
    abc = b_hz[0] * b_hz[1] * b_hz[2]
    q = kt_over_h**1.5 * math.sqrt(math.pi) / (sigma * math.sqrt(abc))
    return dict(e=1.5 * R_EH * t, h=1.5 * R_EH * t, s=R_EH * (1.5 + math.log(q)))


def vibrational(real_cm, t):
    zpe = 0.0
    s = e_th = 0.0
    for nu in real_cm:
        theta = C2 * nu
        x = theta / t
        bose = 1.0 / math.expm1(x)
        zpe += 0.5 * R_EH * theta
        s += x * bose - math.log(-math.expm1(-x))
        e_th += x * bose
    e = zpe + R_EH * t * e_th
    return dict(e=e, h=e, s=R_EH * s), zpe


def rotational_constant_hz(moment_amu_bohr2):
    i_si = moment_amu_bohr2 * ATOMIC_MASS_KG * BOHR_M * BOHR_M
    return PLANCK_J_S / (8.0 * math.pi * math.pi * i_si)


# ---------------------------------------------------------------------------
# K1 constants
r_ref = K_B / E_H
c2_ref = 1.438776877e-2 * 100  # second radiation constant, CODATA 2018 (m K -> cm K)
check("K1 R_EH = k_B / E_h", abs(R_EH / r_ref - 1) < 1e-12, f"{R_EH:.12e}")
check("K1 c2 = 100 h c / k_B", abs(C2 / c2_ref - 1) < 1e-9, f"{C2:.10f} vs {c2_ref}")

# ---------------------------------------------------------------------------
# S1-S3 symbolic derivations from partition functions
T, k, h, m, V, N, P, sigma, B, theta = sp.symbols(
    "T k h m V N P sigma B theta", positive=True
)
A_, B_, C_ = sp.symbols("A B C", positive=True)


def e_from(q):
    return sp.simplify(k * T**2 * sp.diff(sp.log(q), T))


# Translation
qV = (2 * sp.pi * m * k * T / h**2) ** sp.Rational(3, 2) * V
E_tr = e_from(qV)
S_tr = k * sp.log(qV * sp.E / N) + E_tr / T  # Stirling: ln(q^N / N!) / N
q_code = (2 * sp.pi * m * k * T / h**2) ** sp.Rational(3, 2) * k * T / P
S_code = k * (sp.Rational(5, 2) + sp.log(q_code))
S_tr_sub = S_tr.subs(V, N * k * T / P)
check("S1 E_trans = 3/2 kT", sp.simplify(E_tr - sp.Rational(3, 2) * k * T) == 0)
check(
    "S1 S_trans = k (5/2 + ln q)",
    sp.simplify(sp.expand_log(S_tr_sub - S_code, force=True)) == 0,
)
# Same identity against the transcribed constant 2.5 (numerical point).
pt = {m: 18 * 1.66e-27, k: K_B, h: H_PL, T: 298.15, P: 101325.0, N: 1}
s_num = float(S_tr_sub.subs(pt)) / E_H
s_tr_code = translational(18.0 * 1.66e-27 / ATOMIC_MASS_KG, 298.15, 101325.0)["s"]
check("S1 transcribed S_trans", abs(s_num - s_tr_code) < 1e-14, f"{s_num} vs {s_tr_code}")

# Rotation
q_lin = k * T / (sigma * h * B)
check("S2 E_rot linear = kT", sp.simplify(e_from(q_lin) - k * T) == 0)
check(
    "S2 S_rot linear = k (1 + ln q)",
    sp.simplify(k * sp.log(q_lin) + e_from(q_lin) / T - k * (1 + sp.log(q_lin))) == 0,
)
q_nl = (k * T / h) ** sp.Rational(3, 2) * sp.sqrt(sp.pi) / (sigma * sp.sqrt(A_ * B_ * C_))
check("S2 E_rot nonlinear = 3/2 kT", sp.simplify(e_from(q_nl) - sp.Rational(3, 2) * k * T) == 0)

# Vibration
x = sp.symbols("x", positive=True)
q_v = sp.exp(-theta / (2 * T)) / (1 - sp.exp(-theta / T))
E_v = e_from(q_v)
S_v = k * sp.log(q_v) + E_v / T
Cv_v = sp.diff(E_v, T)
xs = theta / T
E_code = k * theta / 2 + k * T * xs / (sp.exp(xs) - 1)
S_code_v = k * (xs / (sp.exp(xs) - 1) - sp.log(1 - sp.exp(-xs)))
Cv_code = k * xs**2 * sp.exp(xs) / (sp.exp(xs) - 1) ** 2
# exp() forms do not always simplify to 0; evaluate at exact rational points
# to 40 digits instead (an identity holds at every point).
def vanishes(expr):
    pts = [{k: 1, theta: sp.Rational(th), T: sp.Rational(tt)} for th, tt in ((2000, 300), (150, 900), (5000, 77))]
    return all(abs(sp.N(expr.subs(p0), 40)) < sp.Float("1e-30") for p0 in pts)


check("S3 E_vib", vanishes(E_v - E_code))
check("S3 S_vib", vanishes(S_v - S_code_v))
check("S3 Cv_vib", vanishes(Cv_v - Cv_code))
# The transcribed vibrational entropy at one point.
vt, _ = vibrational([1500.0], 298.15)
s_sym = float(S_code_v.subs({k: R_EH, theta: C2 * 1500.0, T: 298.15}))
check("S3 transcribed S_vib", abs(vt["s"] - s_sym) < 1e-18, f"{vt['s']} vs {s_sym}")

# L1 limits in x = theta / T
E_x = k * theta * (sp.Rational(1, 2) + 1 / (sp.exp(x) - 1))  # E with T = theta / x
ET_x = E_x.subs(theta, x * T)
S_x = k * (x / (sp.exp(x) - 1) - sp.log(1 - sp.exp(-x)))
Cv_x = k * x**2 * sp.exp(x) / (sp.exp(x) - 1) ** 2
zpe_x = k * x * T / 2
check("L1 high T: E - ZPE -> kT", sp.limit((ET_x - zpe_x) / (k * T), x, 0) == 1)
check("L1 high T: Cv -> k", sp.limit(Cv_x / k, x, 0) == 1)
check("L1 high T: S -> k (1 - ln x)", sp.limit((S_x - k * (1 - sp.log(x))) / k, x, 0) == 0)
check("L1 low T: E -> ZPE", sp.limit((ET_x - zpe_x) / (k * T), x, sp.oo) == 0)
check("L1 low T: S -> 0", sp.limit(S_x, x, sp.oo) == 0)

# ---------------------------------------------------------------------------
# N1 independent route: ln Q of water from level sums, E and S by FD in T.
MASS = {"O": 15.999, "H": 1.008}
geom_ang = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)]
masses = np.array([MASS[s] for s, *_ in geom_ang])
xyz = np.array([c for _, *c in geom_ang]) / 0.52917721092  # Bohr
com = masses @ xyz / masses.sum()
r = xyz - com
inertia = sum(mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)) for mi, ri in zip(masses, r))
moments = np.sort(np.linalg.eigvalsh(inertia))
b_hz = [rotational_constant_hz(i) for i in moments]
freqs = [1650.0, 3800.0, 3900.0]
SIGMA = 2


def transcribed_total(t, p):
    tr = translational(masses.sum(), t, p)
    rot = rotational_nonlinear(b_hz, SIGMA, t)
    vib, _ = vibrational(freqs, t)
    return tr["e"] + rot["e"] + vib["e"], tr["s"] + rot["s"] + vib["s"]


def ln_q_molecule(t, p):
    """ln of the per-molecule partition function with V/N = kT/P (no e)."""
    kt = K_B * t
    ln_tr = 1.5 * math.log(2 * math.pi * masses.sum() * M_U * kt / H_PL**2) + math.log(kt / p)
    ln_rot = (
        1.5 * math.log(kt / H_PL)
        + 0.5 * math.log(math.pi)
        - math.log(SIGMA)
        - 0.5 * sum(math.log(b) for b in b_hz)
    )
    ln_vib = 0.0
    for nu in freqs:
        eps = H_PL * C_L * 100 * nu  # J
        n = np.arange(0, 400)
        ln_vib += math.log(np.sum(np.exp(-(n + 0.5) * eps / kt)))
    return ln_tr, ln_rot + ln_vib


def numeric_total(t, p):
    # E = kT^2 d lnq_int/dT + 3/2 kT (translation at fixed V); S from
    # S = k (ln q_V e / N) + E / T with V / N = kT / P.
    dt = 1e-3 * t
    q_int = lambda tt: ln_q_molecule(tt, p)[1]  # noqa: E731
    # Five-point stencil: truncation dt^4 / 30 |f^(5)| ~ 1e-13 relative.
    d1 = (q_int(t - 2 * dt) - 8 * q_int(t - dt) + 8 * q_int(t + dt) - q_int(t + 2 * dt)) / (12 * dt)
    e_int = K_B * t**2 * d1
    e = 1.5 * K_B * t + e_int
    ln_tr, ln_int = ln_q_molecule(t, p)
    s = K_B * (ln_tr + ln_int + 1.0) + e / t
    return e / E_H, s / E_H


for t in (100.0, 298.15, 1000.0):
    e_c, s_c = transcribed_total(t, 101325.0)
    e_n, s_n = numeric_total(t, 101325.0)
    ok = abs(e_c / e_n - 1) < 1e-10 and abs(s_c / s_n - 1) < 1e-10
    check(f"N1 water T={t}: E, S closed form = level sums (claim N1)", ok, f"{e_c:.10e} {e_n:.10e} {s_c:.10e} {s_n:.10e}")

# N2 rigid-rotor level sum vs closed form (HF, B ~ 20.56 cm^-1)
b_hf = 20.56 * C_L * 100  # Hz
t = 298.15
y = H_PL * b_hf / (K_B * t)
jj = np.arange(0, 400)
q_sum = np.sum((2 * jj + 1) * np.exp(-jj * (jj + 1) * y))
q_closed = 1.0 / y
corr = 1.0 / 3.0 + y / 15.0 + 4.0 * y * y / 315.0
check(
    "N2 rotor level sum = kT/hB + 1/3 + y/15 + 4y^2/315",
    abs((q_sum - q_closed) / corr - 1) < 1e-4,
    f"{q_sum - q_closed:.8f} vs {corr:.8f}",
)

# N3 argon (Sackur-Tetrode)
s_ar = translational(39.948, 298.15, 1.0e5)["s"] * E_H * N_A
check("N3 Ar S(298.15 K, 1 bar) = 154.845 J/(mol K)", abs(s_ar - 154.845) < 0.01, f"{s_ar:.4f}")

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nall checks passed")
