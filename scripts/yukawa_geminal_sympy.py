#!/usr/bin/env python3
"""Yukawa, Slater-geminal and fitted-geminal kernels.

Derivation: `book/src/04-operators/03-yukawa-geminal.md#yukawa-geminal`.
Code: crates/ferric-integrals/src/operator.rs (Operator::yukawa,
Operator::slater_geminal, Operator::stg and its FIT table, read from the
source by this script).

Claims (each prints "ok [label]" or "FAIL [label]"):
  1 [transforms]   The 3-D Fourier transforms (4 pi/k) int_0^inf r K(r) sin(kr) dr
                   are 4 pi/(k^2 + z^2) for the Yukawa kernel exp(-z r)/r and
                   8 pi z/(k^2 + z^2)^2 for the Slater geminal exp(-z r)
                   (sympy), both positive for z > 0, and both agree with a
                   direct mpmath quadrature at several k.
  2 [tu-reduction] The s-s Yukawa integral, written in Fourier space with the
                   Gaussian factor exp(-k^2/4 rho), depends on (rho, z, R) only
                   through T = rho R^2 and U = z^2/(4 rho), times sqrt(rho):
                   scaling rho -> l rho, z -> sqrt(l) z, R -> R/sqrt(l) keeps
                   T and U and multiplies the integral by sqrt(l) (quadrature).
  3 [coulomb-limit] At z -> 0 the same integral tends to erf(sqrt(rho) R)/R,
                   the Coulomb (Boys-function) value, from below.
  4 [fit]          The six-term fit of exp(-r) in Operator::stg (FIT, read from
                   operator.rs) gives sum_i c_i = 1.011522 at r = 0, zero slope
                   there (the exact slope is -1), and its largest deviation
                   from exp(-r) on [0, 10] is 0.0194, at r = 0.417.
  5 [stg-scaling]  With the code's scaling alpha_i -> gamma^2 alpha_i and
                   c_i -> -c_i/gamma, the fit approximates -exp(-gamma r)/gamma
                   with the same relative shape: max deviation 0.0194/gamma at
                   r = 0.417/gamma (gamma = 0.5, 1.3, 2.0).

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  - Slater transform denominator squared -> first power   ("FAIL [transforms]")
  - geminal exponent scaling gamma^2 -> gamma              ("FAIL [stg-scaling]")
  - Yukawa denominator k^2 + z^2 -> k^2 + z (breaks T, U)  ("FAIL [tu-reduction]")
Exits nonzero when any claim fails.
"""

import os
import re
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp

mp.mp.dps = 30
failures = []
OPERATOR_RS = "crates/ferric-integrals/src/operator.rs"


def repo_root():
    """The checkout: next to this script, or the working directory (the runner
    runs mutated copies from a scratch directory)."""
    cands = [Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents]
    if os.environ.get("FERRIC_ROOT"):
        cands.insert(0, Path(os.environ["FERRIC_ROOT"]))
    for c in cands:
        if (c / OPERATOR_RS).is_file():
            return c
    sys.exit("FAIL [fit] repository not found (set FERRIC_ROOT)")


ROOT = repo_root()


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        failures.append(label)


# 1. transforms ---------------------------------------------------------------
r, k, z = sp.symbols("r k z", positive=True)


def radial_ft(kernel):
    return sp.simplify(
        4 * sp.pi / k * sp.integrate(r * kernel * sp.sin(k * r), (r, 0, sp.oo))
    )


yuk = radial_ft(sp.exp(-z * r) / r)
sla = radial_ft(sp.exp(-z * r))
yuk_book = 4 * sp.pi / (k**2 + z**2)
sla_book = 8 * sp.pi * z / (k**2 + z**2) ** 2
sym_ok = sp.simplify(yuk - yuk_book) == 0 and sp.simplify(sla - sla_book) == 0


def sine_ft(power, zv, kv):
    """(4 pi/k) int_0^inf r^power exp(-z r) sin(k r) dr by oscillatory quadrature."""
    integrand = lambda x: x**power * mp.exp(-zv * x) * mp.sin(kv * x)
    return 4 * mp.pi / kv * mp.quadosc(integrand, [0, mp.inf], omega=kv)


worst = mp.mpf(0)
for zv in (mp.mpf("0.4"), mp.mpf("1.7")):
    for kv in (mp.mpf("0.3"), mp.mpf("1.0"), mp.mpf("4.0")):
        by = float(yuk_book.subs({k: kv, z: zv}))
        bs = float(sla_book.subs({k: kv, z: zv}))
        worst = max(
            worst,
            abs(sine_ft(0, zv, kv) - by) / by,
            abs(sine_ft(1, zv, kv) - bs) / bs,
        )
check(
    "transforms",
    sym_ok and worst < 1e-12,
    f"Yukawa {yuk}, Slater {sla}; quadrature worst rel {mp.nstr(worst, 3)}",
)


# 2./3. the s-s Yukawa integral in Fourier space ------------------------------
def ss_yukawa(rho, zeta, R):
    """(1/(2 pi)^3) int d^3k exp(-k^2/4 rho) 4 pi/(k^2 + zeta^2) exp(i k.R)."""
    f = lambda kk: (
        mp.exp(-(kk**2) / (4 * rho))
        * 4
        * mp.pi
        / (kk**2 + zeta**2)
        * mp.sin(kk * R)
        / (kk * R)
        * kk**2
    )
    return 4 * mp.pi / (2 * mp.pi) ** 3 * mp.quad(f, [0, mp.inf])


rho1, z1, R1 = mp.mpf("0.8"), mp.mpf("0.6"), mp.mpf("1.9")
worst_tu = mp.mpf(0)
for lam in (mp.mpf("0.3"), mp.mpf("2.5"), mp.mpf("9.0")):
    rho2, z2, R2 = lam * rho1, mp.sqrt(lam) * z1, R1 / mp.sqrt(lam)
    same_tu = (
        abs(rho2 * R2**2 - rho1 * R1**2) < 1e-25
        and abs(z2**2 / (4 * rho2) - z1**2 / (4 * rho1)) < 1e-25
    )
    ratio = ss_yukawa(rho2, z2, R2) / ss_yukawa(rho1, z1, R1)
    worst_tu = max(worst_tu, abs(ratio - mp.sqrt(lam)) if same_tu else mp.mpf(1))
check(
    "tu-reduction", worst_tu < 1e-20, f"worst |ratio - sqrt(l)| {mp.nstr(worst_tu, 3)}"
)

coul = mp.erf(mp.sqrt(rho1) * R1) / R1
near = ss_yukawa(rho1, mp.mpf("1e-6"), R1)
check(
    "coulomb-limit",
    abs(near - coul) / coul < 1e-5 and ss_yukawa(rho1, z1, R1) < coul and near < coul,
    f"z = 1e-6: rel {mp.nstr(abs(near - coul) / coul, 3)}; z = 0.6 below Coulomb",
)

# 4. the six-term fit of exp(-r), read from the code --------------------------
src = (ROOT / OPERATOR_RS).read_text()
block = re.search(r"const FIT: \[\(f64, f64\); 6\] = \[(.*?)\];", src, re.DOTALL).group(
    1
)
FIT = [
    (float(a.replace("_", "")), float(c.replace("_", "")))
    for a, c in re.findall(r"\(([\d_.]+), ([\d_.]+)\)", block)
]


def fit(x, gamma=1.0):
    """The code's composite operator: sum_i (-c_i/gamma) exp(-alpha_i gamma^2 x^2)."""
    g2 = gamma * gamma
    return sum(-c / gamma * mp.exp(-a * g2 * x**2) for a, c in FIT)


def max_dev(gamma):
    best = (mp.mpf(0), mp.mpf(0))
    for i in range(20001):
        x = mp.mpf(i) / 2000 / gamma
        d = abs(fit(x, gamma) + mp.exp(-gamma * x) / gamma)
        if d > best[0]:
            best = (d, x)
    return best


sum_c = sum(c for _, c in FIT)
slope0 = mp.diff(lambda x: -fit(x), 0)
dev1, at1 = max_dev(1.0)
check(
    "fit",
    len(FIT) == 6
    and abs(sum_c - mp.mpf("1.011522")) < 5e-7
    and abs(slope0) < 1e-20
    and abs(dev1 - mp.mpf("0.0194")) < 5e-5
    and abs(at1 - mp.mpf("0.417")) < 5e-4,
    f"sum c = {mp.nstr(sum_c, 7)}, slope(0) = {mp.nstr(slope0, 2)}, max dev {mp.nstr(dev1, 4)} at r = {mp.nstr(at1, 4)}",
)

# 5. scaling with gamma --------------------------------------------------------
ok = True
for g in (0.5, 1.3, 2.0):
    d, at = max_dev(g)
    ok &= abs(d * g - dev1) < 1e-12 and abs(at * g - at1) < 1e-9
check("stg-scaling", ok, "max deviation scales as 1/gamma at r/gamma")

if failures:
    print(f"{len(failures)} claim(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all yukawa/geminal claims hold")
