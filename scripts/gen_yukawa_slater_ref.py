"""Independent numerical reference for ferric's native libint2 Yukawa/Slater ERIs.

Generates the H2/STO-3G (00|00) reference values hard-coded in the
ferric-integrals engine tests:
  - test_yukawa_quartet_matches_independent_numerical_reference
  - test_exact_slater_geminal_matches_gaussian_fit

Why an in-house quadrature and not PySCF: PySCF's Yukawa 2e intor `int2e_yp` is
absent from the libcgto build in this environment (undefined symbol
`int2e_yp_optimizer`). We instead evaluate the integrals analytically via the
Gaussian product theorem + a 1-D quadrature, and VALIDATE the method by checking
its zeta->0 limit reproduces PySCF's `int2e` Coulomb (00|00) = 0.774605943920
exactly (run this script; the printed Coulomb value must match).

Operators:
  Yukawa:          exp(-zeta r12) / r12   (libint2 Operator::stg_x_coulomb)
  Slater geminal:  exp(-gamma r12)        (libint2 Operator::stg)

Diagonal (00|00): both electrons occupy the SAME STO-3G 1s on the SAME atom, so
the density is single-center. The relative-coordinate density of two s-Gaussian
blobs (exponents p, q) is a Gaussian of exponent mu = p q / (p+q):
    P(s) = (mu/pi)^{3/2} exp(-mu s^2).
Then:
    Yukawa blob  = 4 pi (mu/pi)^{3/2} ∫_0^inf s exp(-mu s^2 - zeta s) ds
    Coulomb blob = 2 sqrt(mu/pi)
    Slater blob  = 4 pi (mu/pi)^{3/2} ∫_0^inf s^2 exp(-mu s^2 - gamma s) ds
    fit blob     = sum_k c_k (mu/(mu+alpha_k))^{3/2}   (6-term Tew-Klopper fit)
"""
import numpy as np
from scipy import integrate

# STO-3G H 1s
alphas = np.array([3.42525091, 0.62391373, 0.16885540])
coeffs = np.array([0.15432897, 0.53532814, 0.44463454])
Nprim = (2.0 * alphas / np.pi) ** 0.75
d = coeffs * Nprim
def s_overlap(a, b): return (np.pi / (a + b)) ** 1.5
S = sum(d[i] * d[j] * s_overlap(alphas[i], alphas[j]) for i in range(3) for j in range(3))
d = d / np.sqrt(S)  # unit-normalized contraction

# rho0 as a sum of unit-normalized s-Gaussian blobs (weights sum to 1).
p_list, w_list = [], []
for i in range(3):
    for j in range(3):
        p = alphas[i] + alphas[j]
        w = d[i] * d[j] * (np.pi / p) ** 1.5
        p_list.append(p); w_list.append(w)
p_arr, w_arr = np.array(p_list), np.array(w_list)
assert abs(w_arr.sum() - 1.0) < 1e-10

def yukawa_blob(p, q, zeta):
    mu = p * q / (p + q)
    f = lambda s: s * np.exp(-mu * s * s - zeta * s)
    val, _ = integrate.quad(f, 0.0, np.inf, limit=200)
    return 4.0 * np.pi * (mu / np.pi) ** 1.5 * val

def coulomb_blob(p, q):
    mu = p * q / (p + q)
    return 2.0 * np.sqrt(mu / np.pi)

def slater_blob_exact(p, q, gamma):
    mu = p * q / (p + q)
    f = lambda s: s * s * np.exp(-mu * s * s - gamma * s)
    val, _ = integrate.quad(f, 0.0, np.inf, limit=200)
    return 4.0 * np.pi * (mu / np.pi) ** 1.5 * val

FIT = [(0.241393, 0.301846), (0.844001, 0.255338), (3.044055, 0.197575),
       (13.499604, 0.139390), (76.617811, 0.082572), (765.962887, 0.034801)]
def slater_blob_fit(p, q, gamma):
    mu = p * q / (p + q)
    g2 = gamma * gamma
    return sum(c * (mu / (mu + a * g2)) ** 1.5 for a, c in FIT)

def contract(blob_fn, *args):
    return sum(w_arr[a] * w_arr[b] * blob_fn(p_arr[a], p_arr[b], *args)
               for a in range(9) for b in range(9))

if __name__ == "__main__":
    print("=== Yukawa exp(-zeta r)/r, (00|00) H2/STO-3G ===")
    for zeta in [0.5, 1.0, 2.0]:
        print("  zeta=%.3f : %.12f" % (zeta, contract(yukawa_blob, zeta)))
    c = sum(w_arr[a] * w_arr[b] * coulomb_blob(p_arr[a], p_arr[b])
            for a in range(9) for b in range(9))
    print("  Coulomb  : %.12f   (must equal PySCF int2e 0.774605943920)" % c)

    print("=== Slater geminal exp(-gamma r), (00|00) H2/STO-3G, gamma=1 ===")
    ex = contract(slater_blob_exact, 1.0)
    ft = contract(slater_blob_fit, 1.0)
    print("  exact    : %.12f" % ex)
    print("  6-term fit: %.12f" % ft)
    print("  fit rel err: %.4e (%.2f%%)" % (abs(ft - ex) / ex, 100 * abs(ft - ex) / ex))
