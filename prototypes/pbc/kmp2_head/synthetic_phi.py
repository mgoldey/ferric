"""Synthetic test of the lattice head weight Phi (issue #230): does a Gamma-centred n^3 mesh with the q = 0 point
replaced by Phi[f0] * w(0) / N_k recover the BZ integral of a q -> 0 degree-0 integrand f0(q^) with l = 4 content?

Model integrand (a = 1 cell, q = 2 pi m / n, periodic smooth envelope w; the singular direction part is exactly the
|h|^2 structure of the MP2 head, f0 = (u . d)^4):
    g(q) = (u.d)^4 w(q) [+ c2 |q|^2 (u.d)^4 w(q), a degree-2 non-analytic term],   w = prod_i ((1+cos q_i)/2)^kappa  (kappa = P = 2).
Phi[f0] = -Z[f0], Z = lim [sum_{m != 0} f0(m^) e^(-eps m^2) - (pi/eps)^(3/2) <f0>] (Richardson in eps) -- the same
definition as run_kcorr_head_anomaly.lattice_phi, here for an arbitrary direction d.
Reference integral: trapezoid on n = 120, 160 (corrected) extrapolated in n^-5.

Artifact hypotheses (before running):
  * Phi exact (physics of the mesh): corrected-mesh error falls as n^-5 (next homogeneous degree, 2 -> n^-(3+2)),
    NOT n^-3; uncorrected (m = 0 dropped) error = Phi_exact/N -> n^-3.
  * Phi wrong / l = 4 weighting error: corrected error keeps an n^-3 tail with coefficient w(0)*(Phi_used - Phi_exact).
  * Anchor: using the WRONG weight (angular average 0.2, or the cubic-head 1/9) must reproduce the n^-3 tail with
    exactly that coefficient (a test that cannot fail would be an assumption).
  * Odd (degree-1) terms: g(q) = g(-q) here by construction, so n^-4 is absent; that is the time-reversal premise.
"""

import sys

import numpy as np


def phi_dir(d, R=110, eps=(0.01, 0.005, 0.0025)):
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    m = np.arange(-R, R + 1, dtype=float)
    X, Y, Zc = np.meshgrid(m, m, m, indexing="ij", sparse=True)
    r2 = X**2 + Y**2 + Zc**2
    msk = r2 > 0
    u = (d[0] * X + d[1] * Y + d[2] * Zc) ** 2 / np.where(msk, r2, 1.0)
    g = np.where(msk, u**2, 0.0)
    ang = 1 / 5  # <(u.d)^4> over the sphere
    z = [(g * np.exp(-e * r2)).sum() - (np.pi / e) ** 1.5 * ang for e in eps]
    return -(8 * z[2] - 6 * z[1] + z[0]) / 3


def mesh_sum(n, d, kappa, c2, phi_w):
    """(1/N) [ sum_{m != 0} g + phi_w * w(0) ] for the model integrand."""
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    m = np.arange(n)
    m = np.where(m > n // 2, m - n, m) if n % 2 else np.where(m >= n // 2, m - n, m)
    q = 2 * np.pi * m / n
    Qx, Qy, Qz = np.meshgrid(q, q, q, indexing="ij", sparse=True)
    q2 = Qx**2 + Qy**2 + Qz**2
    # for the angular part use the *folded* q (smallest image), as the mesh direction at the singular point
    msk = q2 > 0
    u2 = (d[0] * Qx + d[1] * Qy + d[2] * Qz) ** 2 / np.where(msk, q2, 1.0)
    # w = prod_i ((1 + cos q_i)/2)^P: periodic, analytic at q = 0, vanishing to order 2P at every other zone
    # corner/face (q_i = pi): no singularity but q = 0 and no zone-boundary jump of f0(q^)
    w = (((1 + np.cos(Qx)) * (1 + np.cos(Qy)) * (1 + np.cos(Qz))) / 8.0) ** kappa
    g = np.where(msk, u2**2 * w * (1.0 + c2 * q2), 0.0)
    return (g.sum() + phi_w) / n**3, g.sum() / n**3


KAPPA, C2 = 2, 0.3
NS = np.arange(5, 11)


def reference(d, phi):
    """BZ integral of the model: corrected trapezoid at n = 120, 160, extrapolated in n^-5."""
    ref = [mesh_sum(n, d, KAPPA, C2, phi)[0] for n in (120, 160)]
    return (ref[1] * 160**5 - ref[0] * 120**5) / (160**5 - 120**5)


def detector(d, i_ref, phi_used):
    """(A, p): A = n^-3 coefficient from a fit  err*n^3 = A + B n^-2  on n = 5..10; p = tail-3 log-log exponent of
    |err| (n = 8, 9, 10).  A is LINEAR in the weight (A shifts by exactly dPhi * w(0), w(0) = 1) and is the sound
    detector; p is non-monotone in dPhi (the error crosses zero) and is reported only as a cross-check."""
    err = np.array([mesh_sum(n, d, KAPPA, C2, phi_used)[0] - i_ref for n in NS])
    y = err * NS**3
    a = np.linalg.lstsq(np.array([np.ones(len(NS)), NS**-2.0]).T, y, rcond=None)[0][0]
    p = np.polyfit(np.log(NS[-3:]), np.log(np.abs(err[-3:])), 1)[0]
    return a, p


A_PASS = 0.03  # |A| below this = "weight verified"; resolution of the n = 5..10 window is ~0.02 (pre-asymptotic bias)
D_MUT = 0.05  # +-0.05 weight mutant (30% of |Phi_z|, 11-17% of Phi_[110]/[111]) must flip the verdict


def run(name, d):
    phi = phi_dir(d)
    i_ref = reference(d, phi)
    rows = []
    for n in range(3, 11):
        corr = mesh_sum(n, d, KAPPA, C2, phi)[0] - i_ref
        raw = mesh_sum(n, d, KAPPA, C2, 0.0)[1] - i_ref
        ang = mesh_sum(n, d, KAPPA, C2, 0.2)[0] - i_ref
        rows.append((n, raw * n**3, ang * n**3, corr * n**3, corr * n**5))
    print(
        f"\n[{name}] Phi_exact = {phi:.6f} (angular average 0.2, cubic head 1/9 = 0.1111)"
    )
    print("  n   raw*n^3   angavg*n^3   corrected*n^3   corrected*n^5")
    for r in rows:
        print(f"  {r[0]:2d} {r[1]:9.5f} {r[2]:11.5f} {r[3]:14.6f} {r[4]:14.5f}")
    a0, p0 = detector(d, i_ref, phi)
    aw, pw = detector(d, i_ref, 0.2)
    print(f"  exact Phi : A = {a0:+.4f}, tail-3 exponent {p0:.2f}  (n^-5 expected)")
    print(f"  wrong 0.2 : A = {aw:+.4f}, tail-3 exponent {pw:.2f}  (n^-3 expected)")
    scan = [(dp, *detector(d, i_ref, phi + dp)) for dp in (0.005, 0.01, 0.02, 0.05)]
    print(
        "  dPhi scan (A, exponent): "
        + "; ".join(f"{dp:+.3f}: {a:+.4f}, {p:.2f}" for dp, a, p in scan)
    )
    assert abs(a0) < A_PASS, a0
    assert -5.8 < p0 < -4.3, p0
    # real detector test: the wrong weight is seen (A = 0.2 - Phi, exponent ~ -3), the +-D_MUT mutants flip
    assert abs(aw - (0.2 - phi)) < A_PASS, aw
    assert -3.3 < pw < -2.7, pw
    for sgn in (+1, -1):
        am, _ = detector(d, i_ref, phi + sgn * D_MUT)
        assert abs(am) > A_PASS, (sgn, am)
    return rows


def tensor_anchor():
    """Phi_abcd cubic tensor (Phi_aaaa, Phi_aabb = (1 - 3 Phi_aaaa)/6) reproduces the direct lattice Phi for any d."""
    p4 = phi_dir((0, 0, 1))
    p22 = (1 - 3 * p4) / 6
    for d in ((1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 3)):
        v = np.asarray(d, float) / np.linalg.norm(d)
        pred = p4 * (v**4).sum() + p22 * 3 * ((v**2).sum() ** 2 - (v**4).sum())
        print(f"  tensor anchor d={d}: tensor {pred:.6f} direct {phi_dir(d):.6f}")
        assert abs(pred - phi_dir(d)) < 2e-5


if __name__ == "__main__":
    tensor_anchor()
    for name, d in (("z", (0, 0, 1)), ("[110]", (1, 1, 0)), ("[111]", (1, 1, 1))):
        run(name, d)
    sys.exit(0)
