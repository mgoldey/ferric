#!/usr/bin/env python3
"""Positive-definiteness of the terfc kernel OFF the curvature-linked family.

The prior Fourier check (terfc-pq-metric-breakdown-gate0 memory) established
PD at the linked point r0*omega = 1/sqrt2 only. With (r0, omega) decoupled
(commit 03214031) the kernel terfc(r; r0, omega)/r must be re-checked before
any RI-METRIC use at free sharpness.

Reduction: terfc(r) = 1 - (erf(w(r+r0)) + erf(w(r-r0)))/2 depends only on
x = r/r0 and c = r0*omega, so PD is a ONE-parameter question. The 3D Fourier
transform of terfc(r)/r is (4 pi r0^2 / u) * P(u), u = q*r0, with

    P(u; c) = int_0^inf terfc(x; c) sin(u x) dx      (PD <=> P >= 0 for all u)

Known limits (anchors): c -> 0 gives Coulomb (P = 1/u > 0); c -> inf gives
the sharp-truncated Coulomb (P = (1 - cos u)/u >= 0, touching zero at
u = 2 pi k). Large-u tail: P -> terfc(0)/u = 1/u > 0. The danger zone is
moderate u at intermediate c, where the smoothed edge could push the
(1 - cos u)/u zeros negative.

Numerics: Gauss-Legendre on [0, xmax(c)] (integrand decays like
erfc(c(x-1))/2 beyond x = 1), nodes sized to resolve sin(u x) at the largest
u; vectorized over the u grid. Anchors verified in-run: sharp-limit formula
at c = 50 and Coulomb tail at c = 0.05.

Usage: python scripts/terfc_pd_check.py   (single CPU, seconds)
"""
import numpy as np

C_GRID = np.concatenate([
    np.linspace(0.2, 1.0, 17),
    np.linspace(1.0, 4.0, 31),
    np.linspace(4.0, 12.0, 17),
    [2.0**-0.5, 2.06987, 50.0],
])
U_GRID = np.concatenate([
    np.linspace(1e-3, 20.0, 2000),
    np.linspace(20.0, 100.0, 1600),
    np.geomspace(100.0, 400.0, 300),
])
PANEL_WIDTH = 0.05  # Bohr-free x units; ~3 sin periods per panel at u=400
PANEL_NODES = 32


def terfc(x, c):
    from scipy.special import erf
    return 1.0 - 0.5 * (erf(c * (x + 1.0)) + erf(c * (x - 1.0)))


# Composite GL panels, NOT one leggauss(20000): numpy's leggauss builds an
# n x n companion matrix (~3.2 GB at n=20000) at that size — it OOM-killed
# this sweep three times before main() printed a byte. 32-node panels of
# fixed width give the same resolution with O(1) memory; the in-run anchors
# below verify the quadrature end-to-end.
_PN, _PW = np.polynomial.legendre.leggauss(PANEL_NODES)


def p_of_u(c, u_grid):
    xmax = 1.0 + 8.0 / c
    # Panels must resolve BOTH the sin oscillation (PANEL_WIDTH) and the erf
    # edge at x=1 (width ~1/c; c=50 at 0.05-wide panels left a 6e-3 anchor
    # deviation, measured).
    width = min(PANEL_WIDTH, 0.25 / c)
    n_panels = max(int(np.ceil(xmax / width)), 8)
    edges = np.linspace(0.0, xmax, n_panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    x = (edges[:-1, None] + half * (_PN + 1.0)[None, :]).ravel()
    w = np.broadcast_to(half * _PW, (n_panels, PANEL_NODES)).ravel()
    f = terfc(x, c) * w
    # P(u_j) = sum_i f_i sin(u_j x_i); tail beyond xmax is < erfc(8)/2 ~ 6e-30.
    # Chunk the u axis so the sin matrix stays ~100 MB even at the largest xmax.
    out = np.empty(len(u_grid))
    for lo in range(0, len(u_grid), 500):
        blk = u_grid[lo:lo + 500]
        out[lo:lo + 500] = np.sin(np.outer(blk, x)) @ f
    return out


def main():
    print("# terfc kernel PD sweep off the linked family: P(u; c) >= 0 <=> PD")
    print(f"# c = r0*omega grid: {C_GRID.min():.2f}..{C_GRID.max():.0f} ({len(C_GRID)} pts); u grid to {U_GRID.max():.0f} ({len(U_GRID)} pts); GL panels {PANEL_NODES}x{PANEL_WIDTH}-wide", flush=True)

    # Anchors.
    p_sharp = p_of_u(50.0, U_GRID)
    ref_sharp = (1.0 - np.cos(U_GRID)) / U_GRID
    dev = np.abs(p_sharp - ref_sharp).max()
    print(f"# anchor c=50 vs sharp-cutoff formula: max|dev| = {dev:.2e}")
    p_coul = p_of_u(0.2, U_GRID[U_GRID > 200])
    ref_coul = 1.0 / U_GRID[U_GRID > 200]
    devc = np.abs(p_coul / ref_coul - 1.0).max()
    print(f"# anchor c=0.2 large-u vs terfc(0)/u tail: max rel dev = {devc:.2e}\n")

    print(f"{'c=r0w':>8} {'min P(u)':>12} {'at u':>8} {'min P*u':>10}  verdict")
    global_min = np.inf
    for c in np.sort(C_GRID):
        p = p_of_u(float(c), U_GRID)
        i = int(np.argmin(p))
        pu = p * U_GRID
        gm = float(p[i])
        global_min = min(global_min, gm)
        verdict = "PD" if gm >= -1e-12 else "NOT PD"
        mark = "  <-- linked" if abs(c - 2**-0.5) < 1e-9 else ("  <-- Dutoi bound" if abs(c - 2.06987) < 1e-9 else "")
        print(f"{c:>8.3f} {gm:>12.3e} {U_GRID[i]:>8.2f} {float(pu.min()):>10.3e}  {verdict}{mark}", flush=True)

    print(f"\n# global min P over sweep: {global_min:.3e}  ({'ALL PD' if global_min >= -1e-12 else 'PD VIOLATED'})")


if __name__ == "__main__":
    main()
