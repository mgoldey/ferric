"""Iteration 16: huge-box limit of the Gamma RHF force -> molecular RHF gradient (PySCF molecular, independent).

Prediction (stated before running): the Gamma energy residual is E_box - E_mol = c3/a^3 + ..., c3 = -(4pi/3) sigma^2
(Iteration 1, exchange Makov-Payne; the neutral H2 density has no dipole so J contributes nothing at a^-3).  sigma^2
depends on geometry, so the FORCE residual is
    g_box - g_mol = -(4pi/3) (d sigma^2 / dR) / a^3 + O(a^-5)
with d sigma^2/dR from FD of the molecular occupied-orbital second central moment.  exxdiv=none and =ewald give the
SAME force (E_M = -v_M/4 tr(DSDS) = -v_M N/2 is geometry-independent at Gamma RHF).
Artifact predictions: a missing term would leave an a-INDEPENDENT offset (e.g. drop the basis motion in V_ne: O(0.1));
a missing Madelung S-term would make the ewald force differ from none by v_M tr(D dS/dR) ~ 1/a.
"""

import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
from pbc_gamma import Cell

R0 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
OFF = np.array([0.3, 0.2, 0.1])


def mol_ref(atoms):
    m = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
    mf = m.RHF()
    mf.conv_tol = 1e-13
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    Co = mf.mo_coeff[:, :1]
    Do = Co @ Co.T
    r = np.einsum("xmn,mn->x", m.intor("int1e_r_cart"), Do)
    s2 = np.sum(m.intor("int1e_r2_cart") * Do) - r @ r
    return g, s2


def main():
    g_mol, s2 = mol_ref(R0)
    h = 1e-4
    sp = mol_ref([R0[0], ("H", (0, 0, 1.4 + h))])[1]
    sm = mol_ref([R0[0], ("H", (0, 0, 1.4 - h))])[1]
    ds2 = (sp - sm) / (2 * h)  # d sigma^2 / d z_1 ; d/dz_0 = -ds2
    c3_pred = -(4 * np.pi / 3) * ds2
    print(f"molecular: g_z(H1) = {g_mol[1, 2]:.10f}  sigma^2 = {s2:.7f}  d sigma^2/dz1 = {ds2:.7f}  "
          f"predicted force c3 (atom 1, z) = {c3_pred:.5f}")
    res = {}
    for a in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
        cell = Cell(np.eye(3) * a, [(s, np.asarray(r) + OFF) for s, r in R0], "sto-3g")
        t = time.time()
        r = PGd.gamma_rhf_grad(cell, 2, exxdiv="ewald")
        d = r["grad"] - g_mol
        res[a] = d[1, 2]
        extra = ""
        if a == 8.0:
            rn = PGd.gamma_rhf_grad(cell, 2, exxdiv="none")
            extra = f"  |g_ewald - g_none| = {abs(rn['grad'] - r['grad']).max():.1e}"
        print(f"a={a:4.0f}  g_box-g_mol (H1 z) = {d[1, 2]:+.6e}  max|transverse| = {abs(d[:, :2]).max():.1e}  "
              f"a^3*d = {d[1, 2] * a**3:+.5f}  sumF {abs(r['grad'].sum(0)).max():.0e} ({time.time() - t:.0f}s){extra}",
              flush=True)
    for pts in ((12.0, 14.0), (14.0, 16.0), (16.0, 18.0), (18.0, 20.0)):
        X = np.array([[a**-3, a**-5] for a in pts])
        c3, c5 = np.linalg.solve(X, [res[a] for a in pts])
        print(f"fit c3/a^3 + c5/a^5 on {pts}: c3 = {c3:.5f} (pred {c3_pred:.5f}, rel {c3 / c3_pred - 1:+.1e}), c5 = {c5:.2f}")
    a3 = np.array(sorted(res))
    p = np.polyfit(np.log(a3[-3:]), np.log(np.abs([res[a] for a in a3[-3:]])), 1)[0]
    print(f"3-pt power exponent {tuple(a3[-3:])}: {p:.3f}")


if __name__ == "__main__":
    main()
