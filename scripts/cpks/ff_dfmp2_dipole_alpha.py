#!/usr/bin/env python3
"""Finite-field DF-MP2 relaxed dipole and static polarizability (PySCF).

Reference for crates/ferric-mp2/tests/cpks_polar.rs
(`cpks_static_relaxed_dipole_vs_pyscf`, `relaxed_alpha_matches_pyscf_energy_hessian`)
and tests/ff_polar.rs (`ff_relaxed_alpha_matches_pyscf_energy_hessian`). The
relaxed MP2 dipole is mu = -dE/dF and alpha = -d2E/dF2 of the total RHF +
DF-MP2 energy in a uniform field, with ferric's geometry and auxiliary basis
(cc-pvdz-ri), exact-J/K RHF, no frozen core. Central differences: h = 1e-4
(dipole, 2-point), h = 2e-3 (alpha: 5-point diagonal, 4th-order 16-point
mixed stencil off the diagonal).

Run: python3 scripts/cpks/ff_dfmp2_dipole_alpha.py [basis] [water|nh3] [--full]
(defaults sto-3g water; --full adds the off-diagonal alpha elements)
"""

import sys

import numpy as np
from pyscf import gto, scf
from pyscf.mp import dfmp2

GEOMS = {
    "water": "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
    # NH3 in a generic orientation (no principal axis along x, y or z).
    "nh3": "N 0.001600 -0.055708 0.103215; H -0.560130 0.790564 0.124323; "
    "H 0.928089 0.227984 -0.202386; H -0.379098 -0.628661 -0.644480",
}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
basis = args[0] if args else "sto-3g"
name = args[1] if len(args) > 1 else "water"
full = "--full" in sys.argv
mol = gto.M(atom=GEOMS[name], basis=basis, unit="Angstrom", verbose=0)
r = mol.intor("int1e_r")
nuc = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords())


def energy(field):
    mf = scf.RHF(mol)
    h0 = mf.get_hcore()
    # Electron charge -1: V = -F.mu = +F.r.
    v = np.einsum("x,xij->ij", np.asarray(field, dtype=float), r)
    mf.get_hcore = lambda *a: h0 + v
    mf.conv_tol = 1e-13
    mf.kernel()
    m = dfmp2.DFMP2(mf)
    m.with_df.auxbasis = "cc-pvdz-ri"
    m.kernel()
    return mf.e_tot + m.e_corr


def unit(ax, s):
    f = np.zeros(3)
    f[ax] = s
    return f


mu = []
alpha = np.zeros((3, 3))
h, h2 = 1e-4, 2e-3
for ax in range(3):
    mu.append(-(energy(unit(ax, h)) - energy(unit(ax, -h))) / (2 * h) + nuc[ax])
    e = [energy(unit(ax, k * h2)) for k in (-2, -1, 0, 1, 2)]
    alpha[ax, ax] = -(-e[4] + 16 * e[3] - 30 * e[2] + 16 * e[1] - e[0]) / (12 * h2 * h2)
if full:
    # 4th-order mixed second derivative (16-point stencil).
    w = {(1, -2): 8, (2, -1): 8, (-2, 1): 8, (-1, 2): 8,
         (-1, -2): -8, (-2, -1): -8, (1, 2): -8, (2, 1): -8,
         (2, -2): -1, (-2, 2): -1, (-2, -2): 1, (2, 2): 1,
         (-1, -1): 64, (1, 1): 64, (1, -1): -64, (-1, 1): -64}
    for a in range(3):
        for b in range(a + 1, 3):
            s = sum(c * energy(unit(a, i * h2) + unit(b, j * h2)) for (i, j), c in w.items())
            alpha[a, b] = alpha[b, a] = -s / (144 * h2 * h2)
print(f"{name}/{basis}: relaxed DF-MP2 dipole (a.u.) = [{mu[0]:+.8f} {mu[1]:+.8f} {mu[2]:+.8f}]")
print(f"{name}/{basis}: alpha diag (a.u.)            = [{alpha[0,0]:.6f} {alpha[1,1]:.6f} {alpha[2,2]:.6f}]")
if full:
    print(f"{name}/{basis}: alpha xy xz yz (a.u.)        = [{alpha[0,1]:.6f} {alpha[0,2]:.6f} {alpha[1,2]:.6f}]")
