#!/usr/bin/env python3
"""Finite-field DF-MP2 relaxed dipole and static polarizability of water (PySCF).

Reference for crates/ferric-mp2/tests/cpks_polar.rs
(`cpks_static_relaxed_dipole_vs_pyscf`) and tests/ff_polar.rs
(`ff_relaxed_alpha_matches_pyscf_energy_hessian`). The relaxed MP2 dipole is
mu = -dE/dF and alpha = -d2E/dF2 of the total RHF + DF-MP2 energy in a uniform
field, with ferric's geometry and auxiliary basis (cc-pvdz-ri), exact-J/K RHF,
no frozen core. Central differences: h = 1e-4 (dipole, 2-point) and
h = 2e-3 (alpha, 5-point).

Run: python3 scripts/cpks/ff_dfmp2_dipole_alpha.py [basis]   (default sto-3g)
"""

import sys

import numpy as np
from pyscf import gto, scf
from pyscf.mp import dfmp2

basis = sys.argv[1] if len(sys.argv) > 1 else "sto-3g"
mol = gto.M(
    atom="O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
    basis=basis,
    unit="Angstrom",
    verbose=0,
)
r = mol.intor("int1e_r")
nuc = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords())


def energy(field, axis):
    mf = scf.RHF(mol)
    h0 = mf.get_hcore()
    # Electron charge -1: V = -F.mu = +F.r.
    mf.get_hcore = lambda *a: h0 + field * r[axis]
    mf.conv_tol = 1e-13
    mf.kernel()
    m = dfmp2.DFMP2(mf)
    m.with_df.auxbasis = "cc-pvdz-ri"
    m.kernel()
    return mf.e_tot + m.e_corr


mu = []
alpha = []
for ax in range(3):
    h = 1e-4
    mu.append(-(energy(h, ax) - energy(-h, ax)) / (2 * h) + nuc[ax])
    h2 = 2e-3
    e = [energy(k * h2, ax) for k in (-2, -1, 0, 1, 2)]
    d2 = (-e[4] + 16 * e[3] - 30 * e[2] + 16 * e[1] - e[0]) / (12 * h2 * h2)
    alpha.append(-d2)
print(f"{basis}: relaxed DF-MP2 dipole (a.u.) = [{mu[0]:+.8f} {mu[1]:+.8f} {mu[2]:+.8f}]")
print(f"{basis}: alpha diag (a.u.)            = [{alpha[0]:.6f} {alpha[1]:.6f} {alpha[2]:.6f}]")
