"""External stability reference for crates/ferric-scf/tests/scf_stability.rs.

Regenerates the three PySCF numbers the decisive test asserts against, on
HeNe+ / def2-SVP / UHF at R = 2.0 Angstrom:

  1. the STABLE minimum PySCF's default guess finds, and its stability() verdict;
  2. the 2-Pi state ferric converges to (reached here with mom_occ), and its
     stability() verdict;
  3. PySCF's OWN orbital-Hessian lowest eigenvalue on that pi state, built by
     applying newton_ah.gen_g_hop_uhf's h_op to every unit vector and
     diagonalizing. This is the independent NUMBER (not just the boolean) that
     ferric's Davidson result is checked against.

Measured with PySCF 2.13.0:
    stable minimum   E = -130.5053405386   internal+external STABLE
    pi state         E = -130.5003466413   internal UNSTABLE
    pi-state lambda_min = -4.8706729960e-03   (dim 148, |grad| = 9.0e-9)
"""

import numpy as np
from pyscf import gto, scf
from pyscf.soscf import newton_ah

mol = gto.M(
    atom="He 0 0 0; Ne 0 0 2.0",
    basis="def2-svp",
    charge=1,
    spin=1,
    unit="Angstrom",
    verbose=0,
)

# --- 1. default guess: the stable minimum ---
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 500
mf.kernel()
_, _, si, se = mf.stability(internal=True, external=True, return_status=True)
print(
    f"stable minimum   E = {mf.e_tot:.10f}  <S^2> = {mf.spin_square()[0]:.6f}  "
    f"internal_stable={si}  external_stable={se}"
)

# --- 2. the 2-Pi state ferric lands on: move the beta hole into 2p_pi ---
nb = mol.nelec[1]
occ = mf.mo_occ[1].copy()
occ[nb - 2] = 0
occ[nb] = 1
dm = mf.make_rdm1((mf.mo_coeff[0], mf.mo_coeff[1]), (mf.mo_occ[0], occ))
m2 = scf.UHF(mol)
m2.conv_tol = 1e-12
m2.max_cycle = 500
scf.addons.mom_occ(m2, (mf.mo_coeff[0], mf.mo_coeff[1]), (mf.mo_occ[0], occ))
m2.kernel(dm0=dm)
_, _, si2, _ = m2.stability(return_status=True)
print(
    f"pi state         E = {m2.e_tot:.10f}  <S^2> = {m2.spin_square()[0]:.6f}  "
    f"internal_stable={si2}  converged={m2.converged}"
)

# --- 3. PySCF's own orbital Hessian on the pi state ---
out = newton_ah.gen_g_hop_uhf(m2, m2.mo_coeff, m2.mo_occ)
g, h_op, h_diag = out[0], out[-2], out[-1]
n = len(h_diag)
hess = np.zeros((n, n))
for i in range(n):
    e = np.zeros(n)
    e[i] = 1.0
    hess[:, i] = h_op(e)
hess = 0.5 * (hess + hess.T)
w = np.linalg.eigvalsh(hess)
print(f"rotation-space dim = {n}, max |orbital gradient| = {np.abs(g).max():.3e}")
print(f"pi-state lambda_min = {w[0]:+.10e}")
print("lowest few          =", ["%+.4e" % x for x in w[:5]])
