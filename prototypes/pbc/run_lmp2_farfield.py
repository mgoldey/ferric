"""Stage 8b: how much of the LMP2 truncation error is the uniform-field (G->0) coupling?

For every occupied pair a truncation drops ENTIRELY (distance cutoff R_c, or the Eq-8 eps mask
with no surviving element), compute the P2 prediction of its pair energy from molecular-type
quantities only: J_ij^unif = -(4 pi/Omega_sc) mu_i (x) mu_j (mu = z transition dipoles of the LMO/VV-HV
basis, from the Resta matrix), amplitude from the full non-canonical Fvv (Sylvester, direct term).
  dE_farfield(N) = - sum_{dropped pairs} e_ij^unif
The sweep's dE minus this is the part of the error NOT explained by the uniform field (near-field
locality loss + amplitude relaxation).  No CG here: combine with run_lmp2_sweep.py's dE.
Usage: python3 run_lmp2_farfield.py N1 N2 ...
"""

import sys

import numpy as np

import pbc_lmp2 as L
from pbc_supercell import Supercell, build_supercell
from run_lmp2_anchor import A0, AUX, H2_ATOMS
from run_lmp2_sweep import EPS, RCUT


def uniform_pair_energies(p, d, vol):
    mu = L.transition_dipoles_z(p, d, axis=2)
    fo = np.diag(p["Foo"])
    lv, Vv = np.linalg.eigh(p["Fvv"])
    m = mu @ Vv  # (nocc, nvir) in the Fvv eigenbasis
    no = len(fo)
    e = np.zeros((no, no))
    for i in range(no):
        for j in range(no):
            P = -(4 * np.pi / vol) * np.outer(m[i], m[j])
            e[i, j] = -2 * np.sum(P**2 / (lv[:, None] + lv[None, :] - fo[i] - fo[j]))
    return e


def run(N):
    sc = Supercell(A0, H2_ATOMS, "6-31g", (1, 1, N))
    d = build_supercell(sc, AUX, w=0.5)
    scf = L.gamma_scf(d, 2 * N)
    p = L.prepare(sc, d, scf)
    eu = uniform_pair_energies(p, d, sc.sc.vol)
    dist = L.pair_distances(p)
    J = p["J"]
    off = ~np.eye(N, dtype=bool)
    line = [f"N={N:3d}"]
    for eps in EPS:
        kept = ((np.abs(J) > eps) | (np.abs(J.transpose(0, 3, 2, 1)) > eps)).any(
            axis=(1, 3)
        )
        line.append(f"eps{eps:g}:{-eu[~kept & off].sum():+.4e}")
    for rc in RCUT:
        line.append(f"Rc{rc:g}:{-eu[(dist > rc) & off].sum():+.4e}")
    line.append(f"all-offdiag:{-eu[off].sum():+.4e}")
    print("  " + "  ".join(line), flush=True)


if __name__ == "__main__":
    for N in [int(x) for x in sys.argv[1:]] or [8, 12, 16, 24, 32]:
        run(N)
