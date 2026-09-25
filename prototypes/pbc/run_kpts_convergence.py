"""Stage 3 convergence: H2/STO-3G cubic a=4 (the Iteration-1 cell), E per cell vs n x n x n mesh, exxdiv none vs ewald.

Pure AFT, gcut from prec 1e-10 (same K sphere at every n, so differences between meshes are pure BZ sampling).
Hypotheses written BEFORE the run:
  H0 (exact identity, not physics): at fixed orbitals v_M S dm S shifts every occupied level by -v_M, so
     E_none(n) - E_ewald(n) = nocc v_M(n) = nocc * 2.8372974795/(n a) exactly (insulator, D unchanged).
     => exxdiv none converges as 1/n (= N_k^-1/3) with a coefficient known in advance.
  H1 (physics, ewald): the Madelung term removes the q^0 part of the missing q=0 head; the leftover is the q^2
     term of the band-summed pair densities, i.e. the gauge-invariant Marzari-Vanderbilt spread Omega_I:
     E_ewald(n) - E_inf ~ -(4pi/3) Omega_I / (n a)^3  (= N_k^-1; the crystal analogue of Iteration 1's
     c3 = -(4pi/3) sigma^2, which it reduces to for a flat band).
  Artifact: a missing/incorrect q-shift of the kernel or a wrong v_M leaves an n^-1 term in ewald (tested as mutants
  in run_kpts_anchor.py); a mis-signed Bloch phase breaks the supercell anchor.
Usage: python3 run_kpts_convergence.py nmax [a] [nmin]   (a=4 is the Iteration-1 cell; a=6 added after the a=4 run showed
even/odd band-sampling oscillation: flatter bands, closer to the H1 regime)"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from run_kpts_anchor import H2_ATOMS  # noqa: E402

if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    cell = Cell(np.eye(3) * a, H2_ATOMS, "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    rows = []
    nmin = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for m in range(nmin, nmax + 1):
        t = time.time()
        kb = PK.build_k(cell, (m, m, m), exxdiv="ewald", gcut=gcut)
        e_ew, eps_ew, _, C, occ = PK.krhf(kb, 2, conv=1e-12, return_mo=True)
        e_no, eps_no, _ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0)
        vm = kb["madelung"]
        oi = PK.omega_I(cell, kb, C, occ) if m >= 3 else float("nan")
        gap = min(min(e[1:]) for e in eps_ew) - max(e[0] for e in eps_ew)
        rows.append((m, e_no, e_ew, vm, oi))
        print(
            f"n={m}: E_none {e_no:.12f} E_ewald {e_ew:.12f} v_M {vm:.10f} (v_M n a {vm * m * a:.10f}) "
            f"E_none-E_ewald-v_M {e_no - e_ew - vm:.1e} Omega_I {oi:.6f} gap {gap:.4f} ({time.time() - t:.0f}s)",
            flush=True,
        )
    print("ROWS", repr(rows))
