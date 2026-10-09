"""Stage 3 oracle: pbc_kpts k-point RHF vs PySCF 2.13 pbc.scf.KRHF + AFTDF (mesh 61^3), H2/STO-3G a=4 and the
triclinic 4H s+p cell, meshes 1x1x2 and 2x2x2, exxdiv None and 'ewald'.  Ours is built once per (cell, mesh) at
the default pure-AFT cutoff (prec 1e-14) and run with kshift 0 / v_M; PySCF is started from our density matrix
(so both land on the same SCF state) and converged to 1e-11.  Usage: python3 run_kpts_oracle.py [h2|tri] [n1n2n3 ...]"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pbc_kpts import build_k, krhf  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS, SP, TRI_A, TRI_ATOMS  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pyscf.pbc import df as pdf  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import scf as pscf  # noqa: E402

import os  # noqa: E402

PYMESH = int(
    os.environ.get("PYSCF_MESH", "61")
)  # tri 2x2x2 at 61^3 is hours; the mesh error is calibrated at 1x1x2
SYS = {"h2": (H2_A, H2_ATOMS, "sto-3g", 2), "tri": (TRI_A, TRI_ATOMS, SP, 4)}

if __name__ == "__main__":
    name = sys.argv[1]
    meshes = [tuple(int(c) for c in m) for m in (sys.argv[2:] or ["112", "222"])]
    a, atoms, basis, nelec = SYS[name]
    cell = Cell(a, atoms, basis)
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.max_memory = int(os.environ.get("PYSCF_MAXMEM", "1200"))
    pc.build()
    for n in meshes:
        kb = build_k(cell, n, exxdiv="ewald")
        kp = pc.make_kpts(list(n))
        for ex in (None, "ewald"):
            e, eps, it, C, occ = krhf(
                kb,
                nelec,
                conv=1e-12,
                kshift=kb["madelung"] if ex else 0.0,
                return_mo=True,
            )
            dm = np.array(
                [
                    2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T
                    for k in range(kb["Nk"])
                ]
            )
            print(f"{name} {n} exxdiv={ex}: ours {e:.12f} (it {it})", flush=True)
            t = time.time()
            mf = pscf.KRHF(pc, kp, exxdiv=ex)
            mf.with_df = pdf.AFTDF(pc, kp)
            mf.with_df.mesh = [PYMESH] * 3
            mf.conv_tol = 1e-11
            ep = mf.kernel(dm0=dm)
            de = max(
                abs(np.asarray(eps[k]) - mf.mo_energy[k]).max() for k in range(kb["Nk"])
            )
            print(
                f"{name} {n} exxdiv={ex}: ours {e:.12f} pyscf {ep:.12f} dE {e - ep:.1e} max|d eps| {de:.1e} "
                f"(pyscf mesh {PYMESH}^3 {time.time() - t:.0f}s, conv {mf.converged}); nocc per k {[int(o.sum()) for o in occ]}",
                flush=True,
            )
