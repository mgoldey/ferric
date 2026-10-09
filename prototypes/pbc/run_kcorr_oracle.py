"""Stage 9 oracle: pbc_kcorr KMP2 vs PySCF 2.13 pbc.mp.KMP2 on (i) KRHF + AFTDF (mesh 61^3) against our dense
k-point AFT route, and (ii) KRHF + GDF (cart cc-pvdz-ri) against our k-point RS-GDF B route (pbc_kgdf).  H2/STO-3G
a=4, meshes 1x1x2 (TRIM) and 1x1x3 (complex q); PySCF started from our density matrix, conv 1e-11.

Predictions (before the run; PySCF source read: kmp2.py takes mf.mo_energy, has no Madelung code):
  PySCF KMP2 on an exxdiv=None KRHF == our 'unshifted', on an exxdiv='ewald' KRHF == our 'shifted' (mesh v_M),
  to the SCF agreement level (~1e-13 H2 in Iteration 9) for AFTDF and to ~1e-12 for GDF (ours == PySCF GDF to
  <= 7e-12 in HF, Iteration 11).  ARTIFACT: a wrong 1/Nk power would give a ratio of 2 or 3 at 1x1x2 / 1x1x3 (not a
  small difference); a conj error would appear only at 1x1x3.  PySCF has no periodic RPA (Iteration 4), so k-dRPA is
  printed for the record only (its checks are the supercell anchor and the O(Pi^2) == direct KMP2 identity).
Usage: python3 run_kcorr_oracle.py [aft|gdf] n1n2n3 ..."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kcorr as KC  # noqa: E402
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import ferric_basis  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS  # noqa: E402
from pyscf.pbc import df as pdf  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import mp as pmp  # noqa: E402
from pyscf.pbc import scf as pscf  # noqa: E402

if __name__ == "__main__":
    route = sys.argv[1]
    meshes = [tuple(int(c) for c in m) for m in (sys.argv[2:] or ["112", "113"])]
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    pc = pgto.Cell(
        a=H2_A, atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0
    )
    pc.precision = 1e-12
    pc.max_memory = int(os.environ.get("PYSCF_MAXMEM", "1500"))
    pc.build()
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    for n in meshes:
        kb = PK.build_k(cell, n, exxdiv="ewald")
        kg = KG.build_kgdf(cell, n, aux, spherical=False) if route == "gdf" else None
        jk = KG.jk_from_kB(kg) if kg else None
        e, eps, it, C, occ = PK.krhf(
            kb, 2, conv=1e-12, kshift=0.0, return_mo=True, jk=jk
        )
        st = KC.kB_ov(kg, C, 1) if kg else KC.aft_ov(cell, n, C, 1)
        dm = np.array(
            [2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(kb["Nk"])]
        )
        kp = pc.make_kpts(list(n))
        for ex, cv in ((None, "unshifted"), ("ewald", "shifted")):
            eo, ev = KC.k_denominators(eps, 1, kb["madelung"], cv)
            ours = KC.kmp2(st, eo, ev)
            rpa = KC.kdrpa_plasmon(st, eo, ev)
            t = time.time()
            mf = pscf.KRHF(pc, kp, exxdiv=ex)
            if route == "aft":
                mf.with_df = pdf.AFTDF(pc, kp)
                mf.with_df.mesh = [61] * 3
            else:
                mf.with_df = pdf.GDF(pc, kp)
                mf.with_df.auxbasis = aux
            mf.conv_tol = 1e-11
            ehf = mf.kernel(dm0=dm)
            ep = pmp.KMP2(mf).kernel()[0]
            other = KC.kmp2(
                st,
                *KC.k_denominators(
                    eps,
                    1,
                    kb["madelung"],
                    "shifted" if cv == "unshifted" else "unshifted",
                ),
            )[0]
            print(
                f"{route} {n} exxdiv={ex}: HF ours {e:.12f} (none) pyscf {ehf:.12f} | KMP2 ours {cv} {ours[0]:.13e} "
                f"(os {ours[1]:.6e} ss {ours[2]:.6e}) PySCF {ep:.13e} d {ours[0] - ep:.1e} | ours other conv - PySCF "
                f"{other - ep:.1e} | k-dRPA {cv} {rpa:.13e} ({time.time() - t:.0f}s)",
                flush=True,
            )
