"""Iteration 18 PySCF oracle for Gamma RS-GDF forces.  Usage: python3 run_grad_gdf_oracle.py {h2|h3|tri}

(1) Does PySCF 2.13 give GDF/RSDF analytic forces for an all-electron cell at Gamma?  Try pbc.scf.RHF/KRHF(+GDF/RSDF)
    .nuc_grad_method().kernel() and report what happens.
(2) Independent construction for the fitted two-electron term: at a FIXED density D (ours, converged), PySCF's own
    RSDF and GDF (same aux, Cartesian, exxdiv=None) give E2(R) = 1/2 tr(D J) - 1/2 sum_s tr(D_s K_s); central FD
    (h = 1e-4) of that vs our analytic fixed-D 2e force = the sum of the J3_* / J2_* / J3_g0 parts of pbc_grad_gdf.
    PySCF's GDF evaluates the same primed metric by compensated charges and RSDF by its own lattice sums and G=0
    handling, so agreement tests every fitted-2e derivative term against code we did not write.
"""

import os
import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

import pbc_grad as PGd
import pbc_grad_gdf as GG
from pbc_gamma import Cell
from pbc_gdf import ferric_basis
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H = 1e-4
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))]
SYS = {
    "h2": (np.eye(3) * 4.0, H2_ATOMS, "sto-3g", 1, 1, True, 12.0, [(0, 0), (0, 2), (1, 1)]),
    "h3": (np.eye(3) * 4.5, H2_ATOMS + [("H", (1.6, 0.9, 0.7))], "sto-3g", 2, 1, False, 12.0, [(0, 0), (2, 1)]),
    "tri": (TRI_A, TRI_MOVED, SP_BASIS, 3, 1, False, 10.0, [(2, 1)]),
}
TWO_E = ("J3_g0", "J3_bra_sr", "J3_bra_lr", "J3_aux_sr", "J3_aux_lr", "J2_sr", "J2_lr")


def pcell(a, atoms, basis, spin=0):
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=spin)
    pc.precision = 1e-12
    pc.max_memory = int(os.environ.get("PYSCF_MAXMEM", "1500"))
    return pc.build()


def e2_pyscf(a, atoms, basis, aux, Da, Db, cls):
    pc = pcell(a, atoms, basis)
    d = getattr(pdf, cls)(pc)
    d.auxbasis = aux
    vj, vk = d.get_jk(np.array([Da, Db]), hermi=1, kpts=np.zeros(3), exxdiv=None)
    D = Da + Db
    return 0.5 * np.sum(D * (vj[0] + vj[1])) - 0.5 * (np.sum(Da * vk[0]) + np.sum(Db * vk[1]))


def main(which):
    a, atoms, basis, na, nb, rst, gcut, comps = SYS[which]
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    # ---- (1) PySCF analytic GDF forces for an all-electron cell?
    pc = pcell(a, atoms, basis, spin=na - nb)
    kinds = ("RHF", "KRHF") if rst else ("UHF", "KUHF")
    for cls in ("GDF", "RSDF"):
        for kind in kinds:
            try:
                k1 = pc.make_kpts([1, 1, 1])
                mf = getattr(pscf, kind)(pc, exxdiv=None) if kind[0] != "K" else getattr(pscf, kind)(pc, k1, exxdiv=None)
                mf.with_df = getattr(pdf, cls)(pc) if kind[0] != "K" else getattr(pdf, cls)(pc, k1)
                mf.with_df.auxbasis = aux
                mf.conv_tol = 1e-10
                mf.kernel()
                g = mf.nuc_grad_method().kernel()
                print(f"  PySCF {kind}+{cls} nuc_grad: RAN, grad {np.asarray(g).ravel()[:3]}")
            except Exception as ex:  # noqa: BLE001 -- reporting what PySCF does
                print(f"  PySCF {kind}+{cls} nuc_grad: {type(ex).__name__}: {str(ex)[:120]}")
    # ---- (2) fixed-D fitted 2e force: ours analytic vs FD of PySCF's fitted E2
    cell = Cell(a, atoms, basis)
    t = time.time()
    auxmol = GG.make_auxmol(cell, aux)
    r = GG.run(cell, auxmol, na, nb, restricted=rst, gcut=gcut, spherical=False)
    Da, Db = r["scf"]["Da"], r["scf"]["Db"]
    g2 = sum(r["parts"][k] for k in TWO_E)
    print(f"  ours: E {r['e']:.12f}, naux(cart) {r['diag']['naux']}, drop {r['diag']['n_drop']}, metric min "
          f"{r['diag']['s_min']:.2e} ({time.time() - t:.0f}s)", flush=True)
    B = r["gd"]["B"]
    D = Da + Db
    Jd = np.einsum("Pmn,P->mn", B, np.einsum("Pls,ls->P", B, D))
    e2_ours = 0.5 * np.sum(D * Jd) - 0.5 * sum(
        np.sum(Ds * np.einsum("Pms,Psn->mn", np.einsum("Pml,ls->Pms", B, Ds), B)) for Ds in (Da, Db))
    for cls in ("RSDF", "GDF"):
        t = time.time()
        e0 = e2_pyscf(a, atoms, basis, aux, Da, Db, cls)
        print(f"  {cls}: E2(D) ours - PySCF {e2_ours - e0:+.2e}", flush=True)
        for A, x in comps:
            ep = e2_pyscf(a, PGd.displaced(cell, A, x, H).atoms, basis, aux, Da, Db, cls)
            em = e2_pyscf(a, PGd.displaced(cell, A, x, -H).atoms, basis, aux, Da, Db, cls)
            fd = (ep - em) / (2 * H)
            print(f"    ({A},{x}) analytic fixed-D 2e {g2[A, x]:+.10f}  FD(PySCF {cls}) {fd:+.10f}  diff "
                  f"{g2[A, x] - fd:+.2e} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h2")
