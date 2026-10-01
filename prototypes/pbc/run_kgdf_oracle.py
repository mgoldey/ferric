"""pbc_kgdf vs PySCF 2.13 KRHF with GDF and RSDF on the SAME aux (cart, as the cart cell forces), and the fitting
error of both vs our dense k-point AFT (pbc_kpts.build_k at its default converged cutoff, which agreed with PySCF
AFTDF to 1e-12..2e-14 in Iteration 9).  PySCF is started from OUR dense-AFT density (same SCF state).

Predictions (before the run): PHYSICS - ours == PySCF RSDF to ~1e-10 or better (same scheme, Iteration 2 got 6e-13
at Gamma), PySCF GDF == RSDF to ~1e-11; fitting error |dE| ~1e-6..1e-4 (the Gamma sizes), the same for none and
ewald (v_M S dm S is fit-independent), and of the Gamma magnitude per cell (the fit is per-q, nothing k-specific
makes it worse).  ARTIFACT - a k-specific construction error would make ours-vs-PySCF grow with Nk / be absent at
the TRIM-only 1x1x2 mesh (all q real there) but present at a mesh with complex q (none in this set: 2x2x2 and
1x1x2 are all-TRIM, so the complex-q phases are covered only by the 1x1x3 anchors).
Usage: python3 run_kgdf_oracle.py h2|tri n1n2n3 [aux ...]
Counts are printed from build info (no timings claimed)."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import ferric_basis  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS, SP, TRI_A, TRI_ATOMS  # noqa: E402
from pyscf.pbc import df as pdf  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import scf as pscf  # noqa: E402

SYS = {"h2": (H2_A, H2_ATOMS, "sto-3g", 2), "tri": (TRI_A, TRI_ATOMS, SP, 4)}

if __name__ == "__main__":
    name, n = sys.argv[1], tuple(int(c) for c in sys.argv[2])
    auxes = sys.argv[3:] or ["cc-pvdz-ri", "def2-universal-jkfit"]
    a, atoms, basis, nelec = SYS[name]
    cell = Cell(a, atoms, basis)
    t = time.time()
    kb = PK.build_k(cell, n, exxdiv="ewald")
    print(f"  dense build {time.time() - t:.0f}s", flush=True)
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.max_memory = int(os.environ.get("PYSCF_MAXMEM", "1500"))
    pc.build()
    kp = pc.make_kpts(list(n))
    ed, dm0 = {}, {}
    for ex in ("none", "ewald"):
        e, eps, it, C, occ = PK.krhf(
            kb,
            nelec,
            conv=1e-12,
            kshift=kb["madelung"] if ex == "ewald" else 0.0,
            return_mo=True,
        )
        ed[ex] = e
        dm0[ex] = np.array(
            [2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(kb["Nk"])]
        )
        print(f"{name} {n} dense k-AFT {ex}: {e:.12f}", flush=True)
    for auxname in auxes:
        aux = ferric_basis(auxname, [s for s, _ in atoms])
        t = time.time()
        kg = KG.build_kgdf(cell, n, aux, spherical=False)
        inf = kg["info"]
        nao = cell.mol.nao
        print(
            f"  kgdf {auxname}: {time.time() - t:.0f}s; naux(cart) {inf['naux']}, kept per q "
            f"{[inf['per_q'][i]['kept'] for i in range(kg['Nk'])]}, nK per q {[inf['per_q'][i]['nK'] for i in range(kg['Nk'])]}, "
            f"q classes built {inf['n_q_built']}/{kg['Nk']}; SR 3c shell triplets {inf['n_3c_shell_triplets']} "
            f"(= the Gamma set, computed once for all q); pair images {inf['n_pair_images']}, aux images 3c/2c "
            f"{inf['n_aux_images_3c']}/{inf['n_aux_images_2c']}; B complex elements {inf['B_elems']} "
            f"({inf['B_elems'] // kg['Nk'] ** 2} per (k,k') = kept x nao^2, nao {nao}); metric smin/q "
            f"{['%.1e' % inf['per_q'][i]['smin'] for i in range(kg['Nk'])]}",
            flush=True,
        )
        for ex in ("none", "ewald"):
            vm = kb["madelung"] if ex == "ewald" else 0.0
            ek = PK.krhf(kb, nelec, conv=1e-12, kshift=vm, jk=KG.jk_from_kB(kg))[0]
            res = {}
            for cls in ("GDF", "RSDF"):
                mf = pscf.KRHF(pc, kp, exxdiv=None if ex == "none" else "ewald")
                mf.with_df = getattr(pdf, cls)(pc, kp)
                mf.with_df.auxbasis = aux
                mf.conv_tol = 1e-11
                t = time.time()
                res[cls] = mf.kernel(dm0=dm0[ex])
                res[cls + "_t"] = time.time() - t
            print(
                f"{name} {n} {auxname} {ex}: ours {ek:.12f} | PySCF RSDF {res['RSDF']:.12f} GDF {res['GDF']:.12f} | "
                f"ours-RSDF {ek - res['RSDF']:.1e} ours-GDF {ek - res['GDF']:.1e} | fit error vs dense: ours "
                f"{ek - ed[ex]:.3e} RSDF {res['RSDF'] - ed[ex]:.3e} GDF {res['GDF'] - ed[ex]:.3e}",
                flush=True,
            )
