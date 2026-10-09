"""Where does the k-point fitting error live?  H2/STO-3G a=4, cart aux, 1x1x2 and 1x1x3 (complex q), at the
CONVERGED DENSE-AFT density: dE_J (q = 0), dE_K split by q class, first-order (fixed density).
Hypotheses (before the run): H_q - the fit of Bloch pair densities with q != 0 (oscillating e^{iq.r} envelope
across the cell) is what a molecular RI set represents worst, so dE_K(q != 0) dominates and grows vs Gamma;
H_0 - the error is ordinary q = 0 J/K fitting of a different density, so dE_K(q != 0) is a minor share.
Usage: python3 run_kgdf_fiterr_split.py [n1n2n3 ...]"""

import sys

import numpy as np

sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import ferric_basis  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS  # noqa: E402

cell = Cell(H2_A, H2_ATOMS, "sto-3g")
for arg in sys.argv[1:] or ["112", "113"]:
    n = tuple(int(c) for c in arg)
    kb = PK.build_k(cell, n, exxdiv="ewald", verbose=False)
    Nk = kb["Nk"]
    e, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, return_mo=True)
    dm = np.array([2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(Nk)])
    for auxname in ("cc-pvdz-ri", "def2-universal-jkfit"):
        kg = KG.build_kgdf(cell, n, ferric_basis(auxname, ["H"]), spherical=False)
        Jf, Kf = KG.kernels_from_kB(kg)
        dJk = np.einsum("kjmnls,jls->kmn", Jf - kb["Jker"], dm) / Nk
        dEJ = 0.5 * np.einsum("kmn,knm->", dJk, dm).real / Nk
        parts = {}
        for k in range(Nk):
            for j in range(Nk):
                iq = PK.mesh_index(n, kb["ints"][j] - kb["ints"][k])
                dK = np.einsum("mlns,ls->mn", Kf[k, j] - kb["Kker"][k, j], dm[j]) / Nk
                parts[iq] = (
                    parts.get(iq, 0.0)
                    - 0.25 * np.einsum("mn,nm->", dK, dm[k]).real / Nk
                )
        e_fit = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, jk=KG.jk_from_kB(kg))[0]
        print(
            f"{n} {auxname}: SCF fit error {e_fit - e:.3e}; first order at dense dm: dE_J {dEJ:.3e}, dE_K by q "
            f"{ {q: float('%.3e' % v) for q, v in sorted(parts.items())} } sum {dEJ + sum(parts.values()):.3e}",
            flush=True,
        )
