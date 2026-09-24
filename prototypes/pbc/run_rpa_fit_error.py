"""Stage 7 aux-fitting error of Gamma dRPA: RS-GDF B (ferric bundled aux, spherical, w=1) vs the exact
dense pure-AFT (ia|jb) (plasmon), same cell/basis, both denominator conventions.
  'same C'  : fitted B with the EXACT SCF orbitals/eps  (pure dRPA fitting error)
  'own SCF' : SCF also run on the fitted B (the full pipeline a Rust port would run)
Also the MP2 fit error on the same B (to compare aux sensitivity) and, for H2, the molecular DF-dRPA
error with the same (spherical) aux on the same molecule."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto, scf, df
from pbc_gamma import Cell, build_integrals, rhf
from pbc_gdf import build_gdf, jk_from_B
from pbc_mp2 import gamma_mp2, denominators
from pbc_rpa import gamma_drpa, drpa_plasmon, drpa_quad
from run_mp2_fit_error import SYS, AUX


def mol_df_error(atoms, basis, auxbasis):
    mol = gto.M(atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    nocc = mol.nelectron // 2
    Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
    ov = mol.ao2mo((Co, Cv, Co, Cv), compact=False).reshape(
        nocc, Cv.shape[1], nocc, Cv.shape[1]
    )
    ex = drpa_plasmon(ov, eo, ev)
    aux = gto.M(
        atom=atoms, basis=auxbasis, unit="B", cart=True, verbose=0
    )  # cart ints, then -> spherical
    sph = aux.copy()
    sph.cart = False
    c2s = sph.cart2sph_coeff()
    j3 = df.incore.aux_e2(mol, aux) @ c2s  # (nao, nao, naux_sph)
    s, U = np.linalg.eigh(c2s.T @ aux.intor("int2c2e_cart") @ c2s)
    k = s > 1e-10
    Bia = np.einsum("mnP,Pk,mi,na->kia", j3, U[:, k] / np.sqrt(s[k]), Co, Cv)
    return drpa_quad(Bia, eo, ev) - ex, ex


if __name__ == "__main__":
    for name in sys.argv[1:] or SYS:
        a, atoms, basis = SYS[name]
        nel = len(atoms)
        nocc = nel // 2
        cell = Cell(a, atoms, basis)
        ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        vm = ints["madelung"]
        e0, eps0, _, C0 = rhf(
            ints["S"],
            ints["h"],
            ints["I"],
            ints["enn"],
            nel,
            conv=1e-12,
            return_mo=True,
        )
        den = {
            c: np.concatenate(denominators(eps0, nocc, vm, c))
            for c in ("shifted", "unshifted")
        }
        exact = {
            c: gamma_drpa(C0, den[c], nocc, eri=ints["I"], method="plasmon")
            for c in den
        }
        exmp2 = gamma_mp2(C0, den["shifted"], nocc, eri=ints["I"])[0]
        print(
            f"{name}: exact dRPA shifted {exact['shifted']:.10e} unshifted {exact['unshifted']:.10e}  (MP2 shifted {exmp2:.10e})",
            flush=True,
        )
        for an, fn in AUX.items():
            t = time.time()
            g = build_gdf(cell, fn())
            B = g["B"]
            e1, eps1, _, C1 = rhf(
                ints["S"],
                ints["h"],
                None,
                ints["enn"],
                nel,
                conv=1e-12,
                jk=jk_from_B(B),
                return_mo=True,
            )
            row = []
            for c in ("shifted", "unshifted"):
                same = gamma_drpa(C0, den[c], nocc, B=B) - exact[c]
                own = (
                    gamma_drpa(
                        C1, np.concatenate(denominators(eps1, nocc, vm, c)), nocc, B=B
                    )
                    - exact[c]
                )
                row.append(
                    f"{c}: same-C {same:+.3e} own-SCF {own:+.3e} (rel {same / exact[c]:+.1e})"
                )
            m = gamma_mp2(C0, den["shifted"], nocc, B=B)[0] - exmp2
            molerr = ""
            if name.startswith("H2"):
                d, ex = mol_df_error(atoms, basis, fn())
                molerr = f"  molecular DF-dRPA {d:+.3e} (rel {d / ex:+.1e})"
            print(
                f"  {an:22s} naux {g['info']['naux_kept']}/{g['info']['naux']}  dE_HF {e1 - e0:+.3e}  "
                + "  ".join(row)
                + f"  | MP2 same-C {m:+.3e} (rel {m / exmp2:+.1e}){molerr}  ({time.time() - t:.0f}s)",
                flush=True,
            )
