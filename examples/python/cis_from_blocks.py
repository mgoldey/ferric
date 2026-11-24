"""Singlet CIS (TDA on an RHF reference) written in numpy on ferric's MO integrals.

The PySCF ``tdscf.TDA`` pattern: build and diagonalize the singlet A matrix

    A_ia,jb = delta_ij delta_ab (e_a - e_i) + 2 (ia|jb) - (ij|ab).

Two integral sources, both from ``ferric.AoIntegrals``:

* ``df_mo(aux, C1, C2)`` -- density-fitted B^P_pq, the same Coulomb-metric fit
  ``ferric.run_tddft`` builds its response integrals from;
* ``ao2mo(C1, C2, C3, C4)`` -- exact (pq|rs), integral-direct.

Run as a script: water/cc-pVDZ, asserting the DF roots equal
``ferric.run_tddft(functional=None, method="tda")`` to 1e-8 Ha, and the exact
roots equal PySCF ``tdscf.TDA`` to the same bar when PySCF is installed.
Hartree.

``run_tddft`` converges its own RHF at ``run_rhf``'s default thresholds, and
CIS roots move to first order with the orbitals (5e-8 Ha between that SCF and
a 1e-11 one on water/cc-pVDZ). The driver comparison therefore takes its
orbitals from ``run_rhf`` at the same defaults (``orbitals_of``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import WATER, rhf  # noqa: E402


def _a_matrix(e_occ, e_vir, iajb, ijab):
    nocc, nvir = len(e_occ), len(e_vir)
    a = 2.0 * iajb - ijab.transpose(0, 2, 1, 3)
    a = a.reshape(nocc * nvir, nocc * nvir)
    a[np.diag_indices_from(a)] += (e_vir[None, :] - e_occ[:, None]).ravel()
    return a


def cis_singlets(
    ints: ferric.AoIntegrals,
    mf: dict,
    nroots: int = 3,
    aux: ferric.BasisSet | None = None,
) -> np.ndarray:
    """Lowest ``nroots`` singlet CIS excitation energies (Hartree), ascending.

    ``aux`` selects density-fitted integrals over that auxiliary basis;
    ``None`` uses exact integrals.
    """
    nocc = int(np.count_nonzero(mf["mo_occ"]))
    c, e = mf["mo_coeff"], mf["mo_energy"]
    co, cv = c[:, :nocc], c[:, nocc:]
    if aux is None:
        iajb = ints.ao2mo(co, cv, co, cv)
        ijab = ints.ao2mo(co, co, cv, cv)
    else:
        b_ov, b_oo, b_vv = (
            ints.df_mo(aux, x, y) for x, y in ((co, cv), (co, co), (cv, cv))
        )
        iajb = np.einsum("pia,pjb->iajb", b_ov, b_ov)
        ijab = np.einsum("pij,pab->ijab", b_oo, b_vv)
    a = _a_matrix(e[:nocc], e[nocc:], iajb, ijab)
    return np.linalg.eigvalsh(a)[:nroots]


def orbitals_of(result, nocc: int) -> dict:
    """The ``mo_coeff`` / ``mo_energy`` / ``mo_occ`` dict of an ``RhfResult``."""
    e = np.asarray(result.orbital_energies())
    occ = np.zeros(len(e))
    occ[:nocc] = 2.0
    return {
        "mo_coeff": np.asarray(result.mo_coefficients()),
        "mo_energy": e,
        "mo_occ": occ,
    }


def pyscf_cis(xyz: Path, basis: str, nroots: int) -> np.ndarray | None:
    """PySCF's exact-integral singlet TDA roots on RHF, or None without PySCF."""
    try:
        from pyscf import gto, scf, tdscf
    except ImportError:
        return None
    atoms = xyz.read_text().splitlines()[2:]
    mol = gto.M(atom="\n".join(atoms), basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    td = tdscf.TDA(mf)
    td.nstates = nroots
    td.conv_tol = 1e-12
    return np.sort(td.kernel()[0])


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    ints = ferric.AoIntegrals(mol, bs)
    w_df = cis_singlets(
        ints, orbitals_of(ferric.run_rhf(mol, bs), ints.nelec // 2), 3, aux
    )
    ref = np.asarray(
        ferric.run_tddft(
            mol, bs, aux, functional=None, n_roots=3, method="tda"
        ).excitation_energies
    )
    print("numpy CIS (DF)  ", np.array2string(w_df, precision=10))
    print("run_tddft (TDA) ", np.array2string(ref, precision=10))
    assert np.abs(w_df - ref).max() < 1e-8
    w_pyscf = pyscf_cis(WATER, "cc-pvdz", 3)
    if w_pyscf is not None:
        w_exact = cis_singlets(ints, rhf(ints), 3)
        print("numpy CIS (exact)", np.array2string(w_exact, precision=10))
        print("PySCF TDA        ", np.array2string(w_pyscf, precision=10))
        assert np.abs(w_exact - w_pyscf).max() < 1e-8


if __name__ == "__main__":
    main()
