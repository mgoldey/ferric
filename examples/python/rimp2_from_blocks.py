"""Closed-shell RI-MP2 written in numpy on ferric's density-fitted MO tensors.

The PySCF ``mp.dfmp2`` pattern: fitted B^P_ia = sum_Q (P|Q)^-1/2 (Q|ia) from
``AoIntegrals.df_mo(aux, C_occ, C_vir)`` (shape (naux, nocc, nvir), the
Coulomb-metric fit ferric's RI-MP2 uses), then

    E_corr = sum_ijab (ia|jb) [2 (ia|jb) - (ib|ja)] / (e_i + e_j - e_a - e_b).

The reference orbitals come from ``rhf_from_blocks.rhf``. All electrons are
correlated (``run_rimp2(frozen_core=0)``). Hartree.

Run as a script: water/cc-pVDZ with cc-pVDZ-RI, asserting E_corr equals
``ferric.run_rimp2`` to 1e-8 Ha (and PySCF ``DFMP2`` on the same auxiliary
basis to the same bar when PySCF is installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import WATER, rhf  # noqa: E402


def rimp2_corr(ints: ferric.AoIntegrals, aux: ferric.BasisSet, mf: dict) -> float:
    """RI-MP2 correlation energy (Hartree) on the canonical RHF orbitals ``mf``."""
    nocc = int(np.count_nonzero(mf["mo_occ"]))
    c, e = mf["mo_coeff"], mf["mo_energy"]
    b = ints.df_mo(aux, c[:, :nocc], c[:, nocc:])
    iajb = np.einsum("pia,pjb->iajb", b, b)
    eo, ev = e[:nocc], e[nocc:]
    denom = (
        eo[:, None, None, None]
        - ev[None, :, None, None]
        + eo[None, None, :, None]
        - ev[None, None, None, :]
    )
    return float(np.sum(iajb * (2.0 * iajb - iajb.transpose(0, 3, 2, 1)) / denom))


def pyscf_dfmp2_corr(xyz: Path, basis: str, auxbasis: str) -> float | None:
    """PySCF's all-electron DF-MP2 correlation energy, or None without PySCF."""
    try:
        from pyscf import gto, scf
        from pyscf.mp import dfmp2
    except ImportError:
        return None
    atoms = xyz.read_text().splitlines()[2:]
    mol = gto.M(atom="\n".join(atoms), basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    pt = dfmp2.DFMP2(mf)
    pt.with_df.auxbasis = auxbasis
    return pt.kernel()[0]


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    ints = ferric.AoIntegrals(mol, bs)
    e_corr = rimp2_corr(ints, aux, rhf(ints))
    ref = ferric.run_rimp2(mol, bs, aux, frozen_core=0).mp2_corr
    print(f"numpy RI-MP2 {e_corr:.10f}")
    print(f"run_rimp2    {ref:.10f}   diff {e_corr - ref:.1e}")
    assert abs(e_corr - ref) < 1e-8
    e_pyscf = pyscf_dfmp2_corr(WATER, "cc-pvdz", "cc-pvdz-ri")
    if e_pyscf is not None:
        print(f"PySCF DFMP2  {e_pyscf:.10f}   diff {e_corr - e_pyscf:.1e}")
        assert abs(e_corr - e_pyscf) < 1e-8


if __name__ == "__main__":
    main()
