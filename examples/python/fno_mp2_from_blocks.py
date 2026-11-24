"""Frozen natural orbitals from DF-MP2, written on ferric's blocks.

Frozen natural orbitals (Taube, Bartlett, Collect. Czech. Chem. Commun. 70,
837 (2005)) in the form of PySCF's ``mp.MP2.make_fno``. The unrelaxed
DF-MP2 virtual-virtual density

    D_ab = 2 sum_ijc [2 t_ij^ca t_ij^cb - t_ij^ca t_ij^bc],
    t_ij^ab = (ia|jb) / (e_i + e_j - e_a - e_b),

is diagonalized; its eigenvalues are the virtual natural occupations. The
``nvir_keep`` most occupied natural orbitals are kept, semicanonicalized (the
virtual Fock block diagonalized inside them), and MP2 is rerun in the
truncated space. The fitted integrals come from ``AoIntegrals.df_mo`` and
every contraction from ``ferric.einsum`` on ferric's GEMM engine. The RHF
orbitals and energies come from ``run_rhf``'s result. All electrons are
correlated. Hartree.

With every virtual kept, the rotation is a unitary change of the virtual
basis with a diagonal Fock block, and the energy equals ``run_rimp2``: the
exactness anchor.

Run as a script: water/cc-pVDZ with cc-pVDZ-RI, asserting the all-virtual
limit equals ``run_rimp2`` to 1e-10 Ha and, when the PySCF reference exists,
the occupations and the truncated energy equal PySCF's ``DFMP2.make_fno`` to
1e-10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_set2_pyscf.json"

EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def amplitudes(ints, aux, c_occ, c_vir, e_occ, e_vir):
    """(t_ij^ab as t[i, a, j, b], (ia|jb) as the same layout) from df_mo."""
    b = ints.df_mo(aux, c_occ, c_vir)
    iajb = ferric.einsum("pia,pjb->iajb", b, b)
    eia = e_occ[:, None] - e_vir[None, :]
    return iajb / (eia[:, :, None, None] + eia[None, None, :, :]), iajb


def mp2_energy(t: np.ndarray, iajb: np.ndarray) -> float:
    """sum_ijab t_ij^ab [2 (ia|jb) - (ib|ja)]."""
    return ferric.einsum("iajb,iajb->", t, 2.0 * iajb - iajb.transpose(0, 3, 2, 1))


def fno_mp2(ints, aux, mo_coeff, mo_energy, nocc: int, nvir_keep: int | None = None):
    """(natural occupations, semicanonical FNO coefficients, E_corr in the FNOs)."""
    co, cv = mo_coeff[:, :nocc], mo_coeff[:, nocc:]
    eo, ev = mo_energy[:nocc], mo_energy[nocc:]
    t, _ = amplitudes(ints, aux, co, cv, eo, ev)
    # t[i, c, j, a] = t_ij^ca; contract i, c, j in both terms.
    direct = ferric.einsum("icja,icjb->ab", t, t)
    exchange = ferric.einsum("icja,ibjc->ab", t, t)
    occ, v = np.linalg.eigh(2.0 * (2.0 * direct - exchange))
    occ, v = occ[::-1], v[:, ::-1]
    keep = len(occ) if nvir_keep is None else nvir_keep
    _, rot = np.linalg.eigh((v[:, :keep].T * ev) @ v[:, :keep])
    v_keep = v[:, :keep] @ rot
    e_keep = np.einsum("ap,a,ap->p", v_keep, ev, v_keep)
    t, iajb = amplitudes(ints, aux, co, cv @ v_keep, eo, e_keep)
    return occ, np.hstack([co, cv @ v_keep]), mp2_energy(t, iajb)


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    ints = ferric.AoIntegrals(mol, bs)
    scf = ferric.run_rhf(mol, bs, **EXACT_JK, **TIGHT)
    c, e, nocc = scf.mo_coefficients(), scf.orbital_energies(), ints.nelec // 2
    occ, _, e_all = fno_mp2(ints, aux, c, e, nocc)
    ref = ferric.run_rimp2(
        mol, bs, aux, frozen_core=0, scf_energy_conv=1e-11, scf_density_conv=1e-9
    ).mp2_corr
    print(f"virtual NO occupations {np.array2string(occ[:6], precision=6)} ...")
    print(f"all virtuals  {e_all:.12f}   run_rimp2 diff {e_all - ref:.1e}")
    assert abs(e_all - ref) < 1e-10
    _, _, e_half = fno_mp2(ints, aux, c, e, nocc, nvir_keep=len(occ) // 2)
    print(f"half the NOs  {e_half:.12f}   ({e_half / e_all:.4%} of E_corr)")
    if PYSCF_REF.exists():
        ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["fno_mp2"]
        d_occ = np.abs(occ - np.asarray(ref["occupations"])).max()
        d_e = e_half - ref["e_corr_keep"][str(len(occ) // 2)]
        print(f"PySCF occupations max diff {d_occ:.1e}, truncated E diff {d_e:.1e}")
        assert d_occ < 1e-10 and abs(d_e) < 1e-10


if __name__ == "__main__":
    main()
