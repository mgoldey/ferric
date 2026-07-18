"""
S2 spike (open triage item #2): independent cross-check of ferric's
short-range erfc-attenuated RI-MP2 (`attenuated_ri_mp2` /
`crates/ferric-mp2/src/attenuated.rs`) at production omega.

Only limit-behavior tests existed before this spike (SR < full,
omega -> 0 -> full). This script builds a completely independent
reference via PySCF/libcint using `mol.set_range_coulomb(-omega)`
(negative omega => erfc(omega r12)/r12, PySCF's own documented sign
convention -- see `gto.Mole.set_range_coulomb` docstring), which
range-separates *every* subsequent 2-electron integral evaluation
(including the 3-center and 2-center density-fitting integrals used
here). This is a different integral library (libcint vs ferric's
libint2 shim) computing the same physical operator, so it is a real
external cross-check, not an internal consistency check.

RI-MP2 formula replicated exactly from
`crates/ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` /
`spin_components_from_b_ov`:
    V(P|Q)   = 2-center metric under the SAME attenuated operator
    (P|mu nu) = 3-center integral under the SAME attenuated operator
    B_ov[P,ia] = L^{-1} (P|ia)      where V = L L^T (Cholesky; erfc
                                     branch of ferric's
                                     metric_inverse_sqrt, NOT eigh)
    (ia|jb)   = sum_P B_ov[P,ia] B_ov[P,jb]
    e_os = sum_ijab (ia|jb)^2 / D_ijab
    e_ss = sum_ijab (ia|jb)(ia|jb - ib|ja) / D_ijab
    D_ijab = eps_i + eps_j - eps_a - eps_b

Geometry: identical to `attenuated.rs`'s own `setup_h2o()` test fixture
(O 0 0 0.118 / H 0 0.755 -0.471 / H 0 -0.755 -0.471, already in Bohr in
the Rust literal but parsed there via `Molecule::parse_xyz` -- check:
that literal is actually in Angstrom-like magnitudes (O-H ~0.96 A), so
it is intepreted by ferric's parse_xyz in Angstrom same as
testdata/molecules/water.xyz convention). We feed the same numbers to
PySCF's `gto.M(atom=...)` which defaults to Angstrom -- matching units.

omega: production default from `AttenuatedMp2Config::default()` =
0.420 Angstrom^-1 converted to Bohr^-1 via
`BOHR_INV_PER_ANG_INV = 1/1.8897259886` (attenuated.rs:142-147).
PySCF's `set_range_coulomb` omega is also in Bohr^-1 (atomic units
throughout libcint), so we use the exact same converted value -- no
independent re-derivation of the conversion factor.
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import cholesky

sys.path.insert(0, "/home/matt/qc/pyscf")
from pyscf import df, gto, scf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "testdata/reference")
os.makedirs(OUT_DIR, exist_ok=True)

# Same conversion ferric uses (attenuated.rs BOHR_INV_PER_ANG_INV).
BOHR_INV_PER_ANG_INV = 1.0 / 1.8897259886
OMEGA_ANG_INV = 0.420
OMEGA_BOHR_INV = OMEGA_ANG_INV * BOHR_INV_PER_ANG_INV

WATER_XYZ = """
O 0.000 0.000 0.118
H 0.000 0.755 -0.471
H 0.000 -0.755 -0.471
"""


def ri_mp2_spin_components(mol_geom, basis, auxbasis, omega_bohr_inv, frozen_core=0):
    """Replicates ferric's ri_mp2_spin_components exactly, under an
    erfc(omega*r12)/r12 operator applied to BOTH the 2-center metric and
    the 3-center integrals (mol.set_range_coulomb affects all int2e-family
    calls process-wide for this mol object, matching ferric's op-threaded
    Operator::erfc(omega) used for the metric AND the 3-index build)."""
    mol = gto.M(atom=mol_geom, basis=basis, verbose=0, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged
    e_rhf = mf.e_tot

    nocc_total = mol.nelectron // 2
    nbas = mol.nao
    nvir = nbas - nocc_total
    nocc = nocc_total - frozen_core
    first_occ = frozen_core

    eps = mf.mo_energy
    c = mf.mo_coeff
    c_occ = c[:, first_occ:first_occ + nocc]
    c_vir = c[:, nocc_total:]

    # Switch on the short-range erfc operator for ALL subsequent 2e
    # integrals (negative omega => erfc branch, per
    # gto.Mole.set_range_coulomb's own docstring). NOTE: mol and auxmol
    # carry SEPARATE ._env arrays (make_auxmol does not share mol's env),
    # so the omega switch must be applied to BOTH objects independently:
    # df.incore.aux_e2 (3-center) concatenates mol._env+auxmol._env and
    # reads omega off mol's slot, while df.incore.fill_2c2e (2-center)
    # calls auxmol.intor(...) directly and reads omega off auxmol's own
    # slot. Verified empirically (setting only one of the two leaves the
    # other integral class at the unattenuated Coulomb operator).
    mol.set_range_coulomb(-omega_bohr_inv)
    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    auxmol.set_range_coulomb(-omega_bohr_inv)
    naux = auxmol.nao

    # 2-center metric (P|Q) under erfc.
    v2c = df.incore.fill_2c2e(mol, auxmol)  # (naux, naux)

    # 3-center (P|mu nu) under erfc.
    p_munu = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    p_munu = p_munu.reshape(nbas, nbas, naux).transpose(2, 0, 1)  # (naux, nao, nao)

    # Cholesky V = L L^T ; B = L^{-1} (P|ia)  -- same branch ferric uses
    # for erfc (metric_inverse_sqrt: Cholesky, not eigh -- that branch is
    # ErfCoulomb-only).
    L = cholesky(v2c, lower=True)
    Linv = np.linalg.inv(L)

    # (P|mu nu) -> (P|ia), shape (naux, nocc, nvir)
    tmp = np.einsum("Pmn,na->Pma", p_munu, c_vir, optimize=True)
    p_ia = np.einsum("mi,Pma->Pia", c_occ, tmp, optimize=True)

    b_ov = np.einsum("PQ,Qia->Pia", Linv, p_ia, optimize=True)

    e_os = 0.0
    e_ss = 0.0
    for i in range(nocc):
        for j in range(nocc):
            g_ij = np.einsum("Pa,Pb->ab", b_ov[:, i, :], b_ov[:, j, :], optimize=True)  # (ia|jb) over a,b
            g_ji = g_ij.T  # (ib|ja) = (ja|ib) symmetric relabel: g_ji[a,b] = (ja|ib) = (ib|ja)
            e_ij = eps[first_occ + i] + eps[first_occ + j]
            denom = e_ij - eps[nocc_total:, None] - eps[None, nocc_total:]
            e_os += np.sum(g_ij * g_ij / denom)
            e_ss += np.sum(g_ij * (g_ij - g_ji) / denom)

    return {
        "rhf_energy": e_rhf,
        "e_os": e_os,
        "e_ss": e_ss,
        "mp2_corr": e_os + e_ss,
        "total_energy": e_rhf + e_os + e_ss,
        "nbasis": int(nbas),
        "naux": int(naux),
        "nocc": int(nocc_total),
        "omega_bohr_inv": omega_bohr_inv,
    }


if __name__ == "__main__":
    result = ri_mp2_spin_components(
        WATER_XYZ, "cc-pvdz", "cc-pvdz-ri", OMEGA_BOHR_INV
    )
    result["molecule"] = "h2o"
    result["basis"] = "cc-pvdz"
    result["auxbasis"] = "cc-pvdz-ri"
    result["method"] = "attenuated_rimp2_erfc"
    result["omega_ang_inv"] = OMEGA_ANG_INV
    print(json.dumps(result, indent=2))

    out_path = os.path.join(OUT_DIR, "h2o_cc-pvdz_attenuated-rimp2-erfc0p420.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
