"""PySCF references for the ECP row of the atomic-property functions.

Consumer: crates/ferric-scf/tests/properties_effective_z.rs.
Output:   testdata/reference/validation/ecp_properties/<system>_<basis>.json

ferric's `esp_at_atoms`, `mulliken_charges` and `lowdin_charges` must pair the
valence-only density of an ECP atom with the screened nuclear charge
`Z - n_core_ecp` (`Atom::effective_z`), not the bare `Z`. The reference is
PySCF's own RHF + ECP on ferric's basis/ECP JSON and Bohr geometry
(`common.build_pyscf_mol`), with

    ESP(R_A)  = sum_{B != A} Zeff_B / |R_A - R_B| - tr[D <mu|1/|r - R_A||nu>]
    q_A^Mull  = Zeff_A - sum_{mu in A} (D S)_{mu mu}
    q_A^Low   = Zeff_A - sum_{mu in A} (S^{1/2} D S^{1/2})_{mu mu}

where Zeff_A = `mol.atom_charges()` (PySCF's ECP-screened charge) and D is the
total AO density. Per-atom sums are invariant to the AO ordering.

Systems: HI and CH3I (the I atom carries the def2 28-electron ECP), def2-SVP.

Run:
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_ecp_properties.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "ecp_properties"
ROW_NAME = "ESP / Mulliken / Lowdin on ECP atoms"
SYSTEMS = ("hi", "ch3i")
BASIS = "def2-svp"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
ECP_MOL_DIR = common.MOL_DIR / "ecp"


def properties(mol, dm):
    """ESP at the nuclei, Mulliken and Lowdin charges from the total density."""
    import numpy as np
    from scipy.linalg import fractional_matrix_power

    zeff = [float(z) for z in mol.atom_charges()]
    coords = mol.atom_coords()
    natm = mol.natm
    esp = []
    for a in range(natm):
        v_nuc = sum(
            zeff[b] / float(np.linalg.norm(coords[a] - coords[b]))
            for b in range(natm)
            if b != a
        )
        with mol.with_rinv_origin(coords[a]):
            rinv = mol.intor("int1e_rinv")
        esp.append(float(v_nuc - np.einsum("ij,ij->", dm, rinv)))
    s = mol.intor("int1e_ovlp")
    s_half = np.real(fractional_matrix_power(s, 0.5))
    pop_mull = np.diag(dm @ s)
    pop_low = np.diag(s_half @ dm @ s_half)
    slices = mol.aoslice_by_atom()
    mull, low = [], []
    for a in range(natm):
        p0, p1 = slices[a][2], slices[a][3]
        mull.append(float(zeff[a] - pop_mull[p0:p1].sum()))
        low.append(float(zeff[a] - pop_low[p0:p1].sum()))
    return {
        "effective_z": zeff,
        "esp_at_atoms": esp,
        "mulliken_charges": mull,
        "lowdin_charges": low,
    }


def main() -> int:
    import numpy
    import pyscf
    from pyscf import scf

    for system in SYSTEMS:
        xyz = ECP_MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        ecp_json = common.basis_json_path(BASIS)
        mol = common.build_pyscf_mol(xyz, BASIS, ecp_json=ecp_json)
        basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
        ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = CONV_TOL, CONV_TOL_GRAD, 500
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}: RHF did not converge")
        mo_i = mf.stability(internal=True, external=False)[0]
        if not numpy.allclose(mo_i, mf.mo_coeff):
            raise RuntimeError(f"{system}: RHF reference is internally unstable")
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": BASIS,
            "nuclear_repulsion": float(mol.energy_nuc()),
            "ecp_core_electrons": ecp_check["ecp_core_electrons"],
            "scf_energy": float(mf.e_tot),
            "properties": properties(mol, mf.make_rdm1()),
        }
        payload["provenance"] = common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "methods": ["scf.RHF"],
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "ecp": "mol.ecp built by common.pyscf_ecp from the basis JSON",
                "eri": "exact 4-index (no density fitting)",
                "numpy": numpy.__version__,
            },
            basis_name=BASIS,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability={"internal_stable": True},
            ecp_json=ecp_json,
            generator="scripts/validation/gen_ecp_properties.py",
            extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
        )
        path = common.write_reference(ROW, system, BASIS, payload)
        print(f"{system}: E={mf.e_tot:.10f} -> {path}")
    print("GEN_ECP_PROPERTIES_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
