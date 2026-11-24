"""PySCF references for closed-shell anions whose RHF HOMO lies above zero.

Consumer: crates/ferric-cc/tests/cc_anion_occupied_count.rs.
Output:   testdata/reference/<system>_<basis>_anion_ccsd.json

Why these systems: an anion in a basis without diffuse functions can have a
positive RHF HOMO energy. A CC driver that takes the occupied count from the
number of negative orbital energies then correlates fewer electrons, or
fails on a dimension check, instead of the N electrons the RHF solved for.
Every system written here is asserted to have HOMO > 0, so the reference is
never vacuous for that defect. Measured in def2-SVP (PySCF RHF):

    CH3-   HOMO +0.016 Ha, 4 negative MO energies, 5 occupied
    Te2-   HOMO +0.193 Ha, 10 negative MO energies, 13 occupied
    (F-, OH-, Cl-, I- in def2-SVP and F-, OH- in 6-31G all have HOMO < 0.)

Te2- also carries the def2 28-electron ECP, so its occupied count must come
from the VALENCE electron count (26). A free dianion is unbound; in this
finite basis it is a well-defined closed-shell RHF state, which is all the
test needs.

The correlation energies use the like-for-like density-fitting construction
of scripts/validation/gen_cc.py (its `df_factor` and `run_df_block`, imported
unchanged): exact-integral PySCF RHF with ferric's bundled orbital basis,
(pq|rs) = sum_P B^P_pq B^P_rs with ferric's bundled auxiliary basis, a
diagonal canonical Fock, then pyscf.cc.ccsd.CCSD and pyscf.cc.ccsd_t.kernel
on those integrals. The same bars as crates/ferric-cc/tests/validation_cc.rs
apply to the Rust side.

Geometry: given inline below in Angstrom, converted with ferric's constant
(common.ANGSTROM_TO_BOHR) and handed to PySCF in Bohr. The XYZ text is stored
in the JSON so the Rust test builds the identical molecule from it.

Run:
    box-lease exec -- /path/to/.venv/bin/python scripts/gen_pyscf_cc_anion_refs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import common  # noqa: E402
import gen_cc  # noqa: E402

OUT_DIR = ROOT / "testdata" / "reference"

# system -> (xyz in Angstrom, charge, orbital basis, aux basis, has an ECP,
#            frozen-core count for the frozen-core block). The def2 ECP is the
#            block inline in ferric's bundled basis JSON, which is what
#            `Molecule::apply_ecp` reads.
CH3 = """C 0.0 0.0 0.0
H 1.03 0.0 -0.35
H -0.515 0.892 -0.35
H -0.515 -0.892 -0.35"""
CASES = {
    "ch3": (CH3, -1, "def2-svp", "def2-svp-rifit", False, 1),
    "te": ("Te 0.0 0.0 0.0", -2, "def2-svp", "def2-svp-rifit", True, 4),
}


def parse_xyz_rows(rows: str) -> tuple[list[str], list[list[float]]]:
    symbols, coords = [], []
    for row in rows.splitlines():
        tok = row.split()
        symbols.append(tok[0])
        coords.append([float(v) * common.ANGSTROM_TO_BOHR for v in tok[1:4]])
    return symbols, coords


def main() -> int:
    import numpy
    import pyscf
    import scipy
    from pyscf import gto, scf

    only = set(sys.argv[1:])
    for system, (
        rows,
        charge,
        basis_name,
        aux_name,
        has_ecp,
        n_frozen,
    ) in CASES.items():
        if only and system not in only:
            continue
        symbols, coords = parse_xyz_rows(rows)
        ecp_json = common.basis_json_path(basis_name) if has_ecp else None
        basis, cart = common.pyscf_basis(basis_name, symbols)
        aux_basis, aux_cart = common.pyscf_basis(aux_name, symbols)
        if cart or aux_cart:
            raise ValueError(f"{system}: Cartesian shells; not handled")
        atom = common.pyscf_atom_bohr(symbols, coords)
        mol = gto.M(
            atom=atom,
            unit="Bohr",
            basis=basis,
            ecp=common.pyscf_ecp(ecp_json, symbols) if ecp_json else {},
            cart=False,
            charge=charge,
            spin=0,
            verbose=0,
        )
        auxmol = gto.M(
            atom=atom,
            unit="Bohr",
            basis=aux_basis,
            cart=False,
            charge=charge,
            verbose=0,
        )
        assert mol.nao_nr() == common.ferric_nao(basis_name, symbols)
        assert auxmol.nao_nr() == common.ferric_nao(aux_name, symbols)
        ecp_check = (
            common.check_ecp_like_for_like(mol, ecp_json, symbols) if ecp_json else None
        )

        mf = scf.RHF(mol)
        mf.conv_tol = gen_cc.SCF_CONV_TOL
        mf.conv_tol_grad = gen_cc.SCF_CONV_TOL_GRAD
        mf.max_cycle = 200
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}: RHF did not converge")
        nocc = mol.nelectron // 2
        homo = float(mf.mo_energy[nocc - 1])
        n_negative = int((mf.mo_energy < 0.0).sum())
        if homo <= 0.0:
            raise RuntimeError(
                f"{system}/{basis_name}: HOMO {homo} is not positive; the reference "
                "would not exercise the occupied count"
            )

        b_ao, j2c_min, j2c_max = gen_cc.df_factor(mol, auxmol)
        methods = ("ccsd", "ccsd_t")
        all_e = gen_cc.run_df_block(mf, mol, b_ao, 0, methods)
        fc = gen_cc.run_df_block(mf, mol, b_ao, n_frozen, methods)

        payload = {
            "system": f"{system}{charge:+d}",
            "xyz_angstrom": f"{len(symbols)}\n{system}{charge:+d}\n{rows}\n",
            "basis": basis_name,
            "aux_basis": aux_name,
            "ecp": f"inline in {basis_name}" if has_ecp else "none",
            "charge": charge,
            "multiplicity": 1,
            "nelectron": int(mol.nelectron),
            "nao": int(mol.nao_nr()),
            "naux": int(auxmol.nao_nr()),
            "nocc": int(nocc),
            "nvir": int(mol.nao_nr() - nocc),
            "n_negative_mo_energies": n_negative,
            "n_frozen_core_control": int(n_frozen),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "rhf": {
                "energy": float(mf.e_tot),
                "homo": homo,
                "lumo": float(mf.mo_energy[nocc]),
            },
            "df": {"metric_min_eigenvalue": j2c_min, "metric_max_eigenvalue": j2c_max},
            "all_electron": all_e,
            "frozen_core": fc,
            "provenance": {
                "code": "PySCF + numpy",
                "version": pyscf.__version__,
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "reference": "scf.RHF, exact 4-index J/K",
                "scf_conv_tol": gen_cc.SCF_CONV_TOL,
                "scf_conv_tol_grad": gen_cc.SCF_CONV_TOL_GRAD,
                "correlation_integrals": "DF, gen_cc.df_factor (B = L^-1 (P|mn))",
                "cc": "gen_cc.run_df_block: pyscf.cc.ccsd.CCSD + ccsd_t.kernel on DF eris",
                "cc_conv_tol": gen_cc.CC_CONV_TOL,
                "cc_conv_tol_normt": gen_cc.CC_CONV_TOL_NORMT,
                "basis_sha256": common.sha256_file(common.basis_json_path(basis_name)),
                "aux_basis_sha256": common.sha256_file(
                    common.basis_json_path(aux_name)
                ),
                "ecp_sha256": common.sha256_file(ecp_json) if ecp_json else "none",
                "ecp_check": ecp_check,
                "bohr_in_angstrom": common.BOHR_IN_ANGSTROM,
                "generator": "scripts/gen_pyscf_cc_anion_refs.py",
                "git_head": common.git_head(),
            },
        }
        path = OUT_DIR / f"{system}_{basis_name}_anion_ccsd.json"
        import json

        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{system}{charge:+d}/{basis_name} nelec={mol.nelectron} nao={mol.nao_nr()} "
            f"HOMO={homo:+.6f} n(eps<0)={n_negative} nocc={nocc} "
            f"E_CCSD={all_e['ccsd']['e_corr']:.12f} E_T={all_e['ccsd_t']['e_t']:.12f} "
            f"fc E_CCSD={fc['ccsd']['e_corr']:.12f}",
            flush=True,
        )
        print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
