"""PySCF DF-MP2 opposite-spin references for the laplace-sos-mp2 grade.

Consumer: crates/ferric-mp2/tests/validation_sos_mp2.rs
Output:   testdata/reference/validation/sos_mp2/<system>_<basis>.json

SOS-MP2 is E = c_os * E_OS, with E_OS the opposite-spin part of the
density-fitted MP2 correlation energy. ferric's `laplace_sos_mp2` evaluates E_OS
with a minimax-Laplace denominator. The independent reference is PySCF's own
`mp.dfmp2.DFMP2` (frozen=None) with ferric's bundled orbital and auxiliary basis
JSON, on an exact four-centre J/K RHF: its `e_corr_os` is the exact-denominator
E_OS at the same aux basis. The file also records c_os * E_OS for the
Jung/Head-Gordon c_os = 1.3 that is ferric's default, and PySCF's own
E_OS + E_SS = E_corr so a transposed component cannot pass.

Run:
    box-lease exec -- uv run --no-sync python scripts/gen_pyscf_sos_mp2_refs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import common  # noqa: E402

ROW = "sos_mp2"
ROW_NAME = "Laplace SOS-MP2"
# system -> (orbital basis, aux basis), the five systems of the Laplace-MP2 row
SYSTEMS = {
    "h2o": ("cc-pvdz", "cc-pvdz-ri"),
    "h2o_aug": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "ch4": ("cc-pvdz", "cc-pvdz-ri"),
    "butadiene": ("cc-pvdz", "cc-pvdz-ri"),
    "alkane_8": ("cc-pvdz", "cc-pvdz-ri"),
}
GEOMETRY = {
    "h2o": "h2o",
    "h2o_aug": "h2o",
    "ch4": "ch4",
    "butadiene": "butadiene",
    "alkane_8": "alkane_8",
}
C_OS_DEFAULT = 1.3
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10


def main() -> int:
    import pyscf
    from pyscf import df, scf
    from pyscf.mp import dfmp2

    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems {sorted(unknown)}; known {sorted(SYSTEMS)}")
    for system, (basis_name, aux_name) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{GEOMETRY[system]}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)

        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}: PySCF RHF did not converge")

        aux_bas, aux_cart = common.pyscf_basis(aux_name, symbols)
        assert not aux_cart, f"{aux_name}: Cartesian aux shells not supported here"
        mp = dfmp2.DFMP2(mf, frozen=None)
        mp.with_df = df.DF(mol)
        mp.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
        mp.with_df.auxmol.cart = False
        mp.with_df.auxmol.build()
        mp.with_df.auxbasis = aux_bas
        mp.kernel()
        e_os, e_ss = float(mp.e_corr_os), float(mp.e_corr_ss)
        if abs(e_os + e_ss - float(mp.e_corr)) > 1e-12:
            raise RuntimeError(f"{system}: E_OS + E_SS != E_corr")

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "aux_basis": aux_name,
            "geometry_stem": GEOMETRY[system],
            "nao": mol.nao_nr(),
            "nocc": mol.nelectron // 2,
            "nuclear_repulsion": float(mol.energy_nuc()),
            "e_rhf": float(mf.e_tot),
            "dfmp2": {
                "e_os": e_os,
                "e_ss": e_ss,
                "e_corr": float(mp.e_corr),
            },
            "sos": {
                "c_os_default": C_OS_DEFAULT,
                "sos_corr_default": C_OS_DEFAULT * e_os,
                "sos_total_default": float(mf.e_tot) + C_OS_DEFAULT * e_os,
            },
            "provenance": common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "scf": f"scf.RHF exact J/K, conv_tol {CONV_TOL}, "
                    f"conv_tol_grad {CONV_TOL_GRAD}",
                    "mp2": "mp.dfmp2.DFMP2(frozen=None), with_df auxmol from ferric's "
                    "aux JSON; e_corr_os / e_corr_ss",
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=aux_name,
                frozen_core=0,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/gen_pyscf_sos_mp2_refs.py",
                extra={
                    "basis_self_check": basis_check,
                    "aux_basis_json_sha256": common.sha256_file(
                        common.basis_json_path(aux_name)
                    ),
                },
            ),
        }
        path = common.write_reference(ROW, system, basis_name, payload)
        print(f"{system:10s} E_OS={e_os:+.12f} E_SS={e_ss:+.12f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
