"""PySCF references for the VALIDATION.md "UHF / ROHF" row (validation tier W0).

Consumer: crates/ferric-scf/tests/validation_open_shell_scf.rs.
Output:   testdata/reference/validation/uhf_rohf/<system>_<basis>.json

Systems (geometries in testdata/molecules/validation/, sources in each xyz
comment line):

    ho2          HO2   2A''  doublet
    no2          NO2   2A1   doublet
    ch2_triplet  CH2   3B1   triplet
    allyl        C3H5  2A2   doublet (idealized C2v geometry)

x bases 6-31G and def2-SVP, both taken from FERRIC's bundled JSON (never
PySCF's built-in copies) via scripts/validation/common.py.

Per system x basis, both UHF and ROHF are converged at conv_tol 1e-11 from
three initial guesses, each followed through a stability() loop to an
internally stable state; the lowest stable state is kept, and a system with no
stable state is REFUSED rather than written. UHF blocks also carry <S^2>, the
alpha/beta HOMO/LUMO, the alpha SOMO energies, and the lowest eigenvalue of
PySCF's own UHF orbital Hessian (dense). ROHF orbital energies are recorded
but flagged convention-dependent (Roothaan coupling constants differ between
codes), so the Rust test asserts only the ROHF energy.

Why the ROHF state is not stability-checked on the ferric side: ferric's
`solve_rohf` deliberately SKIPS `check_stability` (the Roothaan Hessian is a
third operator; see RhfConfig::check_stability). PySCF's ROHF internal
stability is still run here, so the REFERENCE is on a stable ROHF state and a
ferric saddle shows up as an energy mismatch.

Run (light step; seconds per system):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_uhf_rohf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "uhf_rohf"
ROW_NAME = "UHF / ROHF"
BASES = ("6-31g", "def2-svp")
# system -> (charge, multiplicity)
SYSTEMS = {
    "ho2": (0, 2),
    "no2": (0, 2),
    "ch2_triplet": (0, 3),
    "allyl": (0, 2),
}
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10


def main() -> int:
    import numpy
    import pyscf

    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in BASES:
            mol = common.build_pyscf_mol(
                xyz, basis_name, charge=charge, multiplicity=mult
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            uhf = common.run_open_shell(
                mol,
                "uhf",
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
                max_stab_rounds=MAX_STAB_ROUNDS,
                guesses=GUESSES,
            )
            rohf = common.run_open_shell(
                mol,
                "rohf",
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
                max_stab_rounds=MAX_STAB_ROUNDS,
                guesses=GUESSES,
            )
            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "nao": mol.nao_nr(),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "uhf": uhf,
                "rohf": rohf,
                "provenance": common.provenance(
                    code="PySCF",
                    version=pyscf.__version__,
                    keywords={
                        "methods": ["scf.UHF", "scf.ROHF"],
                        "conv_tol": CONV_TOL,
                        "conv_tol_grad": CONV_TOL_GRAD,
                        "max_cycle": 500,
                        "init_guess_scan": list(GUESSES),
                        "stability": "internal=True external=False, restart from "
                        f"the unstable direction, max {MAX_STAB_ROUNDS} rounds",
                        "eri": "exact 4-index (no density fitting)",
                        "numpy": numpy.__version__,
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid=None,
                    aux=None,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability={
                        "uhf": uhf["stability"],
                        "rohf": rohf["stability"],
                    },
                    generator="scripts/validation/gen_uhf_rohf.py",
                    extra={"basis_self_check": basis_check},
                ),
            }
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
            print(
                f"{system:12s} {basis_name:9s} "
                f"UHF {uhf['energy']:.10f} <S2>={uhf['s_squared']:.6f} "
                f"lmin={uhf['stability']['lambda_min']:+.3e} "
                f"rounds={uhf['stability']['rounds']} "
                f"multi={uhf['multiple_stable_minima']} | "
                f"ROHF {rohf['energy']:.10f} rounds={rohf['stability']['rounds']} "
                f"multi={rohf['multiple_stable_minima']}"
            )
    print(f"GEN_UHF_ROHF_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
