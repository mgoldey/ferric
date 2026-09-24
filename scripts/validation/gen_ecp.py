"""PySCF references for the VALIDATION.md rows "RHF+ECP" and "ECP gradients".

Consumer: crates/ferric-scf/tests/validation_ecp.rs.
Output:   testdata/reference/validation/ecp/<system>_<basis>.json
(ORCA cross-check references come from gen_ecp_orca.py, same directory,
`orca_` prefix.)

Systems (geometries in testdata/molecules/validation/ecp/, each deliberately
OFF equilibrium so the gradient is large and non-vacuous; see the comment
line of each xyz):

    hi       HI        RHF   I carries the def2 28-core ECP (Peterson 2003)
    ch3i     CH3I      RHF   distorted, so all three Cartesian directions move
    rbh      RbH       RHF   a DIFFERENT ECP: def2 Rb (1-term local channel)
    hbr      HBr       RHF   NEGATIVE CONTROL: Br is all-electron in def2
    snh4     SnH4      RHF   def2-TZVP ONLY (ferric's def2-SVP has no Sn)
    i_atom   I atom    UHF   2P doublet, stability-followed

x def2-SVP and def2-TZVP (snh4: def2-TZVP only), both from FERRIC's bundled
JSON. The ECP is read from the SAME JSON file: ferric's def2 basis files carry
their ECPs inline (`ecp_electrons` / `ecp_potentials`), and that inline block
is what `Molecule::apply_ecp` / libecpint use. (`def2-ecp.json` holds only I
and is numerically identical to the inline I block; it is not used here.)
The ECP reaches PySCF through `common.pyscf_ecp`, and
`common.check_ecp_like_for_like` asserts the per-atom core-electron counts,
the electron count and the ECP term count match what ferric builds before any
number is written.

Per system x basis:
  * RHF at conv_tol 1e-12 / conv_tol_grad 1e-8 (exact 4-index ERIs, no DF),
    then PySCF internal RHF stability; an unstable reference is REFUSED.
  * the ANALYTIC nuclear gradient `mf.nuc_grad_method().kernel()` (PySCF's
    ECP derivative integrals). A reference whose max |g| < 1e-3 Ha/Bohr is
    REFUSED: a near-zero gradient cannot distinguish a right one from a wrong
    one (the reason every geometry is stretched).
  * i_atom: UHF via `common.run_open_shell` (three guesses, stability loop,
    unstable states refused), energy + <S^2> only. The 2P atom is spatially
    degenerate, so its orbital Hessian has zero modes (rotating the p hole);
    lambda_min is recorded but NOT asserted by the Rust test.

Run (light step; seconds to a minute per system):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_ecp.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "ecp"
ROW_NAME = "RHF+ECP / ECP gradients"
ECP_MOL_DIR = common.MOL_DIR / "ecp"
BASES = ("def2-svp", "def2-tzvp")
# system -> (charge, multiplicity, method, bases)
SYSTEMS = {
    "hi": (0, 1, "rhf", BASES),
    "ch3i": (0, 1, "rhf", BASES),
    "rbh": (0, 1, "rhf", BASES),
    "hbr": (0, 1, "rhf", BASES),
    "snh4": (0, 1, "rhf", ("def2-tzvp",)),
    "i_atom": (0, 2, "uhf", BASES),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-8
MIN_GRAD = 1e-3
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10


def run_rhf(mol) -> dict:
    """Closed-shell RHF + stability + analytic gradient. Raises on anything
    that would make the reference unfit to write."""
    import numpy as np
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf.converged:
        raise RuntimeError(f"{mol.atom}: RHF did not converge")
    _mo_i, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError(f"{mol.atom}: RHF internal instability; refusing to write")
    g = mf.nuc_grad_method()
    g.verbose = 0
    grad = np.asarray(g.kernel())
    gmax = float(np.max(np.abs(grad)))
    if gmax < MIN_GRAD:
        raise RuntimeError(
            f"max |gradient| {gmax:.2e} < {MIN_GRAD:.0e}: a near-zero gradient is "
            "a vacuous reference (stretch the geometry)"
        )
    nocc = mol.nelectron // 2
    e = mf.mo_energy
    return {
        "energy": float(mf.e_tot),
        "converged": True,
        "homo": float(e[nocc - 1]),
        "lumo": float(e[nocc]),
        "stability": {
            "internal_stable": True,
            "kind": "PySCF RHF internal (real)",
        },
        "gradient": [[float(v) for v in row] for row in grad],
        "gradient_units": "Hartree/Bohr, atoms in xyz order",
        "gradient_max_abs": gmax,
        "gradient_sum": [float(v) for v in grad.sum(axis=0)],
    }


def main() -> int:
    import numpy
    import pyscf

    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult, method, bases) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = ECP_MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in bases:
            ecp_json = common.basis_json_path(basis_name)
            mol = common.build_pyscf_mol(
                xyz,
                basis_name,
                charge=charge,
                multiplicity=mult,
                ecp_json=ecp_json,
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "method": method,
                "nao": mol.nao_nr(),
                "nelectron": int(mol.nelectron),
                "ecp_core_electrons": ecp_check["ecp_core_electrons"],
                "nuclear_repulsion": float(mol.energy_nuc()),
            }
            if method == "rhf":
                block = run_rhf(mol)
                payload["rhf"] = block
                summary = (
                    f"RHF {block['energy']:.10f} |g|max {block['gradient_max_abs']:.3e}"
                )
                stability = block["stability"]
            else:
                block = common.run_open_shell(
                    mol,
                    "uhf",
                    conv_tol=CONV_TOL,
                    conv_tol_grad=CONV_TOL_GRAD,
                    max_stab_rounds=MAX_STAB_ROUNDS,
                    guesses=GUESSES,
                )
                payload["uhf"] = block
                summary = (
                    f"UHF {block['energy']:.10f} <S2>={block['s_squared']:.6f} "
                    f"lmin={block['stability']['lambda_min']:+.3e} "
                    f"multi={block['multiple_stable_minima']}"
                )
                stability = block["stability"]
            payload["provenance"] = common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "methods": ["scf.RHF" if method == "rhf" else "scf.UHF"]
                    + (["nuc_grad_method() (analytic)"] if method == "rhf" else []),
                    "conv_tol": CONV_TOL,
                    "conv_tol_grad": CONV_TOL_GRAD,
                    "max_cycle": 500,
                    "ecp": "mol.ecp built by common.pyscf_ecp from the basis JSON's "
                    "inline ecp_potentials",
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
                stability=stability,
                ecp_json=ecp_json,
                generator="scripts/validation/gen_ecp.py",
                extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
            )
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
            print(
                f"{system:7s} {basis_name:9s} ncore={ecp_check['ecp_core_electrons']} "
                f"nelec={mol.nelectron} {summary}"
            )
    print(f"GEN_ECP_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
