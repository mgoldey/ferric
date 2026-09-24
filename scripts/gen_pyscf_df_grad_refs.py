"""Generate density-fitted SCF nuclear-gradient references (PySCF).

Consumed by `crates/ferric-scf/tests/df_jk_gradient.rs`, which checks that
ferric's analytic gradient of a density-fitted (RI-J / RI-JK) SCF energy is the
derivative of THAT energy — i.e. matches PySCF's DF gradient, which
differentiates the fitted three-centre integrals, the auxiliary-basis centres
and the Coulomb metric.

Cases (Hartree-Fock only: KS gradients differ between the codes at the ~1e-5
Ha/Bohr level from grid construction alone, the same size as the RI effect
being tested, so KS is validated against ferric's own finite differences
instead):

  * h2o_cc-pvdz_hf_rij   — RI-J (only_dfj=True), exact four-centre K
  * h2o_cc-pvdz_hf_rijk  — RI-J + RI-K
  * oh_cc-pvdz_uhf_rijk  — open-shell UHF, RI-J + RI-K

The water geometry is deliberately distorted (no symmetry-zero components).
Aux basis: def2-universal-jkfit for both J and K, as ferric's DfJ/DfK. PySCF
solves the Coulomb metric by Cholesky; ferric's DfK uses V^{-1/2} with an
eigenvalue cut at 1e-10, which drops nothing for the Coulomb metric of these
systems (smallest eigenvalue ~1e-5), so both codes fit the same energy.

Run:  .venv/bin/python scripts/gen_pyscf_df_grad_refs.py
"""

import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf

REFDIR = Path(__file__).resolve().parent.parent / "testdata" / "reference"
AUX = "def2-universal-jkfit"

WATER = "O 0 0 0; H 0.1 0.76 0.59; H -0.05 -0.74 0.62"
OH = "O 0 0 0; H 0.05 0.1 0.97"


def run(label, atom, spin, basis, only_dfj):
    mol = gto.M(atom=atom, basis=basis, spin=spin, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf = mf.density_fit(auxbasis=AUX, only_dfj=only_dfj)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    g = mf.nuc_grad_method()
    # Differentiate the auxiliary-basis centres too (PySCF default, stated
    # explicitly because the ferric fix hinges on it).
    g.auxbasis_response = True
    grad = g.kernel()
    return {
        "label": label,
        "basis": basis,
        "auxbasis": AUX,
        "method": "rhf" if spin == 0 else "uhf",
        "fit": "rij" if only_dfj else "rijk",
        "geometry_angstrom": atom,
        "multiplicity": spin + 1,
        "e_total": float(mf.e_tot),
        "grad": np.asarray(grad).tolist(),
        "converged": bool(mf.converged),
    }


def main():
    REFDIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("h2o_cc-pvdz_hf_rij", WATER, 0, "cc-pvdz", True),
        ("h2o_cc-pvdz_hf_rijk", WATER, 0, "cc-pvdz", False),
        ("oh_cc-pvdz_uhf_rijk", OH, 1, "cc-pvdz", False),
    ]
    for label, atom, spin, basis, only_dfj in cases:
        out = run(label, atom, spin, basis, only_dfj)
        path = REFDIR / f"{label}_dfgrad.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        gmax = max(abs(v) for row in out["grad"] for v in row)
        print(
            f"wrote {path}  E={out['e_total']:.10f}  max|g|={gmax:.4e}  conv={out['converged']}"
        )


if __name__ == "__main__":
    main()
