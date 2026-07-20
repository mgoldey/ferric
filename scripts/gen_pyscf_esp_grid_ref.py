"""
Generate a PySCF reference for the raw molecular electrostatic potential
V_QM(r) at a handful of specific real-space points, to cross-check the grid
evaluation inside `ferric_rpa::properties::chelpg_grid_esp` (the shared core
of `chelpg_charges`/`resp_charges`).

WHY THIS SCRIPT, NOT A CHARGE-VALUE CROSS-CHECK: this local PySCF install
(2.13.0) has no `pyscf.esp`/CHELPG/RESP fitting module (verified: neither
`pyscf.esp` nor any *chelpg*/*resp charge* source file exists anywhere under
site-packages). It DOES have `pyscf.tools.cubegen.mep`, which computes
exactly the same physical quantity ferric's grid-ESP evaluator computes:

    V(r) = Vnuc(r) - Vele(r)
         = sum_B Z_B/|r-R_B|  -  sum_uv D_uv <u|1/|r-r_g|v>

(see pyscf/tools/cubegen.py::mep). This is the SAME sign convention as
`esp_at_atoms`'s documented derivation in properties.rs (V_elec = -integral
rho/|r-R|, folded into "+ Sum D_uv <u|-1/|r-R|v>" via the probe-charge
libint trick). So this script does not attempt to reproduce PySCF's cube
grid; instead it evaluates PySCF's own V(r) formula directly (via
`gto.fakemol_for_charges` + `df.incore.aux_e2`, the same primitive `mep`
uses internally) at a fixed, explicit list of points, so ferric's fitting
grid can be pointed at exactly those same points for a strict apples-to-
apples comparison of the underlying V_QM(r) values (the harder, riskier
half of chelpg/resp; the fitting linear-algebra downstream of it is
self-checked in Rust via the constraint sum-rule and fit-residual tests).

CRITICAL (same lesson as gen_pyscf_mulliken_ref.py): basis loaded from
ferric's OWN bundled JSON, not a PySCF string basis name.

Writes JSON to testdata/reference/{mol}_{basis}_esp_grid.json.

Usage:
  python scripts/gen_pyscf_esp_grid_ref.py water cc-pvdz
  python scripts/gen_pyscf_esp_grid_ref.py methanol cc-pvdz
"""
import json
import os
import sys

import numpy as np
from pyscf import gto, scf, df, lib
from pyscf.data.elements import ELEMENTS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOHR_PER_ANGSTROM = 1.8897259886


def _bse_elem_to_pyscf(elem):
    shells = []
    for sh in elem["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        cols = [[float(x) for x in col] for col in sh["coefficients"]]
        for l in sh["angular_momentum"]:
            block = [l]
            for i, e in enumerate(exps):
                block.append([e] + [cols[j][i] for j in range(len(cols))])
            shells.append(block)
    return shells


def load_ferric_basis(basis):
    path = os.path.join(ROOT, "crates/ferric-core/src/basis/bundled", f"{basis}.json")
    with open(path) as fh:
        d = json.load(fh)
    return {ELEMENTS[int(z)]: _bse_elem_to_pyscf(elem) for z, elem in d["elements"].items()}


# Standard equilibrium-ish geometries (Angstrom), matching the inline
# geometries used in properties_mulliken.rs / properties_lowdin.rs where
# applicable, plus methanol (CHELPG's whole point is a molecule with a
# buried/less-exposed heavy atom -- here, the carbon).
GEOMETRIES = {
    "water": "O 0.000000 0.000000 0.117790; H 0.000000 0.755453 -0.471161; H 0.000000 -0.755453 -0.471161",
    "methanol": (
        "C 0.000000 0.000000 0.000000; "
        "O 0.000000 0.000000 1.430000; "
        "H 0.882700 0.000000 -0.363200; "
        "H -0.441350 0.764460 -0.363200; "
        "H -0.441350 -0.764460 -0.363200; "
        "H 0.882700 0.000000 1.830000"
    ),
}


def build_mol(label, basis):
    atom = GEOMETRIES[label]
    ferric_basis = load_ferric_basis(basis)
    return gto.M(atom=atom, basis=ferric_basis, unit="Angstrom", charge=0, spin=0)


def v_qm_at_points(mol, dm, points_bohr):
    """PySCF's own V(r) = Vnuc - Vele formula (the primitive `cubegen.mep`
    uses internally), evaluated at explicit points given in BOHR (converted
    to PySCF's native Bohr internally -- PySCF's `mol.atom_coord` is already
    in Bohr for a `gto.M` built with unit='Angstrom', so no further
    conversion is needed once `points_bohr` is passed straight through)."""
    coords = np.asarray(points_bohr, dtype=float)

    vnuc = np.zeros(len(coords))
    for i in range(mol.natm):
        r = mol.atom_coord(i)  # Bohr
        z = mol.atom_charge(i)
        rp = r - coords
        vnuc += z / np.linalg.norm(rp, axis=1)

    vele = np.empty(len(coords))
    for p0, p1 in lib.prange(0, len(coords), 600):
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = df.incore.aux_e2(mol, fakemol)
        vele[p0:p1] = np.einsum("ijp,ij->p", ints, dm)

    return (vnuc - vele).tolist()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "water"
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"

    mol = build_mol(label, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    # A small, explicit, deterministic set of points in Bohr: a handful of
    # points a few Bohr from the molecule in different directions, well
    # outside any atom's vdW radius (so they are exactly the kind of point
    # the CHELPG grid keeps) but still close enough for V_QM to be
    # non-negligible.
    points_bohr = [
        [0.0, 0.0, 4.0],
        [0.0, 3.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [-3.0, -1.0, 1.5],
        [0.0, 0.0, -4.0],
    ]

    v = v_qm_at_points(mol, dm, points_bohr)

    out = {
        "molecule": label,
        "basis": basis,
        "scf_energy": float(mf.e_tot),
        "points_bohr": points_bohr,
        "v_qm_hartree": v,
        "note": (
            "V_QM(r) = Vnuc(r) - Vele(r) via PySCF's own primitives "
            "(gto.fakemol_for_charges + df.incore.aux_e2), the same formula "
            "pyscf.tools.cubegen.mep uses internally. No pyscf.esp/CHELPG/"
            "RESP module exists in this install (verified by import + "
            "filesystem search) so this cross-checks the underlying V_QM(r) "
            "grid evaluation shared by ferric's chelpg_charges/resp_charges, "
            "not the fitted charge values themselves. Basis loaded from "
            "ferric's OWN bundled JSON, not a PySCF string basis name."
        ),
    }

    ref_dir = os.path.join(ROOT, "testdata/reference")
    os.makedirs(ref_dir, exist_ok=True)
    out_path = os.path.join(ref_dir, f"{label}_{basis}_esp_grid.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
