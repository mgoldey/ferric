#!/usr/bin/env python3
"""Generate PySCF RHF references for ECP atoms/molecules (U5 SCF validation).

For Xe (single atom) and I2, both with def2-SVP + def2-ECP, dump the converged
RHF total energy and HOMO orbital energy. ferric must reproduce these to
<=1e-5 Ha (energy) and <=1 meV (eps_HOMO).

Geometries are in Bohr (ferric's internal unit); the JSON records them so the
ferric test builds the exact same molecule.

Run:  python3 scripts/gw100/gen_ecp_scf_ref.py
Out:  testdata/reference/xe_def2svp_ecp_rhf.json
      testdata/reference/i2_def2svp_ecp_rhf.json
"""
import json
import os
import numpy as np
from pyscf import gto, scf

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.normpath(os.path.join(HERE, "..", "..", "testdata", "reference"))

HARTREE_TO_EV = 27.211386245988

CASES = {
    "xe_def2svp_ecp_rhf": dict(
        atom="Xe 0.0 0.0 0.0",
        spin=0, charge=0,
    ),
    "i2_def2svp_ecp_rhf": dict(
        # I2 bond length 2.666 A -> Bohr; keep Bohr explicit so ferric matches.
        atom="I 0.0 0.0 0.0; I 0.0 0.0 5.037557",  # 2.666 A in Bohr
        spin=0, charge=0,
    ),
}


def run_case(name, cfg):
    mol = gto.M(atom=cfg["atom"], basis="def2-svp", ecp="def2-svp",
                spin=cfg["spin"], charge=cfg["charge"], unit="Bohr", cart=False)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    e_tot = mf.kernel()
    assert mf.converged, f"{name}: PySCF RHF did not converge"
    mo_e = mf.mo_energy
    nocc = mol.nelectron // 2
    e_homo = float(mo_e[nocc - 1])
    e_lumo = float(mo_e[nocc]) if nocc < len(mo_e) else None
    ref = {
        "atom": cfg["atom"],
        "unit": "Bohr",
        "basis": "def2-svp",
        "ecp": "def2-svp",
        "charge": cfg["charge"],
        "spin": cfg["spin"],
        "nelectron": int(mol.nelectron),
        "nao": int(mol.nao),
        "e_tot": float(e_tot),
        "e_nuc": float(mol.energy_nuc()),
        "homo_index": int(nocc - 1),
        "e_homo": e_homo,
        "e_homo_ev": e_homo * HARTREE_TO_EV,
        "e_lumo": e_lumo,
        "mo_energy": [float(x) for x in mo_e],
    }
    out = os.path.join(REFDIR, name + ".json")
    with open(out, "w") as f:
        json.dump(ref, f)
    print(f"{name}: nelec={mol.nelectron} nao={mol.nao} "
          f"E={e_tot:.10f} Ha  e_HOMO={e_homo:.6f} Ha ({e_homo*HARTREE_TO_EV:.4f} eV)")
    print(f"  e_nuc={mol.energy_nuc():.10f}  -> {out}")


def main():
    os.makedirs(REFDIR, exist_ok=True)
    for name, cfg in CASES.items():
        run_case(name, cfg)


if __name__ == "__main__":
    main()
