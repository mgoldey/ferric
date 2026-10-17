"""PySCF point-group references for ferric_core::symmetry (issue #294, stage 1).

Consumer: crates/ferric-integrals/tests/symmetry_ao_rep.rs.
Output:   testdata/reference/symmetry_pointgroup.json

For each (molecule, basis, pure/Cartesian) records the Abelian point group
PySCF's symmetry-adapted SCF uses (`mol.groupname`; PySCF reports the full
linear group for linear molecules, mapped here to the Abelian subgroup it
blocks with: Dooh -> D2h, Coov -> C2v) and the dimension of every irrep's
symmetry-adapted AO space (`mol.symm_orb`). Irrep LABELS are not compared:
PySCF orients and labels its own frame. Dimensions are a frame-independent
fingerprint of the AO representation.

Run: ~/qc/ferric/.venv/bin/python scripts/gen_pyscf_symmetry_refs.py
"""

from __future__ import annotations

import json
import math
import pathlib

import pyscf
from pyscf import gto

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "testdata" / "reference" / "symmetry_pointgroup.json"
ABELIAN_OF = {"Dooh": "D2h", "Coov": "C2v"}


def benzene() -> str:
    rows = []
    for el, r in (("C", 1.397), ("H", 2.48)):
        for i in range(6):
            t = i * math.pi / 3
            rows.append(f"{el} {r * math.cos(t):.10f} {r * math.sin(t):.10f} 0.0")
    return "\n".join(rows)


MOLECULES = {
    "water": "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692",
    "nh3": "N 0 0 0.1\nH 0.94 0 -0.3\nH -0.47 0.814 -0.3\nH -0.47 -0.814 -0.3",
    "ethylene": "C 0 0 0.667\nC 0 0 -0.667\nH 0 0.92 1.24\nH 0 -0.92 1.24\nH 0 0.92 -1.24\nH 0 -0.92 -1.24",
    "methane": "C 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629",
    "h2": "H 0 0 0\nH 0 0 0.74",
    "n2": "N 0 0 0\nN 0 0 1.098",
    "hf": "H 0 0 0\nF 0 0 0.917",
    "benzene": benzene(),
    "asym": "C 0 0 0\nO 1.2 0.1 0\nH -0.5 0.9 0.3\nF 0.2 -1.1 0.8",
}
BASES = [("sto-3g", False), ("cc-pvdz", False), ("def2-svp", False), ("cc-pvdz", True)]


def main() -> None:
    cases = []
    for name, geom in MOLECULES.items():
        for basis, cart in BASES:
            mol = gto.M(
                atom=geom, basis=basis, cart=cart, unit="A", symmetry=True, verbose=0
            )
            groupname = mol.groupname
            abelian = ABELIAN_OF.get(groupname, groupname)
            if abelian != groupname:
                # PySCF splits linear-group irreps by |m| (E1gx, E2gy, ...),
                # finer than the Abelian subgroup. Rebuild in the subgroup so
                # the dimensions are those of the D2h / C2v blocking.
                mol = gto.M(
                    atom=geom,
                    basis=basis,
                    cart=cart,
                    unit="A",
                    symmetry=abelian,
                    verbose=0,
                )
            cases.append(
                {
                    "molecule": name,
                    "xyz_angstrom": geom,
                    "basis": basis,
                    "cartesian": cart,
                    "nao": int(mol.nao),
                    "pyscf_groupname": groupname,
                    "abelian_group": abelian,
                    "irrep_dimensions": sorted(int(s.shape[1]) for s in mol.symm_orb),
                }
            )
    doc = {
        "generator": "scripts/gen_pyscf_symmetry_refs.py",
        "pyscf": pyscf.__version__,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
