"""
Generate PySCF ROHF reference values for ferric-scf integration tests.

Geometries match those in `crates/ferric-scf/tests/rohf.rs`.
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))  # local checkout

from pyscf import gto, scf

os.makedirs("testdata/reference", exist_ok=True)


def run_rohf(atom: str, basis: str, charge: int, spin: int) -> dict:
    # PySCF "spin" = 2S (nelec_a - nelec_b), unitless.
    mol = gto.M(atom=atom, basis=basis, unit="angstrom", charge=charge, spin=spin, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    ss, multiplicity = mf.spin_square()
    return {
        "energy": float(mf.e_tot),
        "converged": bool(mf.converged),
        "nelec_a": int(mol.nelec[0]),
        "nelec_b": int(mol.nelec[1]),
        "s_squared": float(ss),
        "multiplicity": float(multiplicity),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "basis": basis,
        "method": "rohf",
    }


cases = [
    ("oh_cc-pvdz_rohf.json",  "OH",   "O 0 0 0; H 0 0 0.97",              "cc-pvdz", 0, 1),
    ("ch3_cc-pvdz_rohf.json", "CH3",
        "C 0 0 0; H 1.079 0 0; H -0.539500 0.934441 0; H -0.539500 -0.934441 0",
        "cc-pvdz", 0, 1),
    ("o2_sto-3g_rohf.json",   "O2",   "O 0 0 0; O 0 0 1.208",             "sto-3g", 0, 2),
]

for fname, name, atom, basis, charge, spin in cases:
    d = run_rohf(atom, basis, charge, spin)
    d["molecule"] = name
    print(f"{name:5s}/{basis:10s}  E = {d['energy']:.10f}  <S^2> = {d['s_squared']:.4f}  "
          f"conv = {d['converged']}")
    with open(f"testdata/reference/{fname}", "w") as f:
        json.dump(d, f, indent=2)

print("Wrote testdata/reference/*_rohf.json")
