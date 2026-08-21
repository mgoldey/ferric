"""
Generate PySCF RPA reference values for ferric integration tests.

Critical pattern: use `RPA(mf).with_df = df.DF(mol, auxbasis=...)` instead of
`RPA(mf).density_fit(auxbasis=...)`. The `density_fit()` method on `RPA`
returns a `DFRMP2` object which computes a DIFFERENT quantity (not the same
RI-RPA correlation energy ferric implements). The manual `with_df` assignment
uses the same RI decomposition as ferric, producing matching results.

Run: python3 scripts/pyscf_rpa_ref.py
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))  # local checkout

from pyscf import df, gto, scf
from pyscf.gw.rpa import RPA

os.makedirs("testdata/reference", exist_ok=True)


def run_rpa(atom: str, basis: str, aux: str) -> float:
    mol = gto.M(atom=atom, basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()  # non-DF SCF, matches ferric's RHF
    rpa = RPA(mf)
    rpa.with_df = df.DF(mol, auxbasis=aux)  # explicit RI basis, matches ferric
    rpa.kernel()
    return float(rpa.e_corr)


cases = [
    ("H2",  "H 0 0 0; H 0 0 0.7414",
     "sto-3g",      "sto-3g",      "h2_sto-3g_rpa.json"),
    ("H2O", "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
     "cc-pvdz",     "cc-pvdz-ri",  "h2o_cc-pvdz_rpa.json"),
    ("H2O", "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
     "aug-cc-pvdz", "aug-cc-pvdz-rifit",  "h2o_aug-cc-pvdz_rpa.json"),
    ("H2O", "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
     "aug-cc-pvtz", "aug-cc-pvtz-rifit",  "h2o_aug-cc-pvtz_rpa.json"),
]

for name, atom, basis, aux, fname in cases:
    e = run_rpa(atom, basis, aux)
    print(f"{name}/{basis:15s} (aux={aux:25s}) E_c(RPA) = {e:.10f} Ha")
    with open(f"testdata/reference/{fname}", "w") as f:
        json.dump(
            {"molecule": name, "basis": basis, "aux": aux,
             "method": "rpa", "e_corr": e},
            f, indent=2,
        )

print("Reference files written to testdata/reference/")
