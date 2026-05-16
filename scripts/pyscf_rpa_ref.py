"""
Generate PySCF RPA reference values for ferric integration tests.
Run once, store JSON output in testdata/reference/.
"""
import json, os, sys

try:
    from pyscf import gto, scf
    from pyscf.tddft import rpa as pyscf_rpa
except ImportError:
    sys.exit("pyscf not installed — run: pip install pyscf")

os.makedirs("testdata/reference", exist_ok=True)

# H2 / STO-3G  (geometry in Angstrom matching h2.xyz)
mol_h2 = gto.M(
    atom="H 0 0 0; H 0 0 0.7414",
    basis="sto-3g",
    unit="angstrom",
    verbose=0,
)
mf_h2 = scf.RHF(mol_h2).run()
rpa_h2 = pyscf_rpa.RPA(mf_h2)
rpa_h2.kernel()
e_h2 = float(rpa_h2.e_corr)
print(f"H2/STO-3G  E_c(RPA) = {e_h2:.10f} Ha")
with open("testdata/reference/h2_sto-3g_rpa.json", "w") as f:
    json.dump({"molecule": "H2", "basis": "sto-3g", "method": "rpa", "e_corr": e_h2}, f, indent=2)

# H2O / cc-pVDZ  (geometry in Angstrom matching water.xyz)
mol_h2o = gto.M(
    atom="O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
    basis="cc-pvdz",
    unit="angstrom",
    verbose=0,
)
mf_h2o = scf.RHF(mol_h2o).run()
rpa_h2o = pyscf_rpa.RPA(mf_h2o)
rpa_h2o.kernel()
e_h2o = float(rpa_h2o.e_corr)
print(f"H2O/cc-pVDZ E_c(RPA) = {e_h2o:.10f} Ha")
with open("testdata/reference/h2o_cc-pvdz_rpa.json", "w") as f:
    json.dump({"molecule": "H2O", "basis": "cc-pvdz", "method": "rpa", "e_corr": e_h2o}, f, indent=2)

print("Reference files written to testdata/reference/")
