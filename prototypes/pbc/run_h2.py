import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf, ewald_nn
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf

a = np.eye(3) * 4.0
atoms = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
basis = "sto-3g"
cell = Cell(a, atoms, basis)

# --- oracle: PySCF pbc
pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.build()
print(
    "E_nn  pyscf",
    pc.energy_nuc(),
    " ours(w=0.8,1.5,3):",
    [ewald_nn(cell, w) for w in (0.8, 1.5, 3.0)],
)
S_ref = pc.pbc_intor("int1e_ovlp")
T_ref = pc.pbc_intor("int1e_kin")

res = {}
for w in (None, 3.0, 1.5, 1.0):
    ints = build_integrals(cell, w)
    e, eps, it = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], cell.mol.nelectron)
    res[w] = (e, eps)
    print(
        f"w={w}: E={e:.10f} it={it} eps={eps}  |S-Sref|={abs(ints['S'] - S_ref).max():.1e} |T-Tref|={abs(ints['T'] - T_ref).max():.1e}"
    )

for name, mk in [("GDF", lambda m: m.density_fit()), ("RSJK", lambda m: m.rs_jk())]:
    t = time.time()
    mf = mk(pscf.RHF(pc, exxdiv=None))
    mf.conv_tol = 1e-11
    if name == "RSJK":
        mf.with_df = pdf.GDF(pc)  # hcore (all-electron nuc) from GDF
    e = mf.kernel()
    print(f"pyscf {name}: E={e:.10f} eps={mf.mo_energy} ({time.time() - t:.1f}s)")
