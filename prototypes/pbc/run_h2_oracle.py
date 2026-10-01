import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf

a = np.eye(3) * 4.0
atoms = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
pc = pgto.Cell(a=a, atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.build()
for name in ("AFTDF", "RSJK"):
    t = time.time()
    mf = pscf.RHF(pc, exxdiv=None)
    if name == "AFTDF":
        mf.with_df = pdf.AFTDF(pc)
        mf.with_df.mesh = [61] * 3
    else:
        mf.with_df = pdf.AFTDF(pc)
        mf.with_df.mesh = [61] * 3  # exact hcore
        mf = mf.jk_method("RS")
    mf.conv_tol = 1e-11
    e = mf.kernel()
    print(
        f"pyscf {name}: E={e:.10f} eps={mf.mo_energy} ({time.time() - t:.1f}s)",
        flush=True,
    )
cell = Cell(a, atoms, "sto-3g")
for rb, r2 in ((12, 13), (16, 18)):
    ints = build_integrals(cell, 3.0, rcut_bra=rb, rcut_2e=r2)
    e, eps, it = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], 2)
    print(f"ours w=3 rcut_bra={rb} rcut_2e={r2}: E={e:.10f}", flush=True)
