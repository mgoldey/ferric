"""H2/STO-3G cubic a=4: pure-AFT prototype vs PySCF AFTDF, exxdiv None and 'ewald'."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf

a = np.eye(3) * 4.0
atoms = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
cell = Cell(a, atoms, "sto-3g")
t = time.time()
ints = build_integrals(cell, None, exxdiv="ewald")
print(f"build {time.time() - t:.1f}s  v_M={ints['madelung']:.12f}")
ours = {}
for ex, ks in (("none", 0.0), ("ewald", ints["madelung"])):
    e, eps, it = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], 2, kshift=ks)
    ours[ex] = (e, eps)
    print(f"ours  exxdiv={ex}: E={e:.12f} eps={eps}", flush=True)

pc = pgto.Cell(a=a, atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.build()
for ex in (None, "ewald"):
    t = time.time()
    mf = pscf.RHF(pc, exxdiv=ex)
    mf.with_df = pdf.AFTDF(pc)
    mf.with_df.mesh = [61] * 3
    mf.conv_tol = 1e-12
    e = mf.kernel()
    k = "none" if ex is None else ex
    print(
        f"pyscf AFTDF exxdiv={ex}: E={e:.12f} eps={mf.mo_energy} ({time.time() - t:.1f}s)  "
        f"dE={ours[k][0] - e:.2e} deps={abs(ours[k][1] - mf.mo_energy).max():.2e}",
        flush=True,
    )
