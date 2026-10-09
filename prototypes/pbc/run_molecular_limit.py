"""Independent check of the G=0/exchange convention: box-size sweep vs molecular RHF.

Pure-AFT (w=None) periodic H2/STO-3G in cubic boxes of edge a; E_pbc - E_mol for
exxdiv none vs ewald. Molecular reference: pyscf.scf.RHF, cart=True, same geometry."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf
from pyscf import gto, scf

atoms = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_mol = mf.kernel()
print(f"E_mol = {e_mol:.12f}", flush=True)
edges = [float(x) for x in sys.argv[1:]] or [6, 8, 10, 12, 14]
for a in edges:
    t = time.time()
    cell = Cell(np.eye(3) * a, atoms, "sto-3g")
    ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    e0 = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], 2)[0]
    e1 = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], 2, kshift=ints["madelung"])[
        0
    ]
    print(
        f"a={a:5.1f} nG={len(ints['G']):8d} v_M={ints['madelung']:.10f} "
        f"none: {e0 - e_mol:+.10e}  ewald: {e1 - e_mol:+.10e}  ({time.time() - t:.0f}s)",
        flush=True,
    )
