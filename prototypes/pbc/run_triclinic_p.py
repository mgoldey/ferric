import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf
from pyscf import gto
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf

# triclinic cell, 2 H2 units, basis with a p shell (exercises E^{ij}_t for l>0)
a = np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]])
atoms = [
    ("H", (0.1, 0.2, 0.3)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (2.4, 2.5, 2.2)),
    ("H", (3.6, 2.9, 2.6)),
]
basis = {
    "H": gto.parse("""
H S
  3.42525091  0.15432897
  0.62391373  0.53532814
  0.16885540  0.44463454
H P
  0.8         1.0
""")
}
t = time.time()
pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.build()
mf = pscf.RHF(pc, exxdiv=None)
mf.with_df = pdf.AFTDF(pc)
mf.with_df.mesh = [61] * 3
mf.conv_tol = 1e-11
e_ref = mf.kernel()
print(
    f"pyscf AFTDF: E={e_ref:.10f} eps_occ={mf.mo_energy[:2]} ({time.time() - t:.0f}s)",
    flush=True,
)
cell = Cell(a, atoms, basis)
ints = build_integrals(cell, None)
e, eps, it = rhf(ints["S"], ints["h"], ints["I"], ints["enn"], 4)
print(f"ours pure-AFT: E={e:.10f} eps_occ={eps[:2]}  dE={e - e_ref:.2e}", flush=True)
