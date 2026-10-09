"""A-priori a^-3 coefficients for the H2 box-limit sweeps (HF/MP2/dRPA, shifted convention), from the
molecular |r-r'|^2-kernel derivative (pbc_rpa.r2_kernel_c3).  Validation targets measured earlier:
HF STO-3G c3 = -10.0282 (Iteration 1), MP2 c3 = 0.671358 / 0.784083 (Iteration 3)."""

import sys

sys.path.insert(0, ".")
from pyscf import gto
from pbc_rpa import r2_kernel_c3

ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
for basis in sys.argv[1:] or ["sto-3g", "6-31g"]:
    mol = gto.M(atom=ATOMS, basis=basis, unit="B", cart=True, verbose=0)
    for h in (1e-3, 1e-4):
        c = r2_kernel_c3(mol, h=h)
        print(
            f"{basis} h={h:.0e}: c3 HF {c['hf']:.6f}  MP2 {c['mp2']:.6f}  dRPA {c['drpa']:.6f}",
            flush=True,
        )
