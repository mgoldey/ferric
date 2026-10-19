"""PySCF pbc.dft TPSS references (issue #355) on the 4-hydrogen triclinic cell.

Same cell, basis and integration settings as scripts/gen_pyscf_pbc_mgga_refs.py
(TRI_A / TRI_ATOMS of crates/ferric-pbc/tests/common, STO-3G, AFTDF +
UniformGrids 61^3, exxdiv='ewald', conv_tol 1e-12, cell.precision 1e-13),
xc = 'TPSS' (libxc MGGA_X_TPSS + MGGA_C_TPSS). Runs the single-Gamma-point
pbc.dft.RKS and pbc.dft.KRKS at gamma-centred 1x1x1 and 1x1x2 meshes. Prints a
Rust const table.

Usage: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/gen_pyscf_pbc_tpss_refs.py
"""

import pyscf
from pyscf.dft import libxc

from gen_pyscf_pbc_mgga_refs import run, run_gamma

if __name__ == "__main__":
    print("pyscf", pyscf.__version__, "libxc", libxc.__version__)
    print(f'    ("TPSS", {run_gamma("TPSS"):.12f}),  # pbc.dft.RKS at Gamma')
    for nk in ([1, 1, 1], [1, 1, 2]):
        print(f'    ("TPSS", {nk}, {run("TPSS", nk):.12f}),')
