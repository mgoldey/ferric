"""PySCF pbc.dft.KRKS SCAN / R2SCAN references on a cell where they DIFFER at 1x1x1.

Issue #355: on the H2-in-a-box test cell (one occupied orbital => alpha = 0
everywhere) PySCF's SCAN and r2SCAN energies are identical at 1x1x1, so that
pin cannot tell the two functionals apart. This uses the 4-hydrogen triclinic
cell (two occupied orbitals, alpha != 0) of crates/ferric-pbc/tests/common
(TRI_A / TRI_ATOMS), STO-3G, gamma-centred meshes 1x1x1 and 1x1x2.

Setup mirrors tests/pbc_krks.rs: AFTDF + UniformGrids(61^3), exxdiv='ewald',
conv_tol 1e-12, pyscf 2.13. Prints a Rust const table.

Usage: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/gen_pyscf_pbc_mgga_refs.py
"""
import numpy as np
import pyscf
from pyscf.pbc import gto, df, dft
from pyscf.pbc.dft import gen_grid

A = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
ATOMS = [[0.1, 0.2, 0.3], [0.1, 0.2, 1.7], [2.4, 2.5, 2.2], [3.6, 2.9, 2.6]]


def run(xc, nk):
    cell = gto.Cell()
    cell.a = np.array(A)
    cell.atom = [("H", r) for r in ATOMS]
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.precision = 1e-13
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(nk, with_gamma_point=True)
    mf = dft.KRKS(cell, kpts=kpts)
    mf.xc = xc
    mf.grids = gen_grid.UniformGrids(cell)
    mf.grids.mesh = [61] * 3
    mf.exxdiv = "ewald"
    mf.with_df = df.AFTDF(cell, kpts=kpts)
    mf.with_df.mesh = [61] * 3
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    e = mf.kernel()
    assert mf.converged
    return e


if __name__ == "__main__":
    print("pyscf", pyscf.__version__)
    for nk in ([1, 1, 1], [1, 1, 2]):
        es = {xc: run(xc, nk) for xc in ("SCAN", "R2SCAN", "PBE")}
        for xc, e in es.items():
            print(f'    ("{xc}", {nk}, {e:.12f}),')
        print("  SCAN - R2SCAN =", es["SCAN"] - es["R2SCAN"])
