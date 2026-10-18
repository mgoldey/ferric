"""PySCF pbc.dft.KUKS SCAN / R2SCAN references on genuinely POLARIZED cells
(issue #355, pins for crates/ferric-pbc/tests/pbc_kuks_mgga.rs).

Cells (both have a spin channel with >= 2 occupied orbitals, so alpha != 0 and
SCAN != r2SCAN at 1x1x1; neither is near-degenerate):
  h3   three H in the cubic a = 4 Bohr cell (doublet, N_a = 2, N_b = 1)
  h4t  the four-H triclinic cell of crates/ferric-pbc/tests/common (TRI_A /
       TRI_ATOMS), triplet (N_a = 3, N_b = 1)
STO-3G H (PySCF's digits), gamma-centred meshes 1x1x1 and 1x1x2.

Setup mirrors tests/pbc_kuks.rs / pbc_krks.rs: AFTDF + UniformGrids(61^3),
exxdiv='ewald', conv_tol 1e-12, mf.nelec = (N_a N_k, N_b N_k), but
cell.precision = 1e-12 (the 1e-8 default is known to leave 1e-10..1e-6
errors). A PBE control row on the existing pbc_kuks.rs 'Hd' pin checks the
nelec / setup convention.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/gen_pyscf_kuks_mgga_refs.py
"""

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto
from pyscf.pbc.dft import gen_grid

TRI_A = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
TRI_ATOMS = [[0.1, 0.2, 0.3], [0.1, 0.2, 1.7], [2.4, 2.5, 2.2], [3.6, 2.9, 2.6]]
H3_ATOMS = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]]
CELLS = {
    "h3": (np.diag([4.0] * 3).tolist(), H3_ATOMS, (2, 1)),
    "h4t": (TRI_A, TRI_ATOMS, (3, 1)),
    "hd": (np.diag([4.0] * 3).tolist(), [[0.3, 0.2, 0.1]], (1, 0)),
}


def run(name, xc, nk, mesh=61):
    a, atoms, (na, nb) = CELLS[name]
    cell = gto.Cell()
    cell.a = np.array(a)
    cell.atom = [("H", r) for r in atoms]
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.spin = na - nb
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(nk, with_gamma_point=True)
    mf = dft.KUKS(cell, kpts=kpts)
    mf.xc = xc
    mf.grids = gen_grid.UniformGrids(cell)
    mf.grids.mesh = [mesh] * 3
    mf.exxdiv = "ewald"
    mf.with_df = df.AFTDF(cell, kpts=kpts)
    mf.with_df.mesh = [mesh] * 3
    mf.nelec = (na * len(kpts), nb * len(kpts))
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    e = mf.kernel()
    assert mf.converged, (name, xc, nk)
    return e


if __name__ == "__main__":
    print("pyscf", pyscf.__version__)
    # control: existing pbc_kuks.rs pin ("Hd", "PBE", [1,1,2]) = -0.586191001233
    print(
        "control Hd PBE 1x1x2",
        f"{run('hd', 'PBE', [1, 1, 2]):.12f}",
        "(pin -0.586191001233)",
    )
    for name in ("h3", "h4t"):
        for nk in ([1, 1, 1], [1, 1, 2]):
            es = {xc: run(name, xc, nk) for xc in ("SCAN", "R2SCAN")}
            for xc, e in es.items():
                print(f'    ("{name}", "{xc}", {nk}, {e:.12f}),')
            print("  SCAN - R2SCAN =", es["SCAN"] - es["R2SCAN"])
