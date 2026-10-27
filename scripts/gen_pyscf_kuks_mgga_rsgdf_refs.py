"""PySCF pbc.dft.KUKS SCAN / R2SCAN energies on a POLARIZED cell with a
one-s-per-H basis, the oracle of the ferric RS-GDF open-shell meta-GGA energy
at its exact-aux limit (crates/ferric-pbc/tests/pbc_kuks_mgga_rsgdf.rs,
issue #492).

Why this cell and basis: with ONE primitive s per H (exponent 0.5) every pair
density is an s Gaussian (exponent 1.0) at a pair midpoint, so ferric's aux of
s functions at all pair midpoints (8 half-lattice classes) spans the pair
densities and the RS-GDF fit is EXACT. ferric's RS-GDF energy must then equal
PySCF's exact-J (FFTDF) energy on the same XC grid to SCF noise. PySCF has no
analytic k-point UKS meta-GGA force, and ferric's k-point RS-GDF derivative
needs atom-centred aux, so only the energy is pinned here.

Cell: three H in the cubic a = 4 Bohr cell, doublet (N_a = 2, N_b = 1), atoms
(0.3,0.2,0.1), (0.35,0.12,1.5), (0.3,1.9,0.4) Bohr. Basis {H: [[0, [0.5,
1.0]]]}. Gamma-centred meshes 1x1x1 and 1x1x3. FFTDF with cell.mesh 40^3
(the energy is unchanged to 1e-12 at 32/40/48), XC on UniformGrids 32^3 (the
ferric test's uniform grid), exxdiv ewald (irrelevant: no exact exchange),
conv_tol 1e-12, cell.precision 1e-12, mf.nelec = (N_a N_k, N_b N_k).

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python \
           scripts/gen_pyscf_kuks_mgga_rsgdf_refs.py
"""

import json
import os

import numpy as np
from pyscf.pbc import df, dft, gto
from pyscf.pbc.dft import gen_grid

A = 4.0
ATOMS = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]]
NA, NB = 2, 1
ALPHA = 0.5
J_MESH = 40
XC_MESH = 32
PYSCF_VERSION = "2.13.1"
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_h3_s05_kuks_mgga_rsgdf.json",
)


def make_cell():
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in ATOMS]
    c.a = np.eye(3) * A
    c.unit = "B"
    c.basis = {"H": [[0, [ALPHA, 1.0]]]}
    c.spin = NA - NB
    c.mesh = [J_MESH] * 3
    c.precision = 1e-12
    c.verbose = 0
    c.build()
    return c


def run(xc, kmesh):
    c = make_cell()
    kp = c.make_kpts(kmesh, with_gamma_point=True)
    mf = dft.KUKS(c, kpts=kp, xc=xc)
    mf.grids = gen_grid.UniformGrids(c)
    mf.grids.mesh = [XC_MESH] * 3
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.with_df = df.FFTDF(c, kpts=kp)
    mf.nelec = (NA * len(kp), NB * len(kp))
    mf.kernel()
    assert mf.converged, (xc, kmesh)
    return float(mf.e_tot)


def main():
    import pyscf

    assert pyscf.__version__ == PYSCF_VERSION, (pyscf.__version__, PYSCF_VERSION)
    rows = {}
    for kmesh in ([1, 1, 1], [1, 1, 3]):
        for xc in ("SCAN", "R2SCAN"):
            e = run(xc, kmesh)
            key = f"{xc} {kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
            print(f"{key}: {e:.12f}")
            rows[key] = e
    meta = {
        "pyscf_version": PYSCF_VERSION,
        "cell": "H3 doublet (N_a=2, N_b=1) in the a = 4 Bohr cube, atoms "
        "(0.3,0.2,0.1),(0.35,0.12,1.5),(0.3,1.9,0.4) Bohr, one s per H (alpha 0.5)",
        "settings": "KUKS, FFTDF cell.mesh 40^3, UniformGrids 32^3, gamma-centred "
        "kmesh, exxdiv ewald, conv_tol 1e-12, cell.precision 1e-12, nelec (2 N_k, N_k)",
        "unit": "e_tot per cell, Hartree",
        "e_tot": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
