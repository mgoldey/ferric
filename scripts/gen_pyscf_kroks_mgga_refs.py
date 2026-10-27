"""PySCF pbc.dft.KROKS SCAN / R2SCAN energies on a POLARIZED cell, the oracle
of ferric's k-point ROKS meta-GGA energy (crates/ferric-pbc/tests/
pbc_kroks_mgga.rs, issue #492).

PySCF's KROKS takes the k-point UKS veff (pbc/dft/kroks.py get_veff calls
kuks.get_veff with the ROHF alpha / beta occupations), so its meta-GGA is the
per-spin-tau KUKS one, applied to the restricted-open-shell spin densities.

Cell and basis are those of scripts/gen_pyscf_kuks_mgga_rsgdf_refs.py: three
H in the cubic a = 4 Bohr cell, doublet (N_a = 2, N_b = 1), atoms
(0.3,0.2,0.1), (0.35,0.12,1.5), (0.3,1.9,0.4) Bohr, one s per H (exponent
0.5). Gamma-centred meshes 1x1x1 and 1x1x3. FFTDF with cell.mesh 40^3, XC on
UniformGrids 32^3 (the ferric test's uniform grid), exxdiv ewald (irrelevant:
no exact exchange), conv_tol 1e-12, cell.precision 1e-12,
mf.nelec = (N_a N_k, N_b N_k) (PySCF's default would split the 3 N_k
electrons by cell.spin = 1, not by N_k).

The KUKS energy of the same setup is written alongside as a sanity row: the
ROKS energy is variationally above it (same functional, constrained orbitals).

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python \
           scripts/gen_pyscf_kroks_mgga_refs.py
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
    "pbc_h3_s05_kroks_mgga.json",
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


def run(cls, xc, kmesh):
    c = make_cell()
    kp = c.make_kpts(kmesh, with_gamma_point=True)
    mf = cls(c, kpts=kp, xc=xc)
    mf.grids = gen_grid.UniformGrids(c)
    mf.grids.mesh = [XC_MESH] * 3
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.with_df = df.FFTDF(c, kpts=kp)
    mf.nelec = (NA * len(kp), NB * len(kp))
    mf.kernel()
    assert mf.converged, (cls.__name__, xc, kmesh)
    return float(mf.e_tot)


def main():
    import pyscf

    assert pyscf.__version__ == PYSCF_VERSION, (pyscf.__version__, PYSCF_VERSION)
    roks, uks = {}, {}
    for kmesh in ([1, 1, 1], [1, 1, 3]):
        for xc in ("SCAN", "R2SCAN"):
            key = f"{xc} {kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
            roks[key] = run(dft.KROKS, xc, kmesh)
            uks[key] = run(dft.KUKS, xc, kmesh)
            print(
                f"{key}: KROKS {roks[key]:.12f} KUKS {uks[key]:.12f} "
                f"ROKS - UKS {roks[key] - uks[key]:.3e}"
            )
            assert roks[key] > uks[key], key
    meta = {
        "pyscf_version": PYSCF_VERSION,
        "cell": "H3 doublet (N_a=2, N_b=1) in the a = 4 Bohr cube, atoms "
        "(0.3,0.2,0.1),(0.35,0.12,1.5),(0.3,1.9,0.4) Bohr, one s per H (alpha 0.5)",
        "settings": "KROKS (and KUKS), FFTDF cell.mesh 40^3, UniformGrids 32^3, "
        "gamma-centred kmesh, exxdiv ewald, conv_tol 1e-12, cell.precision 1e-12, "
        "nelec (2 N_k, N_k)",
        "unit": "e_tot per cell, Hartree",
        "e_tot": roks,
        "e_tot_kuks": uks,
    }
    with open(OUT, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
