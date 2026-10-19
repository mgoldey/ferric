"""Reference stress of PySCF's k-point UKS for the pure meta-GGAs SCAN and
r2SCAN on a genuinely POLARIZED cell, for the ferric k-point UKS meta-GGA
stress pins (crates/ferric-pbc/tests/validation_pbc_kuxc_mgga.rs, issue #353).

PySCF 2.13.1 `pbc.grad.kuks_stress` supports MGGA on FFTDF + UniformGrids
(its `pbc.grad.kuks` FORCE refuses meta-GGA, so there is no force pin).
Settings mirror the closed-shell meta-GGA anchor
(scripts/gen_pyscf_kks_stress_refs.py): FFTDF on UniformGrids(61^3),
exxdiv ewald (irrelevant: no exact exchange), conv_tol 1e-12, gamma-centred
1x1x2 mesh, plus cell.precision 1e-12 and mf.nelec = (N_a N_k, N_b N_k) as in
scripts/gen_pyscf_kuks_mgga_refs.py.

Cell: three H in the cubic a = 4 Bohr cell (doublet, N_a = 2, N_b = 1), atoms
(0.3,0.2,0.1), (0.35,0.12,1.5), (0.3,1.9,0.4), STO-3G H (PySCF's digits).

Written per functional: E_tot, the full 3x3 sigma = (1/Omega) dE/d(strain)
(Hartree/Bohr^3, kuks_stress.kernel), and the XC-only fixed-density part
dE_xc/d(strain) (Hartree, kuks_stress.get_vxc with no J and no nuclear term)
at PySCF's converged density. With --fd the analytic sigma is also checked
against the central finite difference of PySCF's own energy under cell
strain (atoms strained with the lattice), h = 1e-4.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python \
         scripts/gen_pyscf_kuks_mgga_stress_refs.py [--fd]
"""

import json
import os
import sys

import numpy as np
from pyscf.pbc import df, dft, gto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.grad import kuks as kuks_grad
from pyscf.pbc.grad import kuks_stress

A = 4.0
ATOMS = np.array([[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]])
NA, NB = 2, 1
KMESH = (1, 1, 2)
MESH = 61
STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
PYSCF_VERSION = "2.13.1"
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_h3_sto3g_kuks_stress_mgga.json",
)


def make_cell(eps=np.zeros((3, 3))):
    lat = (np.eye(3) * A) @ (np.eye(3) + eps).T
    atoms = ATOMS @ (np.eye(3) + eps).T
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in atoms]
    c.a = lat
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.spin = NA - NB
    c.mesh = [MESH] * 3
    c.precision = 1e-12
    c.verbose = 0
    c.build()
    return c


def run(c, xc):
    kp = c.make_kpts(KMESH, with_gamma_point=True)
    mf = dft.KUKS(c, kpts=kp, xc=xc)
    mf.grids = gen_grid.UniformGrids(c)
    mf.grids.mesh = [MESH] * 3
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.with_df = df.FFTDF(c, kpts=kp)
    mf.nelec = (NA * len(kp), NB * len(kp))
    mf.kernel()
    assert mf.converged, xc
    return mf


def main(check_fd):
    import pyscf

    assert pyscf.__version__ == PYSCF_VERSION, (pyscf.__version__, PYSCF_VERSION)
    rows = {}
    for xc in ("SCAN", "R2SCAN"):
        c = make_cell()
        mf = run(c, xc)
        g = kuks_grad.Gradients(mf)
        s = kuks_stress.kernel(g)
        dm = np.asarray(mf.make_rdm1())
        xc_part = kuks_stress.get_vxc(g, c, dm, mf.kpts, with_j=False, with_nuc=False)
        print(xc, "E", mf.e_tot, "vol", c.vol)
        print("sigma\n", np.array2string(s, precision=12))
        print("xc dE/deps\n", np.array2string(np.asarray(xc_part), precision=12))
        row = {
            "e_tot": float(mf.e_tot),
            "sigma": np.asarray(s).tolist(),
            "xc_de_deps": np.asarray(xc_part).real.tolist(),
        }
        if check_fd:
            h = 1e-4
            fd = np.zeros((3, 3))
            for a in range(3):
                for b in range(3):
                    ep = np.zeros((3, 3))
                    ep[a, b] = h
                    e1 = run(make_cell(ep), xc).e_tot
                    e2 = run(make_cell(-ep), xc).e_tot
                    fd[a, b] = (e1 - e2) / (2 * h) / c.vol
            d = float(np.abs(s - fd).max())
            print(xc, "max|analytic - FD| (PySCF self-consistency)", d)
            row["pyscf_analytic_vs_fd"] = d
        rows[xc] = row
    meta = {
        "pyscf_version": PYSCF_VERSION,
        "cell": "H3 doublet (N_a=2, N_b=1) in the a = 4 Bohr cube, atoms "
        "(0.3,0.2,0.1),(0.35,0.12,1.5),(0.3,1.9,0.4), STO-3G H",
        "settings": "KUKS, FFTDF + UniformGrids 61^3, kmesh 1x1x2 gamma-centred, "
        "exxdiv ewald, conv_tol 1e-12, cell.precision 1e-12, nelec (2 N_k, N_k)",
        "unit": "sigma = (1/Omega) dE/d(strain), Hartree/Bohr^3; xc_de_deps = "
        "fixed-density dE_xc/d(strain), Hartree",
        "rows": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main("--fd" in sys.argv)
