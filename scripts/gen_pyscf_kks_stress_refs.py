"""Reference stress of PySCF's k-point RKS (FFTDF on a uniform grid) for the
ferric k-point RKS stress pins (crates/ferric-pbc/tests/pbc_kxcstress.rs).

H2 in the a = 4 Bohr cube (atoms (0.3, 0.2, 0.1), (0.3, 0.2, 1.5)), STO-3G H,
1x1x2 gamma-centred mesh, exxdiv ewald, mesh 61^3, conv_tol 1e-12.
PySCF's krks_stress supports pure functionals on FFTDF/UniformGrids only.
The printed tensor is sigma = (1/Omega) dE/d(strain) in Hartree/Bohr^3; with
--fd the analytic value is also checked against the central finite difference
of PySCF's own energy under cell strain (atoms strained with the lattice).

Run:  python scripts/gen_pyscf_kks_stress_refs.py [--fd]
"""

import sys

import numpy as np
from pyscf.pbc import df, dft, gto
from pyscf.pbc.grad import krks as kg
from pyscf.pbc.grad import krks_stress

A = 4.0
ATOMS = np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]])
STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]


def make_cell(eps=np.zeros((3, 3))):
    lat = (np.eye(3) * A) @ (np.eye(3) + eps).T
    atoms = ATOMS @ (np.eye(3) + eps).T
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in atoms]
    c.a = lat
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.mesh = [61] * 3
    c.verbose = 0
    c.build()
    return c


def run(c, xc, n=(1, 1, 2)):
    kp = c.make_kpts(n)
    mf = dft.KRKS(c, kpts=kp, xc=xc)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.with_df = df.FFTDF(c, kpts=kp)
    mf.kernel()
    return mf


def main(check_fd):
    for xc in ("LDA,VWN", "PBE"):
        c = make_cell()
        mf = run(c, xc)
        s = krks_stress.kernel(kg.Gradients(mf))
        print(xc, "E", mf.e_tot)
        print(np.array2string(s, precision=9))
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
            print(xc, "max|analytic - FD|", np.abs(s - fd).max())


if __name__ == "__main__":
    main("--fd" in sys.argv)
