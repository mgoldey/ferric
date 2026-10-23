"""Reference stress of PySCF's k-point RKS (FFTDF on a uniform grid) for the
ferric k-point RKS stress pins (crates/ferric-pbc/tests/pbc_kxcstress.rs).

H2 in the a = 4 Bohr cube (atoms (0.3, 0.2, 0.1), (0.3, 0.2, 1.5)), STO-3G H,
1x1x2 gamma-centred mesh, exxdiv ewald, mesh 61^3, conv_tol 1e-12.
PySCF's krks_stress supports pure functionals on FFTDF/UniformGrids only.
The printed tensor is sigma = (1/Omega) dE/d(strain) in Hartree/Bohr^3; with
--fd the analytic value is also checked against the central finite difference
of PySCF's own energy under cell strain (atoms strained with the lattice).

The meta-GGA rows (SCAN, r2SCAN, TPSS; pbc_kxcstress_mgga.rs) are also written, in
full (3x3 sigma, E_tot), to testdata/reference/pbc_h2_sto3g_kks_stress_mgga.json.

Run:  python scripts/gen_pyscf_kks_stress_refs.py [--fd]
"""

import json
import os
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


PYSCF_VERSION = "2.13.1"  # the version cited by tests/pbc_kxcstress.rs
# ferric's "LDA" functional is SVWN (LDA_X + LDA_C_VWN) = PySCF "LDA,VWN"
# (PySCF's bare "LDA" is exchange only).
ROWS = (("LDA", "LDA,VWN"), ("PBE", "PBE"))
MGGA_ROWS = (("SCAN", "SCAN"), ("R2SCAN", "R2SCAN"), ("TPSS", "TPSS"))
MGGA_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_h2_sto3g_kks_stress_mgga.json",
)


def main(check_fd):
    import pyscf

    assert pyscf.__version__ == PYSCF_VERSION, (pyscf.__version__, PYSCF_VERSION)
    print(f"// PySCF {pyscf.__version__}; paste into PYSCF_STRESS (name, xx, zz)")
    rows = []
    mgga = {}
    for name, xc in ROWS + MGGA_ROWS:
        c = make_cell()
        mf = run(c, xc)
        s = krks_stress.kernel(kg.Gradients(mf))
        print(xc, "E", mf.e_tot)
        print(np.array2string(s, precision=9))
        if (name, xc) in MGGA_ROWS:
            mgga[name] = {"e_tot": float(mf.e_tot), "sigma": s.tolist()}
        else:
            rows.append(f'    ("{name}", {s[0, 0]:.9e}, {s[2, 2]:.9e}),')
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
    print("const PYSCF_STRESS: [(&str, f64, f64); 2] = [")
    print("\n".join(rows))
    print("];")
    meta = {
        "pyscf_version": PYSCF_VERSION,
        "cell": "H2 in the a = 4 Bohr cube, atoms (0.3,0.2,0.1),(0.3,0.2,1.5), STO-3G H",
        "mesh": "FFTDF UniformGrids 61^3, kmesh 1x1x2 gamma-centred, exxdiv ewald, conv_tol 1e-12",
        "unit": "sigma = (1/Omega) dE/d(strain), Hartree/Bohr^3",
        "rows": mgga,
    }
    with open(MGGA_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main("--fd" in sys.argv)
