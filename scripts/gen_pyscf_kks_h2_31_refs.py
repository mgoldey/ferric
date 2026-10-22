"""PySCF k-point KS pins on the H2 / H cubic cells (a = 4 Bohr, atoms
(0.3, 0.2, 0.1) [+ (0.3, 0.2, 1.5)], STO-3G) on a 31^3 UniformGrids mesh,
for crates/ferric-pbc/tests/pbc_krks_2x2x1.rs (KRKS singlet H2 on 2x2x1,
Gamma-centred and shifted, PBE/PBE0).  AFTDF, exxdiv ewald, conv_tol 1e-12,
cell.precision 1e-10 (matches the 31^3 mesh of pbc_kks_open_ecp.rs).
Output: testdata/reference/pbc_kks_h2_31_sto3g.json.

Run: flock -o /tmp/ferric-compile.lock env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
     python -u scripts/gen_pyscf_kks_h2_31_refs.py
"""

import json
import os
import subprocess
import time

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto

STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
N = 31
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_kks_h2_31_sto3g.json",
)
SYSTEMS = {
    "H2t": ([[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]], 2),
    "Hd": ([[0.3, 0.2, 0.1]], 1),
    "H2s": ([[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]], 0),
}
KUKS_ROWS = []  # dropped: 31^3 PySCF LDA reached another SCF state
KRKS_ROWS = [("H2s", xc, (2, 2, 1), g) for xc in ("PBE", "PBE0") for g in (True, False)]


def cell_of(sys_, nk=1):
    atoms, spin = SYSTEMS[sys_]
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in atoms]
    c.a = np.eye(3) * 4.0
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.spin = spin * nk  # total over the mesh (PySCF), 2S = (na-nb)*nk
    c.precision = 1e-10
    c.mesh = [N] * 3
    c.verbose = 0
    c.build()
    return c


def kernel(cls, c, n, gamma, xc):
    kp = c.make_kpts(n, with_gamma_point=gamma)
    mf = cls(c, kpts=kp, xc=xc)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.with_df = df.AFTDF(c, kpts=kp)
    mf.grids = dft.gen_grid.UniformGrids(c)
    e = float(mf.kernel())
    assert mf.converged
    return e


def main():
    rows = []
    for s, xc, m in KUKS_ROWS:
        e = kernel(dft.KUKS, cell_of(s, int(np.prod(m))), m, True, xc)
        print("KUKS", s, xc, m, f"{e:.12f}", flush=True)
        rows.append(
            dict(
                method="KUKS",
                system=s,
                xc=xc,
                mesh=list(m),
                gamma_centred=True,
                energy=e,
            )
        )
    for s, xc, m, g in KRKS_ROWS:
        e = kernel(dft.KRKS, cell_of(s), m, g, xc)
        print("KRKS", s, xc, m, g, f"{e:.12f}", flush=True)
        rows.append(
            dict(
                method="KRKS", system=s, xc=xc, mesh=list(m), gamma_centred=g, energy=e
            )
        )
    here = os.path.dirname(os.path.abspath(__file__))
    meta = dict(
        pyscf=pyscf.__version__,
        numpy=np.__version__,
        libxc=__import__("pyscf.dft.libxc", fromlist=["x"]).libxc_version(),
        date=time.strftime("%Y-%m-%d"),
        generator_git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True
        ).strip(),
        setup=f"AFTDF, UniformGrids {N}^3, exxdiv=ewald, conv_tol=1e-12, cell.precision=1e-10, PySCF-digit STO-3G H, a=4 Bohr cubic",
        script="scripts/gen_pyscf_kks_h2_31_refs.py",
        rows=rows,
    )
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
