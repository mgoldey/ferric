"""Reference energies of PySCF's k-point RKS on an ALL-ELECTRON cell with a
1s core (LiH, STO-3G, no ECP), for the ferric pins in
crates/ferric-pbc/tests/validation_pbc_kks_core.rs.

Setup (matches gen_pyscf_kks_shifted_refs.py): AFTDF, UniformGrids mesh^3,
exxdiv ewald, conv_tol 1e-12, cell.precision 1e-10. Mesh: Gamma-centred 1x1x1
only (an 80^3 AFT J mesh makes each row ~30 min; see DF_MESH). The Li 1s primitive
(alpha = 16.1) is the sharpest function in any pin of this series.
Output: testdata/reference/pbc_kks_core_sto3g.json (with provenance).

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/gen_pyscf_kks_core_refs.py
"""

import json
import os
import subprocess
import time

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto

BASIS = {
    "H": [
        [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
    ],
    "Li": gto.basis.load("sto-3g", "Li"),
}
CELL = dict(
    a=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.4, 0.3, 6.0]],
    atoms=[["Li", [0.3, 0.2, 0.4]], ["H", [0.3, 0.2, 3.4]]],
    mesh=35,
)
DF_MESH = 80
MESHES = [((1, 1, 1), True)]
XCS = ("PBE",)
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_kks_core_sto3g.json",
)


def make_cell(mesh=None, precision=1e-10):
    c = gto.Cell()
    c.atom = [[s, tuple(x)] for s, x in CELL["atoms"]]
    c.a = np.array(CELL["a"])
    c.unit = "B"
    c.basis = BASIS
    c.precision = precision
    c.mesh = [mesh or CELL["mesh"]] * 3
    c.verbose = 0
    c.build()
    return c


def git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=here, text=True
    ).strip()


def run(c, n, gamma, xc):
    kp = c.make_kpts(n, with_gamma_point=gamma)
    mf = dft.KRKS(c, kpts=kp, xc=xc)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.with_df = df.AFTDF(c, kpts=kp)
    # PySCF's default AFT mesh (cell.mesh) under-resolves the Li 1s pair
    # density: PBE moves 4.8e-3 Ha from 35^3 to 80^3 (ferric's dense AFT is
    # precision-controlled and sits 6.5e-9 from the 80^3 value).
    mf.with_df.mesh = [DF_MESH] * 3
    mf.grids = dft.gen_grid.UniformGrids(c)
    e = mf.kernel()
    assert mf.converged
    return float(e), len(kp)


def main():
    c = make_cell()
    rows = []
    for n, gamma in MESHES:
        for xc in XCS:
            t0 = time.time()
            e, nk = run(c, n, gamma, xc)
            print(n, gamma, xc, f"{e:.12f}", flush=True)
            rows.append(
                dict(
                    cell="lih_core",
                    mesh=list(n),
                    gamma_centred=gamma,
                    xc=xc,
                    energy=e,
                    nk=nk,
                    seconds=round(time.time() - t0),
                )
            )
    meta = dict(
        pyscf=pyscf.__version__,
        numpy=np.__version__,
        libxc=__import__("pyscf.dft.libxc", fromlist=["x"]).libxc_version(),
        date=time.strftime("%Y-%m-%d"),
        generator_git_sha=git_sha(),
        setup="KRKS, AFTDF, UniformGrids, exxdiv=ewald, conv_tol=1e-12, "
        "cell.precision=1e-10, AFT J mesh 80^3 (XC grid 35^3), PySCF sto-3g Li/H (all electron), Bohr/Hartree",
        cells={"lih_core": CELL},
        script="scripts/gen_pyscf_kks_core_refs.py",
        rows=rows,
    )
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
