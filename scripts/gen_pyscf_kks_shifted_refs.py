"""Reference energies of PySCF's k-point RKS on SHIFTED (Monkhorst-Pack,
make_kpts(n, with_gamma_point=False)) meshes, for the ferric pins in
crates/ferric-pbc/tests/pbc_krks_shifted.rs.

Setup (matches pbc_krks.rs): STO-3G H, AFTDF, UniformGrids mesh^3, exxdiv
ewald, conv_tol 1e-12. Cells: H2 in the a = 4 Bohr cube (atoms (0.3,0.2,0.1),
(0.3,0.2,1.5)) mesh 61^3 and the 4-H triclinic cell of
reference/pbc/test_prototype.py (TRI_A / TRI_ATOMS) mesh 45^3.
Only N >= 3 along a shifted axis has non-TRIM q = k - k' (a shifted N = 2
mesh has q in {0, 1/2}, where e^{+iq.T} == e^{-iq.T}), so the 1x1x4 meshes are
the ones that can see a q-phase defect.
Output: testdata/reference/pbc_kks_shifted_sto3g.json (with provenance); one
cell per invocation, rows of the other cell are kept.

Run:  OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_kks_shifted_refs.py CELL
      (CELL = h2_cubic_a4 | h4_triclinic)
"""

import json
import os
import sys

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto

STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
CELLS = {
    "h2_cubic_a4": dict(
        a=np.eye(3) * 4.0,
        atoms=[[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]],
        mesh=61,
        meshes=[(1, 1, 2), (1, 1, 4)],
    ),
    "h4_triclinic": dict(
        a=np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]),
        atoms=[[0.1, 0.2, 0.3], [0.1, 0.2, 1.7], [2.4, 2.5, 2.2], [3.6, 2.9, 2.6]],
        mesh=45,
        meshes=[(1, 1, 2), (1, 1, 4)],
    ),
}
XCS = ("PBE", "PBE0")
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_kks_shifted_sto3g.json",
)


def make_cell(spec):
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in spec["atoms"]]
    c.a = spec["a"]
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.mesh = [spec["mesh"]] * 3
    c.verbose = 0
    c.build()
    return c


def main():
    name = sys.argv[1]
    spec = CELLS[name]
    c = make_cell(spec)
    rows = []
    for n in spec["meshes"]:
        for xc in XCS:
            kp = c.make_kpts(n, with_gamma_point=False)
            mf = dft.KRKS(c, kpts=kp, xc=xc)
            mf.exxdiv = "ewald"
            mf.conv_tol = 1e-12
            mf.with_df = df.AFTDF(c, kpts=kp)
            mf.grids = dft.gen_grid.UniformGrids(c)
            e = mf.kernel()
            assert mf.converged
            print(name, n, xc, f"{e:.12f}", flush=True)
            rows.append(
                dict(cell=name, mesh=list(n), xc=xc, energy=float(e), nk=len(kp))
            )
    old = json.load(open(OUT))["rows"] if os.path.exists(OUT) else []
    rows = [r for r in old if r["cell"] != name] + rows
    meta = dict(
        pyscf=pyscf.__version__,
        setup="KRKS, AFTDF, UniformGrids, exxdiv=ewald, conv_tol=1e-12, "
        "make_kpts(n, with_gamma_point=False), STO-3G H, units Bohr/Hartree",
        cells={
            k: dict(a=v["a"].tolist(), atoms=v["atoms"], mesh=v["mesh"])
            for k, v in CELLS.items()
        },
        script="scripts/gen_pyscf_kks_shifted_refs.py",
        rows=rows,
    )
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
