"""Reference energies of PySCF's k-point UKS / ROKS (and a KROHF probe) on
SHIFTED Monkhorst-Pack meshes of a triclinic open-shell cell, and of k-point
RKS on an ECP (core-electron) cell, for the ferric pins in
crates/ferric-pbc/tests/pbc_kks_open_ecp.rs (issue #356 validation breadth).

Setup (matches pbc_krks_shifted.rs): AFTDF, UniformGrids mesh^3 (the same
discretisation ferric's PeriodicGrid::uniform builds, so any mesh size is a
valid pin; it need not converge the XC integral), exxdiv ewald, conv_tol
1e-12, make_kpts(n, with_gamma_point=False).

Cells:
  h3_triclinic  first 3 atoms of the 4-H triclinic cell of the shifted-mesh
                pins (TRI_A), STO-3G H, doublet, mesh 31^3. KUKS and KROKS
                (mf.nelec = (2 Nk, Nk)) PBE and PBE0, shifted 1x1x2 and 1x1x4.
                A KROHF probe is recorded with its converged flag: PySCF's own
                KROHF occupation rule cycles on some systems (H3 cubic 1x1x2).
  hi_ecp        HI, H STO-3G + I LANL2DZ (explicit digits) + LANL2DZ ECP
                (46 core electrons), diag(6, 6, 7) Bohr, H (0.3,0.2,0.4),
                I (0.3,0.2,3.44), mesh 45^3, KRKS PBE and PBE0, 1x1x2 shifted.
                NOTE PySCF's periodic ECP integral is only converged at cell.precision
                1e-12 (default 1e-8: 2.25e-6 max-element error); the remaining
                ferric-PySCF gap is ~1e-6 Ha (see pbc_kks_open_ecp.rs).

Output: testdata/reference/pbc_kks_open_ecp_sto3g.json (with provenance); one
cell per invocation, rows of the other cell are kept.

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/gen_pyscf_kks_open_ecp_refs.py CELL
      (CELL = h3_triclinic | hi_ecp)
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto, scf

STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
LANL2DZ_I = [
    [0, [0.7242, -2.9731048], [0.4653, 3.4827643]],
    [0, [0.1336, 1.0]],
    [1, [1.29, -0.2092377], [0.318, 1.1035347]],
    [1, [0.1053, 1.0]],
]
TRI_A = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
TRI_ATOMS = [[0.1, 0.2, 0.3], [0.1, 0.2, 1.7], [2.4, 2.5, 2.2], [3.6, 2.9, 2.6]]
CELLS = {
    "h3_triclinic": dict(
        a=TRI_A,
        atoms=[["H", x] for x in TRI_ATOMS[:3]],
        mesh=31,
        spin=1,
        meshes=[(1, 1, 2), (1, 1, 4)],
        xcs=("PBE", "PBE0"),
        methods=("KUKS", "KROKS"),
    ),
    "hi_ecp": dict(
        a=np.diag([6.0, 6.0, 7.0]).tolist(),
        atoms=[["H", [0.3, 0.2, 0.4]], ["I", [0.3, 0.2, 3.44]]],
        mesh=45,
        spin=0,
        meshes=[(1, 1, 2)],
        xcs=("PBE", "PBE0"),
        methods=("KRKS",),
    ),
}
# cell.precision drives PySCF's AFTDF cutoff AND its periodic ECP lattice sum
# (default 1e-8 leaves the ECP integral 2.25e-6 off its converged value).
# 1e-12 and 1e-14 agree (see the NOTE above), so 1e-12 is the converged oracle.
PRECISION = 1e-12
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "testdata",
    "reference",
    "pbc_kks_open_ecp_sto3g.json",
)


def make_cell(spec):
    c = gto.Cell()
    c.atom = [[s, tuple(x)] for s, x in spec["atoms"]]
    c.a = np.array(spec["a"])
    c.unit = "B"
    c.basis = {"H": STO3G_H, "I": LANL2DZ_I}
    if any(s == "I" for s, _ in spec["atoms"]):
        c.ecp = {"I": "lanl2dz"}
    c.spin = spec["spin"]
    c.precision = PRECISION
    c.mesh = [spec["mesh"]] * 3
    c.verbose = 0
    c.build()
    return c


def git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=here, text=True
    ).strip()


def make_mf(c, method, xc, kp):
    if method == "KROHF":
        mf = scf.KROHF(c, kpts=kp)
    else:
        mf = getattr(dft, method)(c, kpts=kp, xc=xc)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.with_df = df.AFTDF(c, kpts=kp)
    if method != "KROHF":
        mf.grids = dft.gen_grid.UniformGrids(c)
    if c.spin:
        na = (c.nelectron + c.spin) // 2
        mf.nelec = (na * len(kp), (c.nelectron - na) * len(kp))
    return mf


def main():
    name = sys.argv[1]
    spec = CELLS[name]
    c = make_cell(spec)
    rows = []
    for n in spec["meshes"]:
        kp = c.make_kpts(n, with_gamma_point=False)
        combos = [(m, xc) for m in spec["methods"] for xc in spec["xcs"]]
        if name == "h3_triclinic":
            combos.append(("KROHF", "HF"))
        for method, xc in combos:
            mf = make_mf(c, method, xc, kp)
            t0 = time.time()
            e = mf.kernel()
            dt = time.time() - t0
            print(name, n, method, xc, f"{e:.12f}", mf.converged, flush=True)
            rows.append(
                dict(
                    cell=name,
                    mesh=list(n),
                    method=method,
                    xc=xc,
                    energy=float(e),
                    converged=bool(mf.converged),
                    nk=len(kp),
                    seconds=round(dt),
                )
            )
    old = json.load(open(OUT))["rows"] if os.path.exists(OUT) else []
    rows = [r for r in old if r["cell"] != name] + rows
    from pyscf.dft import libxc

    meta = dict(
        pyscf=pyscf.__version__,
        numpy=np.__version__,
        libxc=libxc.libxc_version(),
        date=time.strftime("%Y-%m-%d"),
        generator_git_sha=git_sha(),
        setup=f"AFTDF, UniformGrids, exxdiv=ewald, conv_tol=1e-12, cell.precision={PRECISION}, "
        "make_kpts(n, with_gamma_point=False), STO-3G H, units Bohr/Hartree",
        cells={
            k: dict(a=v["a"], atoms=v["atoms"], mesh=v["mesh"], spin=v["spin"])
            for k, v in CELLS.items()
        },
        script="scripts/gen_pyscf_kks_open_ecp_refs.py",
        rows=rows,
    )
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
