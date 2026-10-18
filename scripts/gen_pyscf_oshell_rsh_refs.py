"""PySCF references for the OPEN-SHELL range-separated-hybrid pins
(crates/ferric-pbc/tests/validation_oshell_rsh.rs), mgoldey/ferric#354.

PySCF 2.13 pbc.dft.UKS (Gamma) and pbc.dft.KUKS (1x1x2 mesh) with AFTDF (the
attenuated K is only on AFTDF), UniformGrids MESH^3, conv_tol 1e-12,
cell.precision 1e-11, exxdiv ewald and none, HSE06 and CAM-B3LYP (omega in
Bohr^-1 as libxc gives it; ferric reads the same). STO-3G H is spelled out
digit for digit (common::pyscf_sto3g_h).

Cells (TRICLINIC TRI_A lattice of tests/common, NON-degenerate polarized):
  h3d: three H on the c axis, doublet (N_a, N_b) = (2, 1)
  h4t: the four H of TRI_ATOMS, triplet (N_a, N_b) = (3, 1)
For a k-mesh with N_k points `mf.nelec` is set to (N_a N_k, N_b N_k) (global
aufbau, the giant-determinant spin state; cell.spin cannot say this).

Each row also stores the S^2-free diagnostics PySCF reports (converged flag)
and the RSH-vs-PBE0 energy difference of the same cell as a check that the
long-range piece is not negligible there.

Output: testdata/reference/pbc_oshell_sto3g_rsh.json.

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/gen_pyscf_oshell_rsh_refs.py [--only SUBSTR]
"""

import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import pyscf
from pyscf.dft import libxc
from pyscf.pbc import df, dft, gto

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "testdata", "reference", "pbc_oshell_sto3g_rsh.json")

LATTICE = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
CELLS = {
    "h3d": ([[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]], (2, 1)),
    "h4t": (
        [
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 1.5],
            [1.6, 1.1, 2.9],
            [1.6, 1.1, 4.1],
        ],
        (3, 1),
    ),
}
STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
MESH = 41
PRECISION = 1e-11
FUNCTIONALS = {"HSE06": "HSE06", "CAM_B3LYP": "CAMB3LYP"}
FERRIC_NAMES = {"HSE06": "HSE06", "CAM_B3LYP": "HYB_GGA_XC_CAM_B3LYP"}
# (cell, functional, kmesh, exxdiv)
ROWS = [
    ("h3d", "HSE06", (1, 1, 1), "ewald"),
    ("h3d", "HSE06", (1, 1, 1), None),
    ("h3d", "CAM_B3LYP", (1, 1, 1), "ewald"),
    ("h3d", "HSE06", (1, 1, 2), "ewald"),
    ("h3d", "CAM_B3LYP", (1, 1, 2), "ewald"),
    ("h3d", "HSE06", (1, 1, 2), None),
    ("h4t", "HSE06", (1, 1, 1), "ewald"),
    ("h4t", "CAM_B3LYP", (1, 1, 1), "ewald"),
    ("h4t", "HSE06", (1, 1, 2), "ewald"),
]


def make_cell(name):
    atoms, (na, nb) = CELLS[name]
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in atoms]
    c.a = np.array(LATTICE)
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.spin = na - nb
    c.mesh = [MESH] * 3
    c.precision = PRECISION
    c.verbose = 0
    c.build()
    return c, (na, nb)


def run(c, nab, xc, kmesh, exxdiv):
    nk = int(np.prod(kmesh))
    if nk == 1:
        mf = dft.UKS(c, xc=xc)
        mf.with_df = df.AFTDF(c)
    else:
        kp = c.make_kpts(kmesh)
        mf = dft.KUKS(c, kpts=kp, xc=xc)
        mf.with_df = df.AFTDF(c, kpts=kp)
        mf.nelec = (nab[0] * nk, nab[1] * nk)
    mf.exxdiv = exxdiv
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf.e_tot


def git_sha():
    return subprocess.check_output(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    rows = []
    for cell, name, kmesh, exxdiv in ROWS:
        tag = f"{cell} {name} {kmesh} {exxdiv}"
        if only not in tag:
            continue
        c, nab = make_cell(cell)
        t0 = time.time()
        e = run(c, nab, FUNCTIONALS[name], kmesh, exxdiv)
        dt = time.time() - t0
        row = {
            "cell": cell,
            "functional": name,
            "kmesh": list(kmesh),
            "exxdiv": exxdiv or "none",
            "e_tot": e,
            "runtime_s": round(dt, 1),
        }
        if name == "HSE06" and exxdiv == "ewald":
            # same cell, global hybrid with the same semilocal-ish exchange family
            row["e_pbe0"] = run(c, nab, "PBE0", kmesh, exxdiv)
        print(f"{tag}: E {e:.12f}  ({dt:.0f} s) {row.get('e_pbe0', '')}", flush=True)
        rows.append(row)
    meta = {
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "date": datetime.date.today().isoformat(),
        "generator": "scripts/gen_pyscf_oshell_rsh_refs.py",
        "repo_sha_at_generation": git_sha(),
        "repo_sha_note": "HEAD of the worktree when the script ran: the PARENT of the "
        "commit that adds/changes this script, not the commit containing it",
        "libxc_version": libxc.libxc_version(),
        "xc_names": {
            k: {
                "pyscf": v,
                "ferric": FERRIC_NAMES[k],
                "omega_bohr_inv": libxc.rsh_coeff(v)[0],
                "c_lr": libxc.rsh_coeff(v)[1],
                "c_sr": libxc.rsh_coeff(v)[1] + libxc.rsh_coeff(v)[2],
            }
            for k, v in FUNCTIONALS.items()
        },
        "grids": f"XC grid (NumInt UniformGrids) and AFTDF mesh are both {MESH}^3; "
        f"cell.precision {PRECISION}",
        "cell": "H STO-3G (PySCF digits), Bohr, TRICLINIC lattice; h3d doublet, h4t triplet",
        "lattice": LATTICE,
        "cells": {k: {"atoms": v[0], "nab": list(v[1])} for k, v in CELLS.items()},
        "setup": f"pbc.dft.UKS (Gamma) / KUKS (nelec=(Na*Nk, Nb*Nk)), AFTDF, UniformGrids "
        f"{MESH}^3, conv_tol 1e-12",
        "rows": rows,
    }
    if only:
        print(json.dumps(meta["rows"], indent=1))
        return
    with open(OUT, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
