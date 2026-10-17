"""PySCF references for the Gamma-point range-separated-hybrid pins
(crates/ferric-pbc/tests/pbc_grsh.rs), the Gamma counterpart of
gen_pyscf_pbc_krsh_refs.py (mgoldey/ferric#354).

PySCF 2.13 pbc.dft.RKS (Gamma point, no k-points) with AFTDF (the attenuated K
is only on AFTDF), UniformGrids MESH^3, conv_tol 1e-12, cell.precision 1e-11
(PySCF's default 1e-8 left a -1.5e-10 offset in the k-point pins). STO-3G H is
spelled out digit for digit (common::pyscf_sto3g_h). Cells: H2 in a CUBIC
a = 4 Bohr lattice and in the TRICLINIC TRI_A lattice (atoms H2_ATOMS); HSE06
and CAM-B3LYP; exxdiv ewald and none.

Output: testdata/reference/pbc_h2_sto3g_gamma_rsh.json (provenance: date,
versions, SHA of the repo HEAD at generation = the PARENT of the commit that
adds this script, per-row runtime).

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/gen_pyscf_pbc_grsh_refs.py [--only SUBSTR]
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
OUT = os.path.join(ROOT, "testdata", "reference", "pbc_h2_sto3g_gamma_rsh.json")

LATTICES = {
    "cubic4": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    "triclinic": [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]],
}
ATOMS = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]]
STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
MESH = 41
PRECISION = 1e-11
FUNCTIONALS = {"HSE06": "HSE06", "CAM_B3LYP": "CAMB3LYP"}
FERRIC_NAMES = {"HSE06": "HSE06", "CAM_B3LYP": "HYB_GGA_XC_CAM_B3LYP"}
# (cell, functional, exxdiv)
ROWS = [(cell, f, x) for cell in LATTICES for f in FUNCTIONALS for x in ("ewald", None)]


def make_cell(lat):
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in ATOMS]
    c.a = np.array(lat)
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.mesh = [MESH] * 3
    c.precision = PRECISION
    c.verbose = 0
    c.build()
    return c


def run(c, xc, exxdiv):
    mf = dft.RKS(c, xc=xc)
    mf.exxdiv = exxdiv
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.with_df = df.AFTDF(c)
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
    for cell, name, exxdiv in ROWS:
        tag = f"{cell} {name} {exxdiv}"
        if only not in tag:
            continue
        t0 = time.time()
        e = run(make_cell(LATTICES[cell]), FUNCTIONALS[name], exxdiv)
        dt = time.time() - t0
        print(f"{tag}: E {e:.12f}  ({dt:.0f} s)", flush=True)
        rows.append(
            {
                "cell": cell,
                "functional": name,
                "exxdiv": exxdiv or "none",
                "e_tot": e,
                "runtime_s": round(dt, 1),
            }
        )
    meta = {
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "date": datetime.date.today().isoformat(),
        "generator": "scripts/gen_pyscf_pbc_grsh_refs.py",
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
        "cell": "H2 STO-3G (PySCF digits), atoms H2_ATOMS, Bohr; cubic a=4 and triclinic TRI_A",
        "lattices": LATTICES,
        "atoms": ATOMS,
        "setup": f"pbc.dft.RKS (Gamma), AFTDF, UniformGrids {MESH}^3, conv_tol 1e-12",
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
