"""PySCF references for the k-point range-separated-hybrid pins
(crates/ferric-pbc/tests/pbc_krsh_more.rs).

Widens the existing H2/STO-3G cubic a = 4 pins (pbc_krsh.rs) along three axes:
a TRICLINIC cell (TRI_A lattice of tests/common, atoms H2_ATOMS), exxdiv =
ewald AND exxdiv = None, and meshes beyond 1x1x2 (1x2x2, 2x1x2 and 2x2x2; the
2x2x2 rows were out of reach for the earlier cubic mesh-61 pins).

PySCF 2.13 pbc.dft.KRKS with AFTDF (the attenuated K is only on AFTDF),
UniformGrids MESH^3, conv_tol 1e-12. STO-3G H is spelled out digit for digit
(the same numbers as common::pyscf_sto3g_h).

Output: testdata/reference/pbc_h2_sto3g_triclinic_krsh.json, with provenance
(git SHA of the repo at generation, date, versions, per-row runtime).

Run:  OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_pbc_krsh_refs.py [--only SUBSTR]
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
OUT = os.path.join(ROOT, "testdata", "reference", "pbc_h2_sto3g_triclinic_krsh.json")

LATTICE = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
ATOMS = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]]
STO3G_H = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
MESH = 41
# PySCF default 1e-8 leaves a -1.5e-10 offset at the 1x1x1 mesh (lattice-sum cutoff)
PRECISION = 1e-11
FUNCTIONALS = {"HSE06": "HSE06", "CAM_B3LYP": "CAMB3LYP"}
FERRIC_NAMES = {"HSE06": "HSE06", "CAM_B3LYP": "HYB_GGA_XC_CAM_B3LYP"}
# (functional, kmesh, exxdiv)
ROWS = [
    ("HSE06", (1, 1, 1), "ewald"),
    ("HSE06", (1, 1, 1), None),
    ("CAM_B3LYP", (1, 1, 1), "ewald"),
    ("CAM_B3LYP", (1, 1, 1), None),
    ("HSE06", (1, 2, 2), "ewald"),
    ("HSE06", (1, 2, 2), None),
    ("CAM_B3LYP", (2, 1, 2), "ewald"),
    ("HSE06", (2, 2, 2), "ewald"),
    ("CAM_B3LYP", (2, 2, 2), None),
]


def make_cell():
    c = gto.Cell()
    c.atom = [["H", tuple(x)] for x in ATOMS]
    c.a = np.array(LATTICE)
    c.unit = "B"
    c.basis = {"H": STO3G_H}
    c.mesh = [MESH] * 3
    c.precision = PRECISION
    c.verbose = 0
    c.build()
    return c


def run(c, xc, kmesh, exxdiv):
    kp = c.make_kpts(kmesh)
    mf = dft.KRKS(c, kpts=kp, xc=xc)
    mf.exxdiv = exxdiv
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.with_df = df.AFTDF(c, kpts=kp)
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
    for name, kmesh, exxdiv in ROWS:
        tag = f"{name} {kmesh} {exxdiv}"
        if only not in tag:
            continue
        t0 = time.time()
        e = run(make_cell(), FUNCTIONALS[name], kmesh, exxdiv)
        dt = time.time() - t0
        print(f"{tag}: E {e:.12f}  ({dt:.0f} s)", flush=True)
        rows.append(
            {
                "functional": name,
                "kmesh": list(kmesh),
                "exxdiv": exxdiv or "none",
                "e_tot": e,
                "runtime_s": round(dt, 1),
            }
        )
    meta = {
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "date": datetime.date.today().isoformat(),
        "generator": "scripts/gen_pyscf_pbc_krsh_refs.py",
        "repo_sha_at_generation": git_sha(),
        "repo_sha_note": "HEAD of the worktree when the script ran: the PARENT of the "
        "commit that adds/changes this script, not the commit containing it",
        "libxc_version": libxc.libxc_version(),
        "xc_names": {
            k: {
                "pyscf": v,
                "ferric": FERRIC_NAMES[k],
                "pyscf_rsh_coeff_omega_clr_csr_minus_clr": list(libxc.rsh_coeff(v)),
                "omega_bohr_inv": libxc.rsh_coeff(v)[0],
                "c_lr": libxc.rsh_coeff(v)[1],
                "c_sr": libxc.rsh_coeff(v)[1] + libxc.rsh_coeff(v)[2],
            }
            for k, v in FUNCTIONALS.items()
        },
        "grids": f"XC grid (KNumInt UniformGrids) and AFTDF mesh are both {MESH}^3; "
        f"cell.precision {PRECISION}",
        "cell": "H2 STO-3G (PySCF digits), triclinic TRI_A lattice, atoms H2_ATOMS, Bohr",
        "lattice": LATTICE,
        "atoms": ATOMS,
        "setup": f"KRKS, AFTDF, UniformGrids {MESH}^3, gamma-centred kmesh, conv_tol 1e-12, mf.grids default (uniform, = cell.mesh)",
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
