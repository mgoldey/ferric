"""PySCF Gamma-point pbc.dft.UKS / pbc.dft.ROKS SCAN and r2SCAN references on a
genuinely POLARIZED cell (issue #492, item "Gamma-point UKS/ROKS meta-GGA
energy"; pins for crates/ferric-pbc/tests/pbc_gamma_open_mgga.rs).

Cell: three H in the cubic a = 4 Bohr cell (doublet, N_a = 2, N_b = 1), atoms
(0.3,0.2,0.1), (0.3,0.2,1.5), (0.3,1.9,0.4), STO-3G H (PySCF's digits): the
`h3` cell of scripts/gen_pyscf_kuks_mgga_refs.py (the alpha channel holds two
orbitals, so the iso-orbital indicator is not trivially 0 and SCAN != r2SCAN).

Setup mirrors that script (and so ferric's `PeriodicGrid::uniform` pins):
AFTDF + UniformGrids(61^3), exxdiv 'ewald' (irrelevant: no exact exchange),
conv_tol 1e-12, cell.precision 1e-12. The Gamma-point UKS is run with the
plain molecular-style `pbc.dft.UKS` / `pbc.dft.ROKS` classes (kpt = 0), NOT
KUKS. With --control it also reruns the existing KUKS 1x1x1 SCAN pin
(-1.524551950364, crates/ferric-pbc/tests/pbc_kuks_mgga.rs), which the Gamma
UKS SCAN row must reproduce (measured 2026-10-27: -1.52455195036389, i.e.
equal to the pin to 1e-13). Each SCF takes ~3 min (61^3 grid); the four rows
~12 min, the control ~3 min more.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u \
           scripts/gen_pyscf_gamma_open_mgga_refs.py [--control] [out.json]
"""

import json
import sys

import numpy as np
import pyscf
from pyscf.pbc import df, dft, gto
from pyscf.pbc.dft import gen_grid

A = 4.0
ATOMS = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]]
NA, NB = 2, 1
MESH = 61
KUKS_1X1X1_SCAN_PIN = -1.524551950364


def build_cell():
    cell = gto.Cell()
    cell.a = np.diag([A] * 3)
    cell.atom = [("H", r) for r in ATOMS]
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.spin = NA - NB
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    return cell


def setup(mf, cell, kpts=None):
    mf.grids = gen_grid.UniformGrids(cell)
    mf.grids.mesh = [MESH] * 3
    mf.exxdiv = "ewald"
    mf.with_df = df.AFTDF(cell) if kpts is None else df.AFTDF(cell, kpts=kpts)
    mf.with_df.mesh = [MESH] * 3
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    return mf


def run(method, xc):
    cell = build_cell()
    cls = {"UKS": dft.UKS, "ROKS": dft.ROKS}[method]
    mf = setup(cls(cell), cell)
    mf.xc = xc
    e = mf.kernel()
    assert mf.converged, (method, xc)
    return float(e), float(mf.scf_summary["exc"])


def run_kuks_control():
    cell = build_cell()
    kpts = cell.make_kpts([1, 1, 1], with_gamma_point=True)
    mf = setup(dft.KUKS(cell, kpts=kpts), cell, kpts)
    mf.xc = "SCAN"
    mf.nelec = (NA, NB)
    e = mf.kernel()
    assert mf.converged
    return float(e)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--control"]
    out = args[0] if args else None
    ctrl = run_kuks_control() if "--control" in sys.argv else None
    if ctrl is not None:
        print(
            f"control KUKS 1x1x1 SCAN {ctrl:.12f} (pin {KUKS_1X1X1_SCAN_PIN}, "
            f"diff {ctrl - KUKS_1X1X1_SCAN_PIN:.2e})"
        )
    rows = {}
    for method in ("UKS", "ROKS"):
        for xc in ("SCAN", "R2SCAN"):
            e, exc = run(method, xc)
            rows[f"{method} {xc}"] = {"e_tot": e, "e_xc": exc}
            print(f"{method} {xc}: e_tot {e:.12f} e_xc {exc:.12f}")
    print(
        "UKS SCAN - KUKS 1x1x1 SCAN pin:",
        rows["UKS SCAN"]["e_tot"] - KUKS_1X1X1_SCAN_PIN,
    )
    doc = {
        "pyscf_version": pyscf.__version__,
        "cell": "H3 doublet (N_a=2, N_b=1) in the a = 4 Bohr cube, atoms "
        "(0.3,0.2,0.1),(0.3,0.2,1.5),(0.3,1.9,0.4), STO-3G H",
        "settings": "Gamma point, pbc.dft.UKS / pbc.dft.ROKS, AFTDF + UniformGrids "
        "61^3, exxdiv ewald, conv_tol 1e-12, cell.precision 1e-12",
        "generator": "scripts/gen_pyscf_gamma_open_mgga_refs.py",
        "kuks_1x1x1_scan_pin": KUKS_1X1X1_SCAN_PIN,
        "unit": "Hartree",
        "rows": rows,
    }
    if out:
        with open(out, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print("wrote", out)
