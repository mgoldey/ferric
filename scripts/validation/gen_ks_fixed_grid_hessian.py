"""PySCF references for the VALIDATION.md "Analytic closed-shell KS Hessian on a fixed grid" row.

Consumer: crates/ferric-scf/tests/validation_ks_fixed_grid_hessian.rs
Output:   testdata/reference/validation/ks_fixed_grid_hessian/<system>_<basis>.json
Input:    testdata/reference/validation/ks_fixed_grid_hessian/<system>_grid.f64
          (written by the ferric test `reference_grid_matches_the_grid_ferric_builds` run with FERRIC_WRITE_REFERENCE_GRID=1: rows of
          `x y z w`, little-endian f64, Bohr; ferric's 30 x 50 Becke-Lebedev
          grid for the reference geometry)

LIKE-FOR-LIKE CONDITIONS (what makes the comparison exact rather than
approximate):
  * the SAME grid points and weights: they are read from ferric's file and
    installed as `mf.grids.coords` / `mf.grids.weights`, so PySCF never builds
    its own grid (it only does so when `coords is None`);
  * no grid response: PySCF's `Hessian.grid_response` / `Gradients.grid_response`
    default to False, i.e. the grid is not differentiated, which is the energy
    `E(R)` with the grid frozen in space that ferric differentiates;
  * the SAME basis (ferric's JSON through common.py) and geometry (Bohr, ferric's
    constant), the same functional (per case, both libxc), exact J, no
    density fitting, SCF converged to 1e-12 in energy;
  * masses = ferric's IUPAC-2013 weights for the frequencies.

Stored: PySCF's total energy (the grid check: it must agree with ferric's to
~1e-9 before any Hessian is compared), its analytic Hessian, and the central
FD (Richardson, h and h/2) of PySCF's analytic fixed-grid gradient with every
displaced SCF on the SAME frozen grid (the reference's own consistency).

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python \\
        scripts/validation/gen_ks_fixed_grid_hessian.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

ROW = "ks_fixed_grid_hessian"
XC = "LDA_X,LDA_C_VWN"
# (system, basis, xc, file suffix): the pure-LDA files keep their original names.
HYBRID_XC = "HYB_LDA_XC_LDA0"
CONV_TOL = 1e-12
FD_STEP = 5.0e-4
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
SYSTEMS = (
    ("h2o_distorted", "6-31g", XC, ""),
    ("h2o_distorted", "cc-pvdz", XC, ""),
    ("h2o_distorted", "6-31g", HYBRID_XC, "_hyb"),
    ("h2o_distorted", "6-31g", "PBE0", "_pbe0"),
    ("h2o_distorted", "cc-pvdz", "PBE", "_pbe"),
    ("h2o_distorted", "6-31g", "B3LYP", "_b3lyp"),
)


def _grid(system: str):
    import numpy as np

    path = common.REF_DIR / ROW / f"{system}_grid.f64"
    rows = np.fromfile(path, dtype="<f8").reshape(-1, 4)
    return rows[:, :3].copy(), rows[:, 3].copy()


def _rks(mol, pts, w, xc=XC):
    from pyscf import dft

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.coords = pts
    mf.grids.weights = w
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RKS did not converge")
    if mf.grids.coords is not pts and len(mf.grids.coords) != len(pts):
        raise RuntimeError("PySCF replaced the supplied grid")
    return mf


def _as_3n(h4, natm):
    return h4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _analytic(mf):
    import numpy as np

    hobj = mf.Hessian()
    hobj.grid_response = False
    hobj.max_cycle = 100
    raw = _as_3n(hobj.kernel(), mf.mol.natm)
    return 0.5 * (raw + raw.T), float(np.max(np.abs(raw - raw.T)))


def _grad(mol, pts, w, xc=XC):
    g = _rks(mol, pts, w, xc).nuc_grad_method()
    g.grid_response = False
    return g.kernel().ravel()


def _fd(mol, pts, w, xc=XC):
    import numpy as np

    n = 3 * mol.natm
    x0 = mol.atom_coords(unit="Bohr")

    def column(b, step):
        out = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * step
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            out.append(_grad(m, pts, w, xc))
        return (out[0] - out[1]) / (2.0 * step)

    h = np.zeros((n, n))
    for b in range(n):
        h[:, b] = (4.0 * column(b, 0.5 * FD_STEP) - column(b, FD_STEP)) / 3.0
    return 0.5 * (h + h.T), float(np.max(np.abs(h - h.T)))


def _frequencies(mol, h, masses):
    import numpy as np
    from pyscf.hessian import thermo

    natm = mol.natm
    h4 = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(
        mol, h4, mass=np.asarray(masses), imaginary_freq=False
    )
    return sorted(float(v) for v in np.asarray(res["freq_wavenumber"]).real)


def _run(system: str, basis_name: str, xc: str = XC, suffix: str = "") -> Path:
    import numpy as np
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    pts, w = _grid(system)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    mf = _rks(mol, pts, w, xc)
    h_an, asym = _analytic(mf)
    h_fd, fd_asym = _fd(mol, pts, w, xc)
    gap = float(np.max(np.abs(h_an - h_fd)))
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    print(
        f"{system} {basis_name} E {mf.e_tot:.12f} npts {len(pts)} | analytic-vs-FD "
        f"{gap:.2e} | freqs {[f'{f:.2f}' for f in f_an]} | asym {asym:.1e}",
        flush=True,
    )
    payload = {
        "row": "Analytic closed-shell KS Hessian on a fixed grid",
        "system": system,
        "basis": basis_name,
        "charge": 0,
        "multiplicity": 1,
        "xc": xc,
        "npts": len(pts),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "energy": float(mf.e_tot),
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": asym,
        "hessian_fd": h_fd.tolist(),
        "hessian_fd_asymmetry": fd_asym,
        "pyscf_analytic_vs_fd_max": gap,
        "frequencies_analytic_cm1": f_an,
        "frequencies_fd_cm1": f_fd,
        "masses_amu": masses,
        "fd_step_bohr": FD_STEP,
        "basis_check": basis_check,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "xc": xc,
                "grid": "ferric 30x50 Becke-Lebedev points/weights, frozen",
                "grid_response": False,
                "conv_tol": CONV_TOL,
                "density_fit": False,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid="ferric 30x50 (testdata/reference/validation/"
            f"{ROW}/{system}_grid.f64)",
            scf_conv=CONV_TOL,
            generator="scripts/validation/gen_ks_fixed_grid_hessian.py",
        ),
    }
    path = common.reference_path(ROW, system, basis_name)
    path = path.with_name(path.stem + suffix + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def main() -> int:
    only = set(sys.argv[1:])
    written = [
        _run(s, b, x, sfx) for s, b, x, sfx in SYSTEMS if not only or b + sfx in only
    ]
    for p in written:
        print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
    print(f"GEN_KS_FIXED_GRID_HESSIAN_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
