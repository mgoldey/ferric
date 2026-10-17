"""PySCF references for ECP + density-fitted (RI-JK) RHF analytic Hessians (#406).

Consumer: crates/ferric-scf/tests/validation_ecp_df_hessian.rs.
Output:   testdata/reference/validation/ecp_df_hessian/<system>_<basis>.json

PySCF `scf.RHF(mol).density_fit(aux)` with ferric's own basis, ECP and aux JSON
(like-for-like via common.py), geometry in Bohr. Per system:

* `hessian_analytic`  `mf.Hessian().kernel()` (pyscf.df.hessian.rhf, full aux response).
* `hessian_fd`        central FD of PySCF's ANALYTIC DF gradient (independent of any
                      Hessian formula; the ECP is in hcore on both).
* `hessian_exact_jk`  PySCF's exact-integral ECP Hessian: the NEGATIVE CONTROL (the
                      fitting error is the signal ferric must reproduce).

Usage:
    OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python \
        scripts/validation/gen_ecp_df_hessian.py [system ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_df_hessian as gd  # noqa: E402

ROW = "ecp_df_hessian"
ROW_NAME = "Analytic Hessian with ECPs and RI-JK"
ECP_MOL_DIR = common.MOL_DIR / "ecp"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CPHF_TOL = 1e-12
FD_STEP = 1.0e-3
AUX = "def2-universal-jkfit"
BASIS = "def2-svp"
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "I": 126.90447}
SYSTEMS = ["hi", "ch3i"]


def _rhf(mol, aux=None, dm0=None):
    from pyscf import scf

    mf = scf.RHF(mol)
    if aux is not None:
        mf = mf.density_fit(auxbasis=aux)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.conv_tol_cpscf = CPHF_TOL
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def _fd(mol, mf0, aux):
    import numpy as np

    n = 3 * mol.natm
    x0 = mol.atom_coords(unit="Bohr")
    dm0 = mf0.make_rdm1()
    h = np.zeros((n, n))
    for b in range(n):
        g = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * FD_STEP
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            g.append(_rhf(m, aux, dm0=dm0).nuc_grad_method().kernel().ravel())
        h[:, b] = (g[0] - g[1]) / (2.0 * FD_STEP)
    return 0.5 * (h + h.T), float(np.max(np.abs(h - h.T)))


def _run(system: str) -> Path:
    import numpy as np
    import pyscf

    xyz = ECP_MOL_DIR / f"{system}.xyz"
    symbols, _ = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(BASIS)
    mol = common.build_pyscf_mol(xyz, BASIS, ecp_json=ecp_json)
    common.check_basis_like_for_like(mol, BASIS, symbols)
    common.check_ecp_like_for_like(mol, ecp_json, symbols)
    aux = gd._aux(symbols, AUX)
    gd._check_aux(mol, aux, AUX, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    mf = _rhf(mol, aux)
    h_an, an_asym = gd._analytic_standard(mf)
    h_fd, fd_asym = _fd(mol, mf, aux)
    mf_ex = _rhf(mol)
    h_ex, _ = gd._analytic_standard(mf_ex)
    gap = float(np.max(np.abs(h_fd - h_an)))
    fit = float(np.max(np.abs(h_ex - h_an)))
    print(f"{system}: E {mf.e_tot:.10f} PySCF FD-vs-analytic {gap:.2e}; exact-vs-RIJK {fit:.2e}")
    if fit < 5e-6:
        raise RuntimeError("fitting signal vanishes: vacuous negative control")
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS,
        "aux_basis": AUX,
        "charge": 0,
        "multiplicity": 1,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "energy": float(mf.e_tot),
        "energy_exact": float(mf_ex.e_tot),
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": an_asym,
        "hessian_fd": h_fd.tolist(),
        "hessian_exact_jk": h_ex.tolist(),
        "pyscf_fd_vs_analytic_hessian_max_abs": gap,
        "exact_vs_ri_jk_hessian_max_abs": fit,
        "fd": {"fd_step_bohr": FD_STEP, "fd_asymmetry_max": fd_asym},
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={"method": "rhf", "eri": f"density_fit({AUX}) J+K", "ecp": "ferric def2-ecp"},
            basis_name=BASIS,
            xyz_path=xyz,
            coords_bohr=mol.atom_coords(unit="Bohr").tolist(),
            symbols=symbols,
            aux=AUX,
            ecp_json=ecp_json,
            generator="scripts/validation/gen_ecp_df_hessian.py",
        ),
    }
    path = common.reference_path(ROW, system, BASIS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def main() -> int:
    only = set(sys.argv[1:])
    for s in SYSTEMS:
        if not only or s in only:
            print(_run(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
