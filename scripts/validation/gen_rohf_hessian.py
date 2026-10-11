"""PySCF-based FD references for the analytic ROHF Hessian (issue #291).

Consumers: crates/ferric-scf/tests/rohf_hessian_skeleton.rs (energy and
gradient sanity checks), rohf_hessian_response.rs and
validation_rohf_hessian.rs (the Hessian and frequencies).
Output:   testdata/reference/validation/rohf_hessian/<system>_<basis>.json

WHY A FINITE DIFFERENCE: PySCF has NO analytic ROHF Hessian (`scf.ROHF(mol)`
has no `.Hessian()`), so the only independent reference is a central
difference of PySCF's ANALYTIC ROHF gradient, `scf.ROHF(mol).nuc_grad_method()`,
built with the same construction as `gen_uhf_hessian.py::_fd`: every displaced
SCF is started from the centre density and must stay within BASIN_TOL of the
centre energy (a displaced SCF on another state would fabricate a Hessian).

Each block carries:

* `hessian_fd`    - central FD at FD_STEP, symmetrized (raw asymmetry recorded).
* `hessian_fd_2h` - the same at 2 * FD_STEP. Their max difference is the O(h^2)
  truncation estimate of `hessian_fd` (the central error scales as h^2, so
  the error of `hessian_fd` itself is about a third of the gap).
* `scf` - `common.run_open_shell(.., "rohf")`: the lowest internally STABLE ROHF
  over several guesses (the stability / lowest-state check of gen_uhf_rohf.py);
  a state with no stable solution is refused, never written.
* `freq_fd_cm` - harmonic frequencies of the FD Hessian with ferric's masses.

Systems (geometries in testdata/molecules/validation/, used AS GIVEN), all
cleanly converging with a non-degenerate SOMO (the near-degenerate SOMO of OH
is deliberately avoided):
    nh2 (2B1 doublet), ch2_triplet (3B1), ho2 (2A" doublet) x cc-pVDZ

Run (light; a few minutes in total):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python \\
        scripts/validation/gen_rohf_hessian.py
Restrict to systems by name:  ... gen_rohf_hessian.py nh2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "rohf_hessian"
ROW_NAME = "Analytic ROHF Hessian (FD reference)"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FD_STEP = 1.0e-3
BASIN_TOL = 1e-3

# ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 standard weights, u).
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

# (system, basis, multiplicity)
SYSTEMS = (
    ("nh2", "cc-pvdz", 2),
    ("ch2_triplet", "cc-pvdz", 3),
    ("ho2", "cc-pvdz", 2),
)


def _rohf(mol, dm0):
    from pyscf import scf

    mf = scf.ROHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("displaced ROHF did not converge")
    return mf


def _fd(mol, mf0, step):
    import numpy as np

    n = 3 * mol.natm
    x0 = mol.atom_coords(unit="Bohr")
    dm0 = mf0.make_rdm1()
    e0 = mf0.e_tot
    h = np.zeros((n, n))
    worst_basin = 0.0
    for b in range(n):
        grads = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * step
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            mf = _rohf(m, dm0)
            de = abs(mf.e_tot - e0)
            if de > BASIN_TOL:
                raise RuntimeError(f"displaced ROHF changed state: dE {de:.3e}")
            worst_basin = max(worst_basin, de)
            grads.append(mf.nuc_grad_method().kernel().ravel())
        h[:, b] = (grads[0] - grads[1]) / (2.0 * step)
    asym = float(np.max(np.abs(h - h.T)))
    return 0.5 * (h + h.T), asym, worst_basin


def _frequencies(mol, h, masses):
    import numpy as np
    from pyscf.hessian import thermo

    natm = mol.natm
    h4 = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(
        mol, h4, mass=np.asarray(masses), imaginary_freq=False
    )
    return sorted(float(v) for v in np.asarray(res["freq_wavenumber"]).real)


def _write_compact(system: str, basis_name: str, payload: dict) -> Path:
    """common.write_reference's contract (provenance required, never an
    unstable state) with compact JSON."""
    if "provenance" not in payload:
        raise ValueError("refusing to write a reference with no provenance block")
    if payload["scf"]["stability"]["internal_stable"] is not True:
        raise RuntimeError(f"refusing to write {system}/{basis_name}: UNSTABLE state")
    path = common.reference_path(ROW, system, basis_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def _run(system: str, basis_name: str, mult: int) -> Path:
    import numpy as np
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=mult)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    scf_block, mf = common.run_open_shell(
        mol, "rohf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    h_fd, fd_asym, worst_basin = _fd(mol, mf, FD_STEP)
    h_fd2, _, _ = _fd(mol, mf, 2.0 * FD_STEP)
    f_fd = _frequencies(mol, h_fd, masses)
    f_fd2 = _frequencies(mol, h_fd2, masses)
    gap_h = float(np.max(np.abs(h_fd - h_fd2)))
    gap_f = float(max(abs(a - b) for a, b in zip(f_fd, f_fd2)))
    print(
        f"{system:12s} {basis_name:8s} E {mf.e_tot:.10f} <S2> {scf_block['s_squared']:.4f} "
        f"freqs {['%.1f' % f for f in f_fd]} | h-vs-2h H {gap_h:.2e} freq {gap_f:.3f} cm-1 "
        f"| fd asym {fd_asym:.1e}",
        flush=True,
    )
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "charge": 0,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "frequency_convention": (
            "harmonic_analysis(mass=masses_amu, imaginary_freq=False): ascending, "
            "imaginary reported as negative, translations+rotations projected out"
        ),
        "energy": float(mf.e_tot),
        "scf": scf_block,
        "gradient": np.asarray(mf.nuc_grad_method().kernel()).tolist(),
        "hessian_fd": h_fd.tolist(),
        "freq_fd_cm": f_fd,
        "hessian_fd_2h": h_fd2.tolist(),
        "freq_fd_2h_cm": f_fd2,
        "fd": {
            "fd_step_bohr": FD_STEP,
            "fd_asymmetry_max": fd_asym,
            "fd_n_gradients": 6 * mol.natm,
            "fd_worst_abs_energy_change": worst_basin,
            "h_vs_2h_hessian_max_abs": gap_h,
            "h_vs_2h_freq_max_abs_cm": gap_f,
        },
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "rohf",
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "max_cycle": 500,
                "eri": "exact 4-index (no density fitting)",
                "state": "common.run_open_shell: lowest internally stable ROHF over guesses",
                "analytic_hessian": "none (PySCF has no ROHF Hessian)",
                "fd_hessian": (
                    f"central FD of analytic ROHF gradient, step {FD_STEP} Bohr "
                    f"(and 2x), displaced SCFs from the centre density, "
                    f"|dE| <= {BASIN_TOL}, symmetrized"
                ),
                "masses": "ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 averages)",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=None,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=scf_block["stability"],
            generator="scripts/validation/gen_rohf_hessian.py",
            extra={"basis_self_check": basis_check},
        ),
    }
    return _write_compact(system, basis_name, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = [_run(s, b, m) for s, b, m in SYSTEMS if not only or s in only]
    for p in written:
        print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
    print(f"GEN_ROHF_HESSIAN_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
