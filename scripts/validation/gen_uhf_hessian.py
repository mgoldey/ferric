"""PySCF references for the VALIDATION.md "Analytic UHF Hessian" row.

Consumer: crates/ferric-scf/tests/validation_uhf_hessian.rs.
Output:   testdata/reference/validation/uhf_hessian/<system>_<basis>.json

WHAT FERRIC COMPUTES (crates/ferric-scf/src/hessian.rs, read from the code):
the analytic UHF Hessian with exact four-centre J/K, assembled term by term
the way PySCF's `hessian/uhf.py` assembles it — skeleton (`hess_nuc` +
`partial_hess_elec`: D·h'' with D = D_a + D_b, −W·S'' with the UHF W, Γ·ERI''
with the UHF Γ) plus the coupled alpha/beta CPHF response (`hess_elec` with
`solve_mo1` → `ucphf.solve`). So every block here carries:

* `hessian_analytic` — PySCF `mf.Hessian().kernel()` (CPHF tolerance
  tightened to CPHF_TOL), stored SYMMETRIZED with its raw asymmetry
  recorded. The like-for-like reference for ferric's `uhf_hessian`.
* `hessian_skeleton` — PySCF `partial_hess_elec + hess_nuc` (terms 1-4 at
  frozen densities), so a mismatch can be localized to skeleton vs response.
* `hessian_fd` — central FD of PySCF's ANALYTIC UHF gradient at FD_STEP, every
  displaced SCF started from the centre alpha/beta densities and checked to
  stay within BASIN_TOL of the centre energy, symmetrized: PySCF's own
  independent check of its analytic Hessian.
* frequencies of both via `hessian.thermo.harmonic_analysis(mass = ferric's
  masses, imaginary_freq = False)`.
* `scf` — the `common.run_open_shell` block: the state is the lowest
  internally STABLE UHF solution over several guesses (a reference on an
  unstable state is refused), with <S^2> and the guess scan.
* `beta_hole_quadrupole` — for the linear 2Pi OH only: `Tr((D_a − D_b) r_i r_j)`
  (xx, xy, yy about the origin). Its (xx − yy, 2xy) direction is the angle of
  the singly-occupied pi orbital about the bond (z) axis, which is a free
  choice of the UHF solution: the skeleton's perpendicular blocks follow it,
  the total Hessian does not (see the consumer).

Systems (geometries in testdata/molecules/validation/, used AS GIVEN):
    oh (doublet), nh2 (doublet), ch2_triplet (triplet)  x cc-pVDZ

Basis/geometry like-for-like through common.py (ferric's own basis JSON,
Bohr geometry with ferric's constant). J/K exact (no density fitting).

Run (light; well under a minute per system):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python \\
        scripts/validation/gen_uhf_hessian.py
Restrict to systems by name:  ... gen_uhf_hessian.py nh2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "uhf_hessian"
ROW_NAME = "Analytic UHF Hessian"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CPHF_TOL = 1e-12
FD_STEP = 1.0e-3
BASIN_TOL = 1e-3

# ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 standard weights, u); the
# Rust test asserts atom_masses(mol) equals the `masses_amu` written here.
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

# (system, basis, multiplicity)
SYSTEMS = (
    ("oh", "cc-pvdz", 2),
    ("nh2", "cc-pvdz", 2),
    ("ch2_triplet", "cc-pvdz", 3),
)


def _uhf(mol, dm0):
    from pyscf import scf

    mf = scf.UHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("displaced UHF did not converge")
    return mf


def _as_3n(h4, natm):
    return h4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _analytic(mf):
    import numpy as np
    from pyscf import hessian  # noqa: F401  (registers mf.Hessian)

    mf.conv_tol_cpscf = CPHF_TOL
    hobj = mf.Hessian()
    hobj.max_cycle = 100
    natm = mf.mol.natm
    raw = _as_3n(hobj.kernel(), natm)
    skel = _as_3n(
        hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ) + hobj.hess_nuc(),
        natm,
    )
    asym = float(np.max(np.abs(raw - raw.T)))
    return 0.5 * (raw + raw.T), asym, 0.5 * (skel + skel.T)


def _fd(mol, mf0):
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
            x[b // 3, b % 3] += sgn * FD_STEP
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            mf = _uhf(m, dm0)
            de = abs(mf.e_tot - e0)
            if de > BASIN_TOL:
                raise RuntimeError(f"displaced UHF changed state: dE {de:.3e}")
            worst_basin = max(worst_basin, de)
            grads.append(mf.nuc_grad_method().kernel().ravel())
        h[:, b] = (grads[0] - grads[1]) / (2.0 * FD_STEP)
    asym = float(np.max(np.abs(h - h.T)))
    return 0.5 * (h + h.T), asym, worst_basin


def _frequencies(mol, h, masses):
    import numpy as np
    from pyscf.hessian import thermo

    natm = mol.natm
    h4 = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(mol, h4, mass=np.asarray(masses), imaginary_freq=False)
    return sorted(float(v) for v in np.asarray(res["freq_wavenumber"]).real)


def _spin_quadrupole(mol, mf):
    """Tr((D_a − D_b) r_i r_j) about the origin: [xx, xy, yy]."""
    import numpy as np

    dma, dmb = mf.make_rdm1()
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        rr = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)
    q = np.einsum("ijpq,qp->ij", rr, dma - dmb)
    return [float(q[0, 0]), float(q[0, 1]), float(q[1, 1])]


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
        mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    h_an, an_asym, h_skel = _analytic(mf)
    h_fd, fd_asym, worst_basin = _fd(mol, mf)
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    gap_h = float(np.max(np.abs(h_fd - h_an)))
    gap_f = float(max(abs(a - b) for a, b in zip(f_an, f_fd)))
    print(
        f"{system:12s} {basis_name:8s} E {mf.e_tot:.10f} <S2> {scf_block['s_squared']:.4f} "
        f"freqs {['%.1f' % f for f in f_an]} | FD-vs-an H {gap_h:.2e} freq {gap_f:.3f} cm-1 "
        f"| an asym {an_asym:.1e} | lambda_min {scf_block['stability']['lambda_min']:.3e}",
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
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": an_asym,
        "hessian_skeleton": h_skel.tolist(),
        "freq_analytic_cm": f_an,
        "hessian_fd": h_fd.tolist(),
        "freq_fd_cm": f_fd,
        "fd": {
            "fd_step_bohr": FD_STEP,
            "fd_asymmetry_max": fd_asym,
            "fd_n_gradients": 6 * mol.natm,
            "fd_worst_abs_energy_change": worst_basin,
        },
        "pyscf_fd_vs_analytic_hessian_max_abs": gap_h,
        "pyscf_fd_vs_analytic_freq_max_abs_cm": gap_f,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "uhf",
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "conv_tol_cpscf": CPHF_TOL,
                "max_cycle": 500,
                "eri": "exact 4-index (no density fitting)",
                "state": "common.run_open_shell: lowest internally stable UHF over guesses",
                "analytic_hessian": "pyscf.hessian.uhf (mf.Hessian().kernel()), symmetrized",
                "skeleton": "Hessian.partial_hess_elec + Hessian.hess_nuc",
                "fd_hessian": (
                    f"central FD of analytic UHF gradient, step {FD_STEP} Bohr, displaced "
                    f"SCFs from the centre alpha/beta densities, |dE| <= {BASIN_TOL}, "
                    "symmetrized"
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
            generator="scripts/validation/gen_uhf_hessian.py",
            extra={"basis_self_check": basis_check},
        ),
    }
    if system == "oh":
        payload["beta_hole_quadrupole"] = _spin_quadrupole(mol, mf)
    return _write_compact(system, basis_name, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = [_run(s, b, m) for s, b, m in SYSTEMS if not only or s in only]
    for p in written:
        print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
    print(f"GEN_UHF_HESSIAN_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
