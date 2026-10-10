"""PySCF references for the QM/MM Hessian row (issue #286).

Consumer: crates/ferric-scf/tests/validation_qmmm_hessian.rs.
Output:   testdata/reference/validation/qmmm_hessian/<system>_<basis>.json

PySCF has no QM/MM Hessian, so the reference is the central finite difference
of PySCF's ANALYTIC QM/MM gradient (`qmmm.mm_charge(mf).nuc_grad_method()`),
QM atoms only, MM charges fixed in space — the block ferric's analytic
Hessian computes. Each displaced SCF starts from the centre density and must
stay within BASIN_TOL of the centre energy; the result is symmetrized (raw
asymmetry recorded). `hessian_gas_fd` is the same construction without the
charges: the negative control, which the embedded Hessian must differ from.

Systems (geometries and charges exactly as gen_qmmm.py builds them):
    h2o_q10    RHF  cc-pVDZ   (10 TIP3P-like point charges)
    ch3oh_q20  RHF  def2-SVP  (the 20 shell sites nearest the QM atoms)
    h2o_q10_cation  UHF  cc-pVDZ  (H2O+, doublet, same charges)

Run:
    OPENBLAS_NUM_THREADS=1 python scripts/validation/gen_qmmm_hessian.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import gen_qmmm as g

ROW = "qmmm_hessian"
GEN = "scripts/validation/gen_qmmm_hessian.py"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FD_STEP = 1.0e-3
BASIN_TOL = 1e-3


def _mf(mol, uhf):
    from pyscf import scf

    mf = scf.UHF(mol) if uhf else scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 300
    mf.verbose = 0
    return mf


def _scf(mol, mm, uhf, dm0=None, radii=None):
    from pyscf import qmmm

    mf = _mf(mol, uhf)
    if mm is not None:
        coords, charges = g._mm_arrays(mm)
        kw = {} if radii is None else {"radii": np.asarray(radii, dtype=float)}
        mf = qmmm.mm_charge(mf, coords, charges, unit="Bohr", **kw)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def _fd_hessian(mol, mm, uhf, radii=None):
    mf0 = _scf(mol, mm, uhf, radii=radii)
    if uhf:
        # Refuse a reference on an unstable UHF state.
        _, _, stable, _ = mf0.stability(return_status=True)
        if not stable:
            raise RuntimeError("centre UHF state is not internally stable")
    dm0 = mf0.make_rdm1()
    e0 = mf0.e_tot
    x0 = mol.atom_coords(unit="Bohr")
    n = 3 * mol.natm
    h = np.zeros((n, n))
    worst = 0.0
    for b in range(n):
        grads = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * FD_STEP
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            mf = _scf(m, mm, uhf, dm0=dm0, radii=radii)
            worst = max(worst, abs(mf.e_tot - e0))
            if worst > BASIN_TOL:
                raise RuntimeError(f"displaced SCF changed state: dE {worst:.3e}")
            grads.append(mf.nuc_grad_method().kernel().reshape(-1))
        h[:, b] = (grads[0] - grads[1]) / (2.0 * FD_STEP)
    asym = float(np.max(np.abs(h - h.T)))
    return 0.5 * (h + h.T), asym, worst, float(e0)


def _case(system, xyz, basis, mm, uhf, charge=0, mult=1, radii=None):
    mol, payload = g.common_payload(
        "QM/MM Hessian",
        system,
        xyz,
        basis,
        mm,
        "uhf" if uhf else "rhf",
        f"qmmm.mm_charge(scf.{'UHF' if uhf else 'RHF'}) exact J/K; central FD of "
        f"nuc_grad_method() at step {FD_STEP} Bohr; conv_tol {CONV_TOL}",
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        extra_prov={"fd_step_bohr": FD_STEP, "basin_tol": BASIN_TOL},
    )
    payload["provenance"]["generator"] = GEN
    if radii is not None:
        # Gaussian-smeared charges: width in Bohr == PySCF `radii` with unit="Bohr"
        # (see gen_qmmm.py's SMEARED-CHARGE UNIT CONVENTION).
        for c, w in zip(payload["mm_charges"], radii):
            c["width"] = float(w)
        payload["provenance"]["keywords"] += f"; radii (Bohr) {list(map(float, radii))}"
    if charge or mult != 1:
        from pyscf import gto

        symbols, coords = common.read_xyz(xyz)
        bas, _ = common.pyscf_basis(basis, symbols)
        mol = gto.M(
            atom=common.pyscf_atom_bohr(symbols, coords),
            unit="Bohr",
            basis=bas,
            cart=False,
            charge=charge,
            spin=mult - 1,
            verbose=0,
        )
        payload["charge"], payload["multiplicity"] = charge, mult
        payload["nuclear_repulsion"] = float(mol.energy_nuc())
    h, asym, basin, e0 = _fd_hessian(mol, mm, uhf, radii)
    hg, asym_g, _, _ = _fd_hessian(mol, None, uhf)
    payload.update(
        {
            "energy": e0,
            "hessian_fd": h.tolist(),
            "hessian_fd_asymmetry": asym,
            "hessian_gas_fd": hg.tolist(),
            "hessian_gas_fd_asymmetry": asym_g,
            "fd_worst_basin_dE": basin,
        }
    )
    path = common.write_reference(ROW, system, basis, payload)
    print(
        f"{path.name}: max|H| {np.abs(h).max():.4f} asym {asym:.2e} "
        f"|H - H_gas| {np.abs(h - hg).max():.3e}"
    )
    return path


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    cases = {
        "h2o_q10": lambda: _case(
            "h2o_q10", g._xyz("h2o.xyz"), "cc-pvdz", g.water_mm(), False
        ),
        "ch3oh_q20": lambda: _case(
            "ch3oh_q20",
            g._xyz("ch3oh.xyz"),
            "def2-svp",
            g.nearest_sites(
                g.shell_mm(), common.read_xyz(g._xyz("ch3oh.xyz"))[1], g.KS_N_CHARGES
            ),
            False,
        ),
        "h2o_q10_cation": lambda: _case(
            "h2o_q10_cation", g._xyz("h2o.xyz"), "cc-pvdz", g.water_mm(), True, 1, 2
        ),
    }
    # Gaussian-smeared MM charges (#358): widths cycle 0.7 / 1.1 / 1.6 Bohr.
    widths = [(0.7, 1.1, 1.6)[i % 3] for i in range(len(g.water_mm()))]
    cases["h2o_q10_smeared"] = lambda: _case(
        "h2o_q10_smeared", g._xyz("h2o.xyz"), "cc-pvdz", g.water_mm(), False, radii=widths
    )
    cases["h2o_q10_cation_smeared"] = lambda: _case(
        "h2o_q10_cation_smeared",
        g._xyz("h2o.xyz"),
        "cc-pvdz",
        g.water_mm(),
        True,
        1,
        2,
        radii=widths,
    )
    for name, fn in cases.items():
        if not only or name in only:
            fn()
    print("GEN_QMMM_HESSIAN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
