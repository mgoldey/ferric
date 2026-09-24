"""PySCF references for the VALIDATION.md "Harmonic frequencies" row (W1).

Consumer: crates/ferric-scf/tests/validation_frequencies.rs.
Output:   testdata/reference/validation/frequencies/<system>_<basis>.json

WHAT FERRIC COMPUTES (crates/ferric-scf/src/frequencies.rs, read from the
code): the Cartesian Hessian is a CENTRAL FINITE DIFFERENCE of ferric's
ANALYTIC nuclear gradient, step `DEFAULT_DELTA` = 5e-3 Bohr, one displaced SCF
(fresh guess) per +/- step on each of the 3N coordinates, then symmetrized.
Frequencies come from `frequencies_from_cartesian_hessian`: mass-weight with
the IUPAC-2013 isotope-averaged masses (`ferric_core::elements::atomic_mass`),
project the six (five if linear) mass-weighted translation/rotation vectors
about the centre of mass out on both sides (P H P), diagonalize.

Therefore every block here carries TWO Hessians:

* `hessian_analytic` — PySCF's analytic second derivative (`hessian.rhf`,
  `rks`, `uhf`, `uks`). For HF this is the exact quantity ferric's FD
  approximates, to O(delta^2). For KS it is NOT like-for-like: PySCF's
  semilocal KS Hessian has no grid-response (moving-grid) term, whereas
  ferric's KS gradient DOES include grid response (ferric_dft::gradient,
  `build_atomic_grid_with_response`), so ferric's FD Hessian contains its
  derivative. The block records PySCF's own |FD_gr - analytic| gap, which is
  the size of that construction difference, measured on the reference side
  (5-9 cm-1 on H2O/NH3, 35 cm-1 on the CH3 UKS umbrella mode at (75,110); on
  H2O PBE/6-31G it falls from 1.3e-3 Ha/Bohr^2 at (75,110) to 1.0e-5 at
  (99,590), the HF-like FD truncation, so it IS the grid-response term).
* `hessian_fd` — central FD of PySCF's ANALYTIC gradient (KS with
  `grid_response=True`), SAME step as ferric (5e-3 Bohr), symmetrized: the
  identical construction, so the O(delta^2) truncation cancels between the
  two codes and the comparison floor is SCF/grid noise only. This is the
  tight reference for every method, and the ONLY reference for ROHF (PySCF
  has no ROHF Hessian), where it is built from `grad.rohf` and step-converged
  (`fd_step_study`: steps 1e-2, 5e-3, 2.5e-3 and the Richardson estimate).

Frequencies are PySCF `hessian.thermo.harmonic_analysis(mol, H, mass=M,
imaginary_freq=False)` with M = FERRIC's masses (below; the Rust test asserts
they equal `atom_masses(mol)` exactly). Both codes project the SAME subspace
(mass-weighted translations + rotations about the centre of mass; PySCF
expresses the rotations in the principal-axis frame, a rotation of the same
span), PySCF by restricting H to the complement, ferric by P H P — identical
nonzero spectra. So a non-stationary geometry is fine; the geometries are
used AS GIVEN (not optimized). The Rust side proves the projection claim
directly by feeding `hessian_analytic` into ferric's post-processing.

Systems x methods (geometries in testdata/molecules/validation/):

    h2o, nh3  x 6-31G, cc-pVDZ  x RHF, RKS-PBE, RKS-B3LYP
    oh (2Pi), ch3 (2A2''), ho2 (2A'')  x 6-31G  x UHF
    ch3  x 6-31G  x UKS-PBE, ROHF

J/K: EXACT 4-index on both sides (ferric RhfConfig df_j_aux = df_k_aux =
Some(""), the explicit no-fit sentinel — None would auto-enable RI-J for
closed-shell KS; PySCF without density_fit). Grid: (75,110) unpruned,
Becke partition with Becke-1988 radii adjustment (the recipe the KS-energy row
matches to 1e-12 Ha). Open shells: three guesses, each followed through an
internal stability() loop at the reference geometry; every DISPLACED SCF
starts from the reference-geometry density and is itself stability-checked
and <S^2>-checked, so the FD reference follows ONE state; a displaced point
that is unstable is REFUSED (RuntimeError), never written.

Run (light; a few minutes, dominated by the cc-pVDZ B3LYP FD sweeps):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_frequencies.py
Restrict to systems by name:  ... gen_frequencies.py ch3 oh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "frequencies"
ROW_NAME = "Harmonic frequencies"
MAIN_GRID = (75, 110)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FD_STEP = 5.0e-3  # == ferric_scf::frequencies::DEFAULT_DELTA (Rust test asserts it)
ROHF_STEPS = (1.0e-2, 5.0e-3, 2.5e-3)
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10
S2_DRIFT_TOL = 1e-3  # displaced-point <S^2> must stay within this of the centre

# ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 standard weights, u), for
# the elements this row uses. The Rust test asserts atom_masses(mol) equals the
# `masses_amu` written here EXACTLY, so a drift in either table fails there.
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

PYSCF_XC = {"pbe": "PBE,PBE", "b3lyp": "B3LYP"}

CLOSED = {"h2o": (0, 1), "nh3": (0, 1)}
CLOSED_BASES = ("6-31g", "cc-pvdz")
CLOSED_METHODS = ("rhf", "rks_pbe", "rks_b3lyp")

# system -> (charge, multiplicity, methods), all at 6-31G
OPEN = {
    "oh": (0, 2, ("uhf",)),
    "ch3": (0, 2, ("uhf", "uks_pbe", "rohf")),
    "ho2": (0, 2, ("uhf",)),
}
OPEN_BASES = ("6-31g",)


def _make_mf(mol, method: str):
    from pyscf import dft, scf

    if method == "rhf":
        mf = scf.RHF(mol)
    elif method == "uhf":
        mf = scf.UHF(mol)
    elif method == "rohf":
        mf = scf.ROHF(mol)
    else:
        kind, xc = method.split("_")
        mf = {"rks": dft.RKS, "uks": dft.UKS}[kind](mol, xc=PYSCF_XC[xc])
        mf.grids.atom_grid = MAIN_GRID
        mf.grids.prune = None
        # ferric uses Becke (1988) size adjustment, not PySCF's default Treutler.
        mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def _is_ks(method: str) -> bool:
    return method.startswith(("rks", "uks"))


def _is_open(method: str) -> bool:
    return method in ("uhf", "rohf") or method.startswith("uks")


def _stable(mf) -> tuple[bool, object]:
    out = mf.stability(internal=True, external=False, return_status=True)
    mo_i, _mo_e, stable_i, _stable_e = out
    return bool(stable_i), mo_i


def _converge(mf, dm0=None):
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def _follow_to_stable(mf):
    rounds = 0
    while True:
        stable, mo_i = _stable(mf)
        if stable or rounds >= MAX_STAB_ROUNDS:
            return mf, stable, rounds
        mf = _converge(mf, dm0=mf.make_rdm1(mo_i, mf.mo_occ))
        rounds += 1


def _centre_scf(mol, method: str):
    """Converged (and, for open shells, internally stable) SCF at the reference
    geometry. Open shells: lowest stable state over GUESSES."""
    if not _is_open(method):
        mf = _converge(_make_mf(mol, method))
        return mf, None
    best, scan = None, []
    for guess in GUESSES:
        mf = _make_mf(mol, method)
        mf.init_guess = guess
        try:
            mf = _converge(mf)
        except RuntimeError:
            scan.append({"guess": guess, "converged": False})
            continue
        mf, stable, rounds = _follow_to_stable(mf)
        scan.append(
            {
                "guess": guess,
                "energy": float(mf.e_tot),
                "internal_stable": stable,
                "stability_rounds": rounds,
            }
        )
        if stable and (best is None or mf.e_tot < best[0].e_tot):
            best = (mf, guess, rounds)
    if best is None:
        raise RuntimeError(f"{method}: no stable state from any guess: {scan}")
    mf, guess, rounds = best
    stable_es = [g["energy"] for g in scan if g.get("internal_stable")]
    state = {
        "internal_stable": True,
        "kind": f"PySCF {method.upper()} internal (real)",
        "selected_guess": guess,
        "rounds": rounds,
        "guess_scan": scan,
        "multiple_stable_minima": (max(stable_es) - min(stable_es)) > 1e-6,
    }
    if method != "rohf":
        state["s_squared"] = float(mf.spin_square()[0])
    return mf, state


def _gradient(mf, method: str):
    g = mf.nuc_grad_method()
    if _is_ks(method):
        g.grid_response = True
    return g.kernel()


def _displaced_mol(mol, coords_bohr):
    m = mol.copy()
    m.set_geom_(coords_bohr, unit="Bohr")
    return m


def _fd_hessian(mol, method: str, mf0, step: float, s2_ref):
    """Central FD of the analytic gradient, each displaced SCF started from
    the centre density and checked to be on the same (stable) state."""
    import numpy as np

    natm = mol.natm
    n = 3 * natm
    x0 = mol.atom_coords(unit="Bohr")
    dm0 = mf0.make_rdm1()
    h = np.zeros((n, n))
    worst_s2 = 0.0
    worst_2nd_diff = 0.0
    for b in range(n):
        es = []
        grads = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * step
            m = _displaced_mol(mol, x)
            mf = _converge(_make_mf(m, method), dm0=dm0)
            if _is_open(method):
                mf, stable, _ = _follow_to_stable(mf)
                if not stable:
                    raise RuntimeError(
                        f"{method}: displaced point coord {b} sign {sgn:+} is UNSTABLE"
                    )
                if s2_ref is not None:
                    d = abs(mf.spin_square()[0] - s2_ref)
                    worst_s2 = max(worst_s2, d)
                    if d > S2_DRIFT_TOL:
                        raise RuntimeError(
                            f"{method}: displaced point coord {b} <S^2> drifted by {d:.2e}"
                        )
            es.append(mf.e_tot)
            grads.append(_gradient(mf, method).ravel())
        h[:, b] = (grads[0] - grads[1]) / (2.0 * step)
        worst_2nd_diff = max(worst_2nd_diff, abs(es[0] + es[1] - 2.0 * mf0.e_tot))
    asym = float(np.max(np.abs(h - h.T)))
    h = 0.5 * (h + h.T)
    diag = {
        "fd_step_bohr": step,
        "fd_asymmetry_max": asym,
        "fd_max_energy_second_difference": worst_2nd_diff,
        "fd_n_gradients": 2 * n,
    }
    if s2_ref is not None:
        diag["fd_max_s2_drift"] = worst_s2
    return h, diag


def _analytic_hessian(mf):
    from pyscf import hessian  # noqa: F401  (registers mf.Hessian)

    hobj = mf.Hessian()
    e2 = hobj.kernel()
    natm = mf.mol.natm
    return e2.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _frequencies(mol, h, masses):
    import numpy as np
    from pyscf.hessian import thermo

    natm = mol.natm
    h4 = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(
        mol, h4, mass=np.asarray(masses), imaginary_freq=False
    )
    return sorted(float(v) for v in np.asarray(res["freq_wavenumber"]).real)


def _method_block(mol, method: str, masses) -> dict:
    import numpy as np

    mf0, state = _centre_scf(mol, method)
    s2_ref = state.get("s_squared") if state else None
    block = {"energy": float(mf0.e_tot), "converged": True}
    if state is not None:
        block["stability"] = state
        if s2_ref is not None:
            block["s_squared"] = s2_ref
    block["gradient"] = np.asarray(_gradient(mf0, method)).tolist()

    if method == "rohf":
        study = {}
        hs = {}
        for step in ROHF_STEPS:
            hs[step], diag = _fd_hessian(mol, method, mf0, step, None)
            study[f"{step:.1e}"] = diag
        h_ref = hs[FD_STEP]
        # Richardson (O(delta^2) error): H* = (4 H(d/2) - H(d)) / 3, d = FD_STEP.
        h_rich = (4.0 * hs[2.5e-3] - hs[5.0e-3]) / 3.0
        study["max_abs_H(1e-2)-H(5e-3)"] = float(np.max(np.abs(hs[1e-2] - hs[5e-3])))
        study["max_abs_H(5e-3)-H(2.5e-3)"] = float(
            np.max(np.abs(hs[5e-3] - hs[2.5e-3]))
        )
        study["max_abs_H(5e-3)-richardson"] = float(np.max(np.abs(h_ref - h_rich)))
        study["freq_richardson_cm"] = _frequencies(mol, h_rich, masses)
        block["hessian_fd"] = h_ref.tolist()
        block["freq_fd_cm"] = _frequencies(mol, h_ref, masses)
        block["hessian_richardson"] = h_rich.tolist()
        block["fd_step_study"] = study
        block["fd"] = study[f"{FD_STEP:.1e}"]
        block["hessian_analytic"] = None
        block["analytic_note"] = "PySCF has no ROHF Hessian; FD of grad.rohf only"
        return block

    h_an_raw = _analytic_hessian(mf0)
    # PySCF's analytic KS Hessian is not exactly symmetric (2.4e-7 Ha/Bohr^2
    # for B3LYP/6-31G water), and harmonic_analysis's eigh reads one
    # triangle. ferric symmetrizes before diagonalizing, so the reference
    # stores the symmetrized Hessian and records the asymmetry it removed.
    h_an = 0.5 * (h_an_raw + h_an_raw.T)
    block["hessian_analytic_asymmetry"] = float(np.max(np.abs(h_an_raw - h_an_raw.T)))
    h_fd, diag = _fd_hessian(mol, method, mf0, FD_STEP, s2_ref)
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    block.update(
        {
            "hessian_analytic": h_an.tolist(),
            "freq_analytic_cm": f_an,
            "hessian_fd": h_fd.tolist(),
            "freq_fd_cm": f_fd,
            "fd": diag,
            # PySCF-internal gap between its two constructions. HF: pure FD
            # truncation + noise. KS: that PLUS the grid-response term the
            # analytic KS Hessian omits.
            "pyscf_fd_vs_analytic_hessian_max_abs": float(np.max(np.abs(h_fd - h_an))),
            "pyscf_fd_vs_analytic_freq_max_abs_cm": float(
                max(abs(a - b) for a, b in zip(f_fd, f_an))
            ),
        }
    )
    return block


def _run(system, charge, mult, basis_name, methods) -> Path:
    import numpy
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]
    blocks = {}
    for method in methods:
        blocks[method] = _method_block(mol, method, masses)
        b = blocks[method]
        extra = (
            f"fd-vs-an freq {b['pyscf_fd_vs_analytic_freq_max_abs_cm']:.3f} cm-1"
            if b.get("hessian_analytic") is not None
            else f"rohf H(5e-3)-rich {b['fd_step_study']['max_abs_H(5e-3)-richardson']:.2e}"
        )
        print(
            f"{system:5s} {basis_name:8s} {method:10s} E {b['energy']:.10f} "
            f"freqs {['%.1f' % f for f in b['freq_fd_cm']]} | {extra}",
            flush=True,
        )
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "frequency_convention": (
            "harmonic_analysis(mass=masses_amu, imaginary_freq=False): ascending, "
            "imaginary reported as negative, translations+rotations projected out"
        ),
        **blocks,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "methods": list(methods),
                "xc": {k: PYSCF_XC[k.split("_")[1]] for k in methods if _is_ks(k)},
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "max_cycle": 500,
                "eri": "exact 4-index (no density fitting)",
                "analytic_hessian": "pyscf.hessian.{rhf,rks,uhf,uks} (KS: no grid response)",
                "fd_hessian": (
                    f"central FD of analytic gradient, step {FD_STEP} Bohr "
                    "(KS gradient grid_response=True), displaced SCFs from the centre "
                    "density, symmetrized"
                ),
                "rohf_fd_steps": list(ROHF_STEPS),
                "masses": "ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 averages)",
                "numpy": numpy.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "radii_adjust": "becke_atomic_radii_adjust",
                "applies_to": "KS methods only",
            },
            aux=None,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability={m: blocks[m].get("stability") for m in methods if _is_open(m)}
            or None,
            generator="scripts/validation/gen_frequencies.py",
            extra={"basis_self_check": basis_check},
        ),
    }
    return common.write_reference(ROW, system, basis_name, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in CLOSED.items():
        if only and system not in only:
            continue
        for basis_name in CLOSED_BASES:
            written.append(_run(system, charge, mult, basis_name, CLOSED_METHODS))
    for system, (charge, mult, methods) in OPEN.items():
        if only and system not in only:
            continue
        for basis_name in OPEN_BASES:
            written.append(_run(system, charge, mult, basis_name, methods))
    print(f"GEN_FREQUENCIES_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
